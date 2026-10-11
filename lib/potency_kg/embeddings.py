"""Scoped immutable embedding persistence and exact vector retrieval."""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from typing import TypeVar
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .contracts import Scope, ScopedPassageIdentity
from .db import connection_scope
from .embedding_client import EmbeddingBatch, EmbeddingClient, EmbeddingInputError
from .ids import derive_embedding_version, derive_index_version


_T = TypeVar("_T")


@dataclass(frozen=True)
class EmbeddingReport:
    embedding_version: str
    index_version: str
    active_passages: int
    embedded_passages: int
    reused_passages: int


@dataclass(frozen=True)
class QueryEmbedding:
    vector: tuple[float, ...]
    model: str
    dimensions: int
    normalization: str


@dataclass(frozen=True)
class PassageHit:
    source_id: str
    source_version: str
    passage_id: str
    score: float


def _normalize_input(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _input_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _embedding_version(client: EmbeddingClient) -> str:
    identity = client.identity
    return derive_embedding_version(
        identity.model,
        identity.dimensions,
        identity.encoding_format,
        identity.normalization,
    )


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _active_rows(connection: Connection, scope: Scope) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in connection.execute(
            text(
                """
                SELECT passage.source_id, passage.source_version, passage.passage_id,
                       passage.text, passage.text_sha256
                FROM source_passage AS passage
                JOIN source_head AS head
                  ON (head.tenant_id, head.project_id, head.source_id, head.source_version)
                   = (passage.tenant_id, passage.project_id, passage.source_id, passage.source_version)
                JOIN source_document AS document
                  ON (document.tenant_id, document.project_id, document.source_id, document.source_version)
                   = (passage.tenant_id, passage.project_id, passage.source_id, passage.source_version)
                WHERE passage.tenant_id=:tenant AND passage.project_id=:project
                  AND document.expected_passage_count = (
                    SELECT count(*)
                    FROM source_passage AS candidate
                    WHERE (candidate.tenant_id, candidate.project_id, candidate.source_id, candidate.source_version)
                        = (document.tenant_id, document.project_id, document.source_id, document.source_version)
                  )
                ORDER BY passage.source_id, passage.source_version, passage.passage_id
                """
            ),
            {"tenant": scope.tenant_id, "project": scope.project_id},
        ).mappings()
    ]


def _existing_binding(connection: Connection, scope: Scope, row: dict[str, object], embedding_version: str, input_sha256: str) -> UUID | None:
    return connection.execute(
        text(
            """
            SELECT run.embedding_run_id
            FROM passage_embedding AS binding
            JOIN embedding_run AS run
              ON (run.embedding_run_id, run.tenant_id, run.project_id, run.source_id, run.source_version, run.passage_id)
               = (binding.embedding_run_id, binding.tenant_id, binding.project_id, binding.source_id, binding.source_version, binding.passage_id)
            WHERE binding.tenant_id=:tenant AND binding.project_id=:project
              AND binding.source_id=:source_id AND binding.source_version=:source_version
              AND binding.passage_id=:passage_id AND run.embedding_version=:embedding_version
              AND run.input_sha256=:input_sha256 AND run.status='succeeded' AND run.vector IS NOT NULL
            ORDER BY run.created_at DESC
            LIMIT 1
            """
        ),
        {
            "tenant": scope.tenant_id,
            "project": scope.project_id,
            "source_id": row["source_id"],
            "source_version": row["source_version"],
            "passage_id": row["passage_id"],
            "embedding_version": embedding_version,
            "input_sha256": input_sha256,
        },
    ).scalar_one_or_none()


def _cached_vector(connection: Connection, scope: Scope, embedding_version: str, input_sha256: str) -> Sequence[float] | None:
    value = connection.execute(
        text(
            """
            SELECT vector FROM embedding_run
            WHERE tenant_id=:tenant AND project_id=:project AND embedding_version=:embedding_version
              AND input_sha256=:input_sha256 AND status='succeeded' AND vector IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id, "embedding_version": embedding_version, "input_sha256": input_sha256},
    ).scalar_one_or_none()
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(float(item) for item in value.strip("[]").split(",") if item)
    return tuple(float(item) for item in value)


def _insert_run(
    connection: Connection,
    scope: Scope,
    row: dict[str, object],
    embedding_version: str,
    client: EmbeddingClient,
    input_sha256: str,
    vector: Sequence[float],
    batch: EmbeddingBatch | None,
) -> UUID:
    run_id = uuid4()
    identity = client.identity
    connection.execute(
        text(
            """
            INSERT INTO embedding_run
              (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id,
               embedding_version, model, sanitized_origin, credential_env, dimensions,
               encoding_format, normalization, input_sha256, vector, usage, status)
            VALUES
              (:run_id, :tenant, :project, :source_id, :source_version, :passage_id,
               :embedding_version, :model, :origin, :credential_env, :dimensions,
               :encoding_format, :normalization, :input_sha256, CAST(:vector AS vector),
               CAST(:usage AS jsonb), 'succeeded')
            """
        ),
        {
            "run_id": run_id,
            "tenant": scope.tenant_id,
            "project": scope.project_id,
            "source_id": row["source_id"],
            "source_version": row["source_version"],
            "passage_id": row["passage_id"],
            "embedding_version": embedding_version,
            "model": identity.model,
            "origin": identity.sanitized_origin,
            "credential_env": identity.credential_env,
            "dimensions": identity.dimensions,
            "encoding_format": identity.encoding_format,
            "normalization": identity.normalization,
            "input_sha256": input_sha256,
            "vector": _vector_literal(vector),
            "usage": json.dumps({"prompt_tokens": batch.prompt_tokens, "total_tokens": batch.total_tokens} if batch else {}),
        },
    )
    connection.execute(
        text(
            """
            INSERT INTO passage_embedding
              (tenant_id, project_id, source_id, source_version, passage_id, embedding_run_id)
            VALUES (:tenant, :project, :source_id, :source_version, :passage_id, :run_id)
            """
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id, "source_id": row["source_id"], "source_version": row["source_version"], "passage_id": row["passage_id"], "run_id": run_id},
    )
    return run_id


def _chunks(values: list[_T], size: int) -> Iterator[list[_T]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _validated_batch(
    client: EmbeddingClient,
    batch: EmbeddingBatch,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    identity = client.identity
    if (
        identity.dimensions != 3072
        or identity.encoding_format != "float"
        or len(batch.vectors) != expected_count
        or batch.model != identity.model
        or batch.dimensions != identity.dimensions
        or batch.encoding_format != identity.encoding_format
        or batch.normalization != identity.normalization
    ):
        raise EmbeddingInputError("embedding client returned incompatible batch")
    try:
        vectors = tuple(tuple(float(value) for value in vector) for vector in batch.vectors)
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingInputError("embedding client returned incompatible batch") from None
    if any(len(vector) != identity.dimensions for vector in vectors) or any(
        not math.isfinite(value) for vector in vectors for value in vector
    ):
        raise EmbeddingInputError("embedding client returned incompatible batch")
    return vectors


def embed_active_passages(engine: Engine, scope: Scope, client: EmbeddingClient) -> EmbeddingReport:
    """Embed each complete active passage, then atomically snapshot exact membership."""
    embedding_version = _embedding_version(client)
    embedded = reused = 0
    with connection_scope(engine) as connection:
        rows = _active_rows(connection, scope)
        bindings: dict[tuple[str, str, str], UUID] = {}
        misses_by_hash: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            embedding_input = _normalize_input(str(row["text"]))
            input_sha256 = _input_hash(embedding_input)
            row["embedding_input"] = embedding_input
            row["input_sha256"] = input_sha256
            existing = _existing_binding(connection, scope, row, embedding_version, input_sha256)
            if existing is not None:
                bindings[(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))] = existing
                continue
            cached = _cached_vector(connection, scope, embedding_version, input_sha256)
            if cached is None:
                misses_by_hash.setdefault(input_sha256, []).append(row)
                continue
            cached_batch = EmbeddingBatch(
                model=client.identity.model,
                vectors=(tuple(cached),),
                dimensions=client.identity.dimensions,
                encoding_format=client.identity.encoding_format,
                normalization=client.identity.normalization,
                prompt_tokens=0,
                total_tokens=0,
            )
            vector = _validated_batch(client, cached_batch, 1)[0]
            run_id = _insert_run(connection, scope, row, embedding_version, client, input_sha256, vector, None)
            bindings[(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))] = run_id
            reused += 1

        for chunk in _chunks(list(misses_by_hash.values()), 64):
            batch = client.embed([str(group[0]["embedding_input"]) for group in chunk])
            vectors = _validated_batch(client, batch, len(chunk))
            for group, vector in zip(chunk, vectors, strict=True):
                for position, row in enumerate(group):
                    run_id = _insert_run(
                        connection,
                        scope,
                        row,
                        embedding_version,
                        client,
                        str(row["input_sha256"]),
                        vector,
                        batch if position == 0 else None,
                    )
                    bindings[(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))] = run_id
                embedded += 1
                reused += len(group) - 1

        member_rows = [
            (str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]), str(row["text_sha256"]), bindings[(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))])
            for row in rows
        ]
        index_version = derive_index_version(
            embedding_version,
            [
                (
                    ScopedPassageIdentity(
                        scope.tenant_id,
                        scope.project_id,
                        source_id,
                        source_version,
                        passage_id,
                    ),
                    text_hash,
                )
                for source_id, source_version, passage_id, text_hash, _run_id in member_rows
            ],
        )
        connection.execute(
            text(
                """
                INSERT INTO index_snapshot (index_version, tenant_id, project_id, embedding_version)
                VALUES (:index_version, :tenant, :project, :embedding_version)
                ON CONFLICT (tenant_id, project_id, index_version) DO NOTHING
                """
            ),
            {"index_version": index_version, "tenant": scope.tenant_id, "project": scope.project_id, "embedding_version": embedding_version},
        )
        for source_id, source_version, passage_id, _text_hash, run_id in member_rows:
            connection.execute(
                text(
                    """
                    INSERT INTO index_member
                      (index_version, tenant_id, project_id, source_id, source_version, passage_id, embedding_run_id)
                    VALUES (:index_version, :tenant, :project, :source_id, :source_version, :passage_id, :run_id)
                    ON CONFLICT (index_version, tenant_id, project_id, source_id, source_version, passage_id) DO NOTHING
                    """
                ),
                {"index_version": index_version, "tenant": scope.tenant_id, "project": scope.project_id, "source_id": source_id, "source_version": source_version, "passage_id": passage_id, "run_id": run_id},
            )
    return EmbeddingReport(embedding_version, index_version, len(rows), embedded, reused)


def embed_query(client: EmbeddingClient, text: str) -> QueryEmbedding:
    batch = client.embed([text])
    try:
        vector = _validated_batch(client, batch, 1)[0]
    except EmbeddingInputError:
        raise EmbeddingInputError("embedding client returned incompatible query batch") from None
    return QueryEmbedding(vector, batch.model, batch.dimensions, batch.normalization)


def vector_search(connection: Connection, scope: Scope, index_version: str, query_vector: Sequence[float], limit: int) -> list[PassageHit]:
    if limit < 1:
        return []
    try:
        vector = tuple(float(value) for value in query_vector)
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingInputError("query vector must contain finite values") from None
    if len(vector) != 3072 or any(not math.isfinite(value) for value in vector):
        raise EmbeddingInputError("query vector must contain finite values")
    rows = connection.execute(
        text(
            """
            SELECT member.source_id, member.source_version, member.passage_id,
                   1 - (run.vector <=> CAST(:query AS vector)) AS score
            FROM index_member AS member
            JOIN embedding_run AS run
              ON (run.embedding_run_id, run.tenant_id, run.project_id, run.source_id, run.source_version, run.passage_id)
               = (member.embedding_run_id, member.tenant_id, member.project_id, member.source_id, member.source_version, member.passage_id)
            WHERE member.tenant_id=:tenant AND member.project_id=:project AND member.index_version=:index_version
            ORDER BY run.vector <=> CAST(:query AS vector), member.source_id, member.source_version, member.passage_id
            LIMIT :limit
            """
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id, "index_version": index_version, "query": _vector_literal(vector), "limit": limit},
    ).mappings()
    return [PassageHit(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]), float(row["score"])) for row in rows]
