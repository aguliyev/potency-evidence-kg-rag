"""Deterministic vector-seeded, bounded graph retrieval trace."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .age_projection import active_graph_name, bounded_graph_traversal
from .contracts import Scope
from .embedding_client import EmbeddingClient
from .embeddings import PassageHit, embed_query, validate_index_membership, vector_search
from .ids import derive_embedding_version, derive_scoped_id


RETRIEVER_VERSION = "vector-seeded-age-v1"


@dataclass(frozen=True)
class RetrievalLimits:
    vector_hits: int = 8
    semantic_seeds: int = 8
    hops: int = 2
    nodes: int = 30
    edges: int = 50

    def __post_init__(self) -> None:
        values = (self.vector_hits, self.semantic_seeds, self.hops, self.nodes, self.edges)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("retrieval limits must be nonnegative integers")
        if (
            self.vector_hits > 8
            or self.semantic_seeds > 8
            or self.hops > 2
            or self.nodes > 30
            or self.edges > 50
        ):
            raise ValueError("retrieval limits exceed the bounded smoke-retrieval maxima")

    def as_dict(self) -> dict[str, int]:
        return {
            "vector_hits": self.vector_hits,
            "semantic_seeds": self.semantic_seeds,
            "hops": self.hops,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    @property
    def max_hops(self) -> int:
        return self.hops

    @property
    def max_nodes(self) -> int:
        return self.nodes

    @property
    def max_edges(self) -> int:
        return self.edges


@dataclass(frozen=True)
class RetrievedSourceRef:
    tenant_id: str
    project_id: str
    source_id: str
    source_version: str
    passage_id: str
    text: str
    text_sha256: str
    locator: dict[str, object]

    def as_tuple(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.tenant_id,
            self.project_id,
            self.source_id,
            self.source_version,
            self.passage_id,
            self.text,
            self.text_sha256,
        )


@dataclass(frozen=True)
class RetrievalTrace:
    retriever_version: str
    query_hash: str
    embedding_version: str
    index_version: str
    graph_version: str
    graph_name: str
    limits: RetrievalLimits
    vector_hits: tuple[PassageHit, ...]
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    source_refs: tuple[RetrievedSourceRef, ...]

    @property
    def ordered_vector_hits(self) -> tuple[PassageHit, ...]:
        return self.vector_hits

    @property
    def traversed_edge_ids(self) -> tuple[str, ...]:
        return self.edge_ids

    @property
    def selected_source_refs(self) -> tuple[RetrievedSourceRef, ...]:
        return self.source_refs


@dataclass(frozen=True)
class _ActiveGraphBinding:
    graph_version: str
    graph_name: str
    index_version: str
    embedding_version: str


def _active_graph_binding(
    connection: Connection,
    scope: Scope,
    index_version: str,
    expected_embedding_version: str,
) -> _ActiveGraphBinding:
    graph = connection.execute(
        text(
            """SELECT graph_version,active_graph_name,index_version,embedding_version
               FROM graph_scope WHERE tenant_id=:tenant AND project_id=:project"""
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id},
    ).mappings().one_or_none()
    if graph is None:
        raise LookupError("scope has no active graph")
    if graph["index_version"] is None or str(graph["index_version"]) != index_version:
        raise ValueError("requested index is not bound to the active graph")
    binding = validate_index_membership(connection, scope, index_version)
    if (
        binding.embedding_version != str(graph["embedding_version"])
        or binding.embedding_version != expected_embedding_version
    ):
        raise ValueError("query adapter does not match the index embedding version")
    return _ActiveGraphBinding(
        str(graph["graph_version"]),
        str(graph["active_graph_name"]),
        binding.index_version,
        binding.embedding_version,
    )


def _source_rows(connection, scope: Scope, hits: list[PassageHit]) -> tuple[RetrievedSourceRef, ...]:
    result: list[RetrievedSourceRef] = []
    for hit in hits:
        row = connection.execute(
            text(
                """SELECT text,text_sha256,locator FROM source_passage
                   WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                     AND source_version=:source_version AND passage_id=:passage_id"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "source_id": hit.source_id,
                "source_version": hit.source_version,
                "passage_id": hit.passage_id,
            },
        ).mappings().one_or_none()
        if row is None:
            raise LookupError("vector hit has no exact scoped relational passage")
        result.append(
            RetrievedSourceRef(
                scope.tenant_id,
                scope.project_id,
                hit.source_id,
                hit.source_version,
                hit.passage_id,
                str(row["text"]),
                str(row["text_sha256"]),
                dict(row["locator"] or {}),
            )
        )
    return tuple(result)


def _semantic_seed_candidates(
    connection,
    scope: Scope,
    hits: list[PassageHit],
    limit: int,
) -> tuple[str, ...]:
    if limit == 0:
        return ()
    seeds: list[str] = []
    for hit in hits:
        candidate_ids = connection.execute(
            text(
                """SELECT ref.node_id FROM kg_node_source_ref AS ref
                   JOIN kg_node AS node
                     ON (node.tenant_id,node.project_id,node.node_id) =
                        (ref.tenant_id,ref.project_id,ref.node_id)
                   WHERE ref.tenant_id=:tenant AND ref.project_id=:project
                     AND ref.source_id=:source_id AND ref.source_version=:source_version
                     AND ref.passage_id=:passage_id
                     AND node.node_type NOT IN ('SourceDocument','SourcePassage')
                   ORDER BY ref.node_id LIMIT :limit"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "source_id": hit.source_id,
                "source_version": hit.source_version,
                "passage_id": hit.passage_id,
                "limit": limit,
            },
        ).scalars()
        for value in candidate_ids:
            node_id = str(value)
            if node_id not in seeds:
                seeds.append(node_id)
                if len(seeds) >= limit:
                    return tuple(seeds)
    return tuple(seeds)


def retrieve_graph_smoke(
    engine: Engine,
    scope: Scope,
    query_text: str,
    embedding_client: EmbeddingClient,
    index_version: str,
    limits: RetrievalLimits,
) -> RetrievalTrace:
    """Embed one query, search an exact index, and return evidence plus a bounded trace."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query text must be nonempty")
    expected_embedding_version = derive_embedding_version(
        embedding_client.identity.model,
        embedding_client.identity.dimensions,
        embedding_client.identity.encoding_format,
        embedding_client.identity.normalization,
    )
    # This is deliberately an unlocked, short database preflight. The provider
    # call below must not retain a transaction or block graph activation.
    with engine.connect() as connection:
        preflight = _active_graph_binding(
            connection, scope, index_version, expected_embedding_version
        )
    query = embed_query(embedding_client, query_text)
    with engine.begin() as connection:
        connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        graph_name = active_graph_name(connection, scope)
        current = _active_graph_binding(
            connection, scope, index_version, expected_embedding_version
        )
        if current.graph_name != graph_name or current != preflight:
            raise ValueError("active graph binding changed during query embedding")
        hits = vector_search(connection, scope, index_version, query.vector, limits.vector_hits)
        source_refs = _source_rows(connection, scope, hits)
        passage_ids = tuple(
            derive_scoped_id("passage", scope, hit.source_id, hit.source_version, hit.passage_id)
            for hit in hits
        )
        seed_candidates = _semantic_seed_candidates(
            connection, scope, hits, limits.semantic_seeds
        )
        traversal = bounded_graph_traversal(
            connection,
            graph_name,
            passage_ids,
            seed_candidates,
            semantic_seed_limit=limits.semantic_seeds,
            hops=limits.hops,
            node_limit=limits.nodes,
            edge_limit=limits.edges,
        )
    return RetrievalTrace(
        RETRIEVER_VERSION,
        sha256(query_text.encode("utf-8")).hexdigest(),
        expected_embedding_version,
        index_version,
        current.graph_version,
        graph_name,
        limits,
        tuple(hits),
        traversal.seed_node_ids,
        traversal.node_ids,
        traversal.edge_ids,
        source_refs,
    )
