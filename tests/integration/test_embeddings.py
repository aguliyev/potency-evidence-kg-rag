from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import pytest
from pydantic import SecretStr
from sqlalchemy import text

from potency_kg.contracts import Scope, ScopedPassageIdentity
import potency_kg.embedding_client as embedding_client_module
from potency_kg.embedding_client import (
    EmbeddingBatch,
    EmbeddingClientIdentity,
    EmbeddingInputError,
    EmbeddingProtocolError,
    OpenAIEmbeddingClient,
)
from potency_kg.embeddings import embed_active_passages, embed_query, vector_search
from potency_kg.ids import derive_embedding_version, derive_index_version
from potency_kg.settings import load_embedding_settings


DIMENSIONS = 3072
SCOPE_ONE = Scope("tenant", "project-one")
SCOPE_TWO = Scope("tenant", "project-two")


def _settings():
    return load_embedding_settings(
        {
            "EMBEDDING_LLM_URI": "https://api.openai.com/v1",
            "EMBEDDING_LLM_MODEL": "text-embedding-3-large",
            "EMBEDDING_LLM_API_KEY": "test-key",
        }
    )


def _response(items: list[tuple[int, Sequence[float]]]):
    return SimpleNamespace(
        model="text-embedding-3-large",
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items],
        usage=SimpleNamespace(prompt_tokens=11, total_tokens=13),
    )


class _RecordingSDK:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@dataclass
class _DeterministicClient:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity(
        provider="test-double",
        model="test-model",
        dimensions=DIMENSIONS,
        encoding_format="float",
        normalization="provider_unit_length_v1",
        sanitized_origin="https://test.invalid/v1",
        credential_env="EMBEDDING_LLM_API_KEY",
    )
    calls: list[tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        self.calls = [] if self.calls is None else self.calls

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        values = tuple(texts)
        self.calls.append(values)
        vectors = tuple(tuple(float(index + 1) for _ in range(DIMENSIONS)) for index, _ in enumerate(values))
        return EmbeddingBatch(
            model=self.identity.model,
            vectors=vectors,
            dimensions=DIMENSIONS,
            encoding_format="float",
            normalization="provider_unit_length_v1",
            prompt_tokens=len(values),
            total_tokens=len(values),
        )


def _insert_active_passage(connection, scope: Scope, version: str, passage_id: str, body: str) -> None:
    _insert_active_document(connection, scope, version, [(passage_id, body)])


def _insert_active_document(
    connection,
    scope: Scope,
    version: str,
    passages: list[tuple[str, str]],
    *,
    expected_count: int | None = None,
) -> None:
    document = {
        "tenant": scope.tenant_id,
        "project": scope.project_id,
        "version": version,
        "sha": f"sha-{version}",
        "expected": len(passages) if expected_count is None else expected_count,
    }
    connection.execute(text("INSERT INTO source_document (tenant_id, project_id, source_id, source_version, title, source_sha256, expected_passage_count, status) VALUES (:tenant, :project, 'source', :version, 'title', :sha, :expected, 'final')"), document)
    for passage_id, body in passages:
        params = {**document, "passage": passage_id, "body": body, "text_sha": f"sha-{version}-{passage_id}"}
        connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES (:tenant, :project, 'source', :version, :passage, :body, :text_sha)"), params)
        connection.execute(text("INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason) VALUES (:tenant, :project, 'source', :version, :passage, 'selected', 'jev_bundle_selected')"), params)
    connection.execute(text("INSERT INTO source_head (tenant_id, project_id, source_id, source_version) VALUES (:tenant, :project, 'source', :version) ON CONFLICT (tenant_id, project_id, source_id) DO UPDATE SET source_version=EXCLUDED.source_version"), document)


def test_adapter_orders_and_validates_openai_response_without_secret_leakage() -> None:
    """Returning SDK order, malformed indexes, or credentials would break this adapter boundary."""
    first = [1.0] * DIMENSIONS
    second = [2.0] * DIMENSIONS
    sdk = _RecordingSDK(_response([(1, second), (0, first)]))
    client = OpenAIEmbeddingClient(_settings(), SecretStr("super-secret"), client=sdk)

    batch = client.embed(["first", "second"])

    assert sdk.calls == [{"model": "text-embedding-3-large", "input": ["first", "second"], "encoding_format": "float", "dimensions": 3072}]
    assert batch.vectors == (tuple(first), tuple(second))
    assert (batch.model, batch.prompt_tokens, batch.total_tokens) == ("text-embedding-3-large", 11, 13)
    assert "super-secret" not in repr(client)
    assert "super-secret" not in repr(batch)


def test_adapter_constructs_the_official_openai_client_with_only_sanitized_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Using another endpoint, timeout, or exposed key would send the real request incorrectly."""
    created: list[dict[str, object]] = []
    sdk = _RecordingSDK(_response([(0, [1.0] * DIMENSIONS)]))

    def build_client(**kwargs):
        created.append(kwargs)
        return sdk

    monkeypatch.setattr(embedding_client_module, "OpenAI", build_client)
    client = OpenAIEmbeddingClient(_settings(), SecretStr("construction-secret"))
    client.embed(["one"])

    assert created == [{"base_url": "https://api.openai.com/v1", "api_key": "construction-secret", "timeout": 60.0}]
    assert sdk.calls == [{"model": "text-embedding-3-large", "input": ["one"], "encoding_format": "float", "dimensions": 3072}]


def test_adapter_redacts_provider_failure_details() -> None:
    """Persisting or surfacing a provider exception must not disclose a credential."""
    sdk = _RecordingSDK(_response([]))

    def fail(**kwargs):
        raise RuntimeError("provider rejected super-secret")

    sdk.embeddings = SimpleNamespace(create=fail)
    client = OpenAIEmbeddingClient(_settings(), SecretStr("super-secret"), client=sdk)
    with pytest.raises(EmbeddingProtocolError) as raised:
        client.embed(["one"])
    assert "super-secret" not in str(raised.value)


def test_adapter_redacts_client_construction_failure_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Letting SDK construction errors escape could disclose the resolved credential."""
    def fail(**kwargs):
        raise RuntimeError(f"provider rejected {kwargs['api_key']}")

    monkeypatch.setattr(embedding_client_module, "OpenAI", fail)
    with pytest.raises(EmbeddingProtocolError) as raised:
        OpenAIEmbeddingClient(_settings(), SecretStr("super-secret"))
    assert "super-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("texts", "items", "error"),
    [
        ([], [], EmbeddingInputError),
        ([" "], [], EmbeddingInputError),
        (["one"], [], EmbeddingProtocolError),
        (["one", "two"], [(0, [1.0] * DIMENSIONS), (0, [2.0] * DIMENSIONS)], EmbeddingProtocolError),
        (["one"], [(0, [1.0] * (DIMENSIONS - 1))], EmbeddingProtocolError),
        (["one"], [(0, [float("nan")] * DIMENSIONS)], EmbeddingProtocolError),
        (["one"], [(0, ["super-secret"] * DIMENSIONS)], EmbeddingProtocolError),
    ],
)
def test_adapter_rejects_invalid_inputs_and_protocol_responses(texts, items, error) -> None:
    """Accepting malformed request/response data would persist unusable vectors."""
    client = OpenAIEmbeddingClient(_settings(), SecretStr("super-secret"), client=_RecordingSDK(_response(items)))
    with pytest.raises(error) as raised:
        client.embed(texts)
    assert "super-secret" not in str(raised.value)


def test_adapter_enforces_the_configured_64_input_limit() -> None:
    """An oversized provider request must fail locally before it leaves the process."""
    client = OpenAIEmbeddingClient(_settings(), SecretStr("super-secret"), client=_RecordingSDK(_response([])))
    with pytest.raises(EmbeddingInputError, match="64"):
        client.embed(["text"] * 65)


def test_query_embedding_uses_the_same_adapter_once() -> None:
    """Routing query embeddings through another path would allow incompatible vectors."""
    client = _DeterministicClient()
    query = embed_query(client, "query")
    assert client.calls == [("query",)]
    assert query.vector == tuple([1.0] * DIMENSIONS)
    assert query.model == "test-model"


def test_active_passages_are_cached_per_scope_and_replacement_gets_new_membership(db_engine) -> None:
    """Skipping active rows, sharing another project cache, or reusing old membership must fail."""
    with db_engine.begin() as connection:
        _insert_active_passage(connection, SCOPE_ONE, "v1", "p1", "same text")
        _insert_active_passage(connection, SCOPE_TWO, "v1", "p1", "same text")
    client = _DeterministicClient()

    first = embed_active_passages(db_engine, SCOPE_ONE, client)
    second = embed_active_passages(db_engine, SCOPE_ONE, client)
    other_project = embed_active_passages(db_engine, SCOPE_TWO, client)

    with db_engine.begin() as connection:
        connection.execute(text("UPDATE passage_processing SET reason='inactive_source_version' WHERE tenant_id=:tenant AND project_id=:project AND source_version='v1'"), {"tenant": SCOPE_ONE.tenant_id, "project": SCOPE_ONE.project_id})
        _insert_active_passage(connection, SCOPE_ONE, "v2", "p1", "same text")
    correction = embed_active_passages(db_engine, SCOPE_ONE, client)

    assert len(client.calls) == 2
    assert (first.embedded_passages, first.reused_passages) == (1, 0)
    assert (second.embedded_passages, second.reused_passages) == (0, 0)
    assert other_project.embedded_passages == 1
    assert correction.reused_passages == 1
    expected_embedding_version = derive_embedding_version(
        "test-model", DIMENSIONS, "float", "provider_unit_length_v1"
    )
    expected_first_index = derive_index_version(
        expected_embedding_version,
        [(ScopedPassageIdentity("tenant", "project-one", "source", "v1", "p1"), "sha-v1-p1")],
    )
    assert first.embedding_version == expected_embedding_version
    assert first.index_version == expected_first_index
    assert other_project.index_version != first.index_version
    assert correction.index_version != first.index_version
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM embedding_run WHERE tenant_id=:tenant AND project_id=:project"), {"tenant": SCOPE_ONE.tenant_id, "project": SCOPE_ONE.project_id}).scalar_one() == 2
        assert connection.execute(text("SELECT count(*) FROM index_snapshot WHERE tenant_id=:tenant AND project_id=:project"), {"tenant": SCOPE_ONE.tenant_id, "project": SCOPE_ONE.project_id}).scalar_one() == 2
        hits = vector_search(connection, SCOPE_ONE, correction.index_version, tuple([1.0] * DIMENSIONS), 5)
        wrong_scope_hits = vector_search(connection, SCOPE_TWO, correction.index_version, tuple([1.0] * DIMENSIONS), 5)
    assert [(hit.source_version, hit.passage_id) for hit in hits] == [("v2", "p1")]
    assert wrong_scope_hits == []


def test_incomplete_active_document_is_not_embedded_or_indexed(db_engine) -> None:
    """Trusting a manually-created head without reconciling its passage count would index partial evidence."""
    with db_engine.begin() as connection:
        _insert_active_document(connection, SCOPE_ONE, "v1", [("p1", "only passage")], expected_count=2)
    client = _DeterministicClient()

    report = embed_active_passages(db_engine, SCOPE_ONE, client)

    assert client.calls == []
    assert report.active_passages == 0
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM embedding_run")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM index_member")).scalar_one() == 0


def test_duplicate_normalized_inputs_reuse_one_vector_with_distinct_bindings(db_engine) -> None:
    """Calling the provider twice for one scoped input hash would bypass the promised vector cache."""
    with db_engine.begin() as connection:
        _insert_active_document(
            connection,
            SCOPE_ONE,
            "v1",
            [("p1", "same text"), ("p2", "same text")],
        )
    client = _DeterministicClient()

    report = embed_active_passages(db_engine, SCOPE_ONE, client)

    assert client.calls == [("same text",)]
    assert (report.embedded_passages, report.reused_passages) == (1, 1)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM embedding_run")).scalar_one() == 2
        assert connection.execute(text("SELECT count(*) FROM passage_embedding")).scalar_one() == 2
        assert connection.execute(text("SELECT count(*) FROM index_member")).scalar_one() == 2


def test_batch_usage_is_accounted_once_across_unique_inputs(db_engine) -> None:
    """One provider batch must contribute its usage totals exactly once to the audit sum."""
    with db_engine.begin() as connection:
        _insert_active_document(
            connection,
            SCOPE_ONE,
            "v1",
            [("p1", "first unique text"), ("p2", "second unique text")],
        )
    client = _DeterministicClient()

    embed_active_passages(db_engine, SCOPE_ONE, client)

    with db_engine.connect() as connection:
        usage = connection.execute(
            text(
                """SELECT sum(COALESCE((usage->>'prompt_tokens')::bigint,0)),
                              sum(COALESCE((usage->>'total_tokens')::bigint,0))
                   FROM embedding_run"""
            )
        ).one()
    assert usage == (2, 2)


def test_incompatible_client_batch_rolls_back_without_claiming_success(db_engine) -> None:
    """Persisting a short vector from a protocol implementation would create an unusable succeeded run."""
    with db_engine.begin() as connection:
        _insert_active_passage(connection, SCOPE_ONE, "v1", "p1", "text")
    client = _DeterministicClient()

    def malformed(_texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch(
            model="test-model",
            vectors=((1.0,),),
            dimensions=DIMENSIONS,
            encoding_format="float",
            normalization="provider_unit_length_v1",
            prompt_tokens=1,
            total_tokens=1,
        )

    client.embed = malformed  # type: ignore[method-assign]
    with pytest.raises(EmbeddingInputError, match="incompatible batch"):
        embed_active_passages(db_engine, SCOPE_ONE, client)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM embedding_run")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM index_snapshot")).scalar_one() == 0


def test_second_batch_failure_rolls_back_first_batch_and_snapshot(db_engine) -> None:
    """Committing the first 64 rows before a later provider failure would make status untruthful."""
    with db_engine.begin() as connection:
        _insert_active_document(
            connection,
            SCOPE_ONE,
            "v1",
            [(f"p{index:03d}", f"text {index}") for index in range(65)],
        )
    client = _DeterministicClient()
    successful_embed = client.embed

    def fail_second_batch(texts: Sequence[str]) -> EmbeddingBatch:
        if client.calls:
            client.calls.append(tuple(texts))
            raise EmbeddingProtocolError("second batch failed")
        return successful_embed(texts)

    client.embed = fail_second_batch  # type: ignore[method-assign]
    with pytest.raises(EmbeddingProtocolError, match="second batch failed"):
        embed_active_passages(db_engine, SCOPE_ONE, client)
    assert [len(call) for call in client.calls] == [64, 1]
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM embedding_run")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM passage_embedding")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM index_snapshot")).scalar_one() == 0


def test_vector_search_rejects_non_numeric_query_values(db_engine) -> None:
    """Letting a conversion error escape would bypass the public input-error contract."""
    with db_engine.connect() as connection:
        with pytest.raises(EmbeddingInputError, match="finite values"):
            vector_search(connection, SCOPE_ONE, "index", ["not-a-number"] * DIMENSIONS, 5)
