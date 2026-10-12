from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import Sequence
from uuid import uuid4

import pytest
from potency_kg.age_projection import refresh_graph
from potency_kg.contracts import Scope, ScopedSourceVersion
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.embeddings import embed_active_passages
from potency_kg.graph_store import write_structural_version
from potency_kg.ids import derive_scoped_id
from potency_kg.retrieval import RetrievalLimits, retrieve_graph_smoke
from sqlalchemy import event, text

DIMENSIONS = 3072
SCOPE = Scope("tenant", "retrieve")
OTHER_SCOPE = Scope("tenant", "other-project")


@dataclass
class _Client:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity("test", "model", DIMENSIONS, "float", "provider_unit_length_v1", "https://invalid", "KEY")
    calls: list[tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        self.calls = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(tuple(texts))
        vectors = tuple(tuple(1.0 for _ in range(DIMENSIONS)) for _ in texts)
        return EmbeddingBatch("model", vectors, DIMENSIONS, "float", "provider_unit_length_v1", 0, 0)


@dataclass
class _BlockingQueryClient:
    entered: Event
    release: Event
    identity: EmbeddingClientIdentity = _Client.identity

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test did not release query embedding")
        values = tuple(texts)
        vectors = tuple(tuple(1.0 for _ in range(DIMENSIONS)) for _ in values)
        return EmbeddingBatch("model", vectors, DIMENSIONS, "float", "provider_unit_length_v1", 0, 0)


def test_retrieval_embeds_outside_activation_lock_then_revalidates_exact_binding(db_engine) -> None:
    """External embedding I/O must not block activation, and stale preflight state must fail closed."""
    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','title','sha',1,'final')"), params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) VALUES (:tenant,:project,'source','v1','p1','evidence','text-sha')"), params)
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1','p1','structural_only','none')"), params)
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))
    setup_client = _Client()
    index = embed_active_passages(db_engine, SCOPE, setup_client)
    refresh_graph(db_engine, SCOPE, mode="semantic")

    embedding_entered = Event()
    release_embedding = Event()
    refresh_at_activation = Event()
    refresh_done = Event()
    retrieval_errors: list[BaseException] = []
    refresh_errors: list[BaseException] = []

    def retrieve() -> None:
        try:
            retrieve_graph_smoke(
                db_engine,
                SCOPE,
                "what evidence?",
                _BlockingQueryClient(embedding_entered, release_embedding),
                index.index_version,
                RetrievalLimits(),
            )
        except BaseException as error:
            retrieval_errors.append(error)

    def force_refresh() -> None:
        try:
            refresh_graph(
                db_engine,
                SCOPE,
                mode="semantic",
                force=True,
                failure_injector=lambda point: (
                    refresh_at_activation.set() if point == "before_activation" else None
                ),
            )
        except BaseException as error:
            refresh_errors.append(error)
        finally:
            refresh_done.set()

    retrieval_thread = Thread(target=retrieve)
    refresh_thread = Thread(target=force_refresh)
    retrieval_thread.start()
    assert embedding_entered.wait(5)
    refresh_thread.start()
    assert refresh_at_activation.wait(5)
    refresh_completed_during_embedding = refresh_done.wait(2)
    release_embedding.set()
    retrieval_thread.join(5)
    refresh_thread.join(5)

    assert not retrieval_thread.is_alive() and not refresh_thread.is_alive()
    assert refresh_errors == []
    assert refresh_completed_during_embedding is True
    assert len(retrieval_errors) == 1
    assert isinstance(retrieval_errors[0], ValueError)
    assert "changed during query embedding" in str(retrieval_errors[0])


def test_retrieval_embeds_query_and_returns_scoped_source_text_and_bounded_trace(db_engine) -> None:
    """Using another index/scope or AGE text would make the evidence trace irreproducible."""
    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','title','sha',64,'final')"), params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v1','p1','relational original text','text-sha',CAST(:locator AS jsonb))"), {**params, "locator": '{"page":2}'})
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1','p1','structural_only','none')"), params)
        for number in range(2, 65):
            passage_id = f"z{number:02d}"
            connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v1',:passage_id,:body,:text_sha,CAST(:locator AS jsonb))"), {**params, "passage_id": passage_id, "body": f"decoy {number}", "text_sha": f"decoy-sha-{number}", "locator": f'{{"page":{number}}}'})
            connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1',:passage_id,'structural_only','none')"), {**params, "passage_id": passage_id})
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))
        other_params = {"tenant": OTHER_SCOPE.tenant_id, "project": OTHER_SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','other title','other-sha',1,'final')"), other_params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v1','p1','other project private text','other-text-sha',CAST(:locator AS jsonb))"), {**other_params, "locator": '{"page":99}'})
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1','p1','structural_only','none')"), other_params)
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), other_params)
        write_structural_version(connection, ScopedSourceVersion(*OTHER_SCOPE.as_tuple(), "source", "v1"))
    client = _Client()
    index = embed_active_passages(db_engine, SCOPE, client)
    graph = refresh_graph(db_engine, SCOPE, mode="semantic")
    other_index = embed_active_passages(db_engine, OTHER_SCOPE, client)
    other_graph = refresh_graph(db_engine, OTHER_SCOPE, mode="semantic")

    age_result_sizes: list[int] = []

    def record_age_result_size(_connection, cursor, statement, _parameters, _context, _many) -> None:
        if "FROM cypher(" in statement:
            age_result_sizes.append(cursor.rowcount)

    event.listen(db_engine, "after_cursor_execute", record_age_result_size)
    try:
        trace = retrieve_graph_smoke(db_engine, SCOPE, "what evidence?", client, index.index_version, RetrievalLimits())
    finally:
        event.remove(db_engine, "after_cursor_execute", record_age_result_size)

    assert client.calls[-1] == ("what evidence?",)
    assert trace.graph_version == graph.graph_version
    assert trace.index_version == index.index_version and trace.embedding_version == index.embedding_version
    assert trace.limits == RetrievalLimits(vector_hits=8, semantic_seeds=8, hops=2, nodes=30, edges=50)
    assert trace.vector_hits[0].passage_id == "p1"
    assert trace.source_refs[0].text == "relational original text"
    assert trace.source_refs[0].as_tuple()[:5] == ("tenant", "retrieve", "source", "v1", "p1")
    assert trace.index_version != other_index.index_version
    assert trace.graph_name != other_graph.graph_name
    assert all(ref.project_id == SCOPE.project_id for ref in trace.source_refs)
    assert all(ref.text != "other project private text" for ref in trace.source_refs)
    assert len(trace.node_ids) <= 30 and len(trace.edge_ids) <= 50
    assert age_result_sizes and max(age_result_sizes) <= 50


def test_retrieval_uses_active_age_topology_and_passage_provenance_for_semantic_seeds(db_engine) -> None:
    """Walking current relational rows or only passage edges would detach a trace from its graph version."""
    extraction_run_id = uuid4()
    assertion_id = derive_scoped_id(
        "assertion", SCOPE, "source", "v1", "p1", str(extraction_run_id), "a1"
    )
    standalone_assertion_id = derive_scoped_id(
        "assertion", SCOPE, "source", "v1", "p1", str(extraction_run_id), "a2"
    )
    product_id = derive_scoped_id("entity", SCOPE, "Product", "product")
    evidence_edge_id = derive_scoped_id(
        "evidence", SCOPE, "source", "v1", "p1", str(extraction_run_id), "r1"
    )
    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','title','sha',1,'final')"), params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v1','p1','relational original text','text-sha',CAST(:locator AS jsonb))"), {**params, "locator": '{"page":2}'})
        connection.execute(
            text(
                """INSERT INTO llm_extraction_run
                   (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,
                    extraction_version,model,sanitized_origin,credential_env,instructor_mode,
                    selected_bundles,input_sha256,profile_name,profile_aliases,
                    profile_source_sha256,profile_version,alias_catalog_version,status,response)
                   VALUES (:run_id,:tenant,:project,'source','v1','p1','extraction-test','test',
                    'https://invalid','KEY','openai','[]','input','empty-v1','[]','profile-sha',
                    'profile-v1','approved-alias-catalog-v1-retrieval','succeeded','{}')"""
            ),
            {**params, "run_id": extraction_run_id},
        )
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason,active_extraction_run_id) VALUES (:tenant,:project,'source','v1','p1','extracted','extracted_successfully',:run_id)"), {**params, "run_id": extraction_run_id})
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))
        connection.execute(
            text("INSERT INTO assertion (tenant_id,project_id,assertion_id,source_id,source_version,passage_id,extraction_run_id,claim,chronology_state,review_state,original_quote) VALUES (:tenant,:project,:assertion_id,'source','v1','p1',:run_id,'claim','reported','unreviewed','exact quote')"),
            {**params, "assertion_id": assertion_id, "run_id": extraction_run_id},
        )
        connection.execute(
            text("INSERT INTO assertion (tenant_id,project_id,assertion_id,source_id,source_version,passage_id,extraction_run_id,claim,chronology_state,review_state,original_quote) VALUES (:tenant,:project,:assertion_id,'source','v1','p1',:run_id,'standalone claim','reported','unreviewed','standalone quote')"),
            {**params, "assertion_id": standalone_assertion_id, "run_id": extraction_run_id},
        )
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:assertion_id,'Assertion','{}'),(:tenant,:project,:standalone_assertion_id,'Assertion','{}'),(:tenant,:project,:product_id,'Product','{}')"), {**params, "assertion_id": assertion_id, "standalone_assertion_id": standalone_assertion_id, "product_id": product_id})
        connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,:edge_id,'applies_to',:assertion_id,:product_id,:assertion_id,true,'{}')"), {**params, "edge_id": evidence_edge_id, "assertion_id": assertion_id, "product_id": product_id})
        for node_id in (assertion_id, standalone_assertion_id, product_id):
            connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:node_id,'source','v1','p1')"), {**params, "node_id": node_id})
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:edge_id,'source','v1','p1')"), {**params, "edge_id": evidence_edge_id})

    client = _Client()
    index = embed_active_passages(db_engine, SCOPE, client)
    graph = refresh_graph(db_engine, SCOPE, mode="semantic")
    with db_engine.begin() as connection:
        connection.execute(
            text("DELETE FROM kg_edge_source_ref WHERE tenant_id=:tenant AND project_id=:project AND edge_id=:edge_id"),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "edge_id": evidence_edge_id},
        )
        connection.execute(
            text("DELETE FROM kg_edge WHERE tenant_id=:tenant AND project_id=:project AND edge_id=:edge_id"),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "edge_id": evidence_edge_id},
        )

    trace = retrieve_graph_smoke(
        db_engine, SCOPE, "what evidence?", client, index.index_version, RetrievalLimits()
    )

    assert trace.graph_version == graph.graph_version
    assert trace.seed_node_ids == tuple(sorted((assertion_id, standalone_assertion_id, product_id)))
    assert trace.edge_ids == (evidence_edge_id,)

    no_seed_trace = retrieve_graph_smoke(
        db_engine,
        SCOPE,
        "what evidence?",
        client,
        index.index_version,
        RetrievalLimits(semantic_seeds=0),
    )
    assert no_seed_trace.seed_node_ids == ()


def test_retrieval_rejects_historical_index_not_bound_to_active_graph(db_engine) -> None:
    """A valid historical scoped index must not seed the graph activated for newer membership."""
    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(
            text(
                """INSERT INTO source_document
                   (tenant_id,project_id,source_id,source_version,title,source_sha256,
                    expected_passage_count,status)
                   VALUES (:tenant,:project,'source','v1','title','sha-v1',1,'final')"""
            ),
            params,
        )
        connection.execute(
            text(
                """INSERT INTO source_passage
                   (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
                   VALUES (:tenant,:project,'source','v1','p1','old evidence','text-v1')"""
            ),
            params,
        )
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1','p1','structural_only','none')"), params)
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))
    client = _Client()
    historical = embed_active_passages(db_engine, SCOPE, client)

    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(
            text(
                """INSERT INTO source_document
                   (tenant_id,project_id,source_id,source_version,title,source_sha256,
                    expected_passage_count,status)
                   VALUES (:tenant,:project,'source','v2','title','sha-v2',1,'final')"""
            ),
            params,
        )
        connection.execute(
            text(
                """INSERT INTO source_passage
                   (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
                   VALUES (:tenant,:project,'source','v2','p1','current evidence','text-v2')"""
            ),
            params,
        )
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v2','p1','structural_only','none')"), params)
        connection.execute(text("UPDATE source_head SET source_version='v2' WHERE tenant_id=:tenant AND project_id=:project AND source_id='source'"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v2"))
    current = embed_active_passages(db_engine, SCOPE, client)
    refresh_graph(db_engine, SCOPE, mode="semantic")

    assert historical.index_version != current.index_version
    with pytest.raises(ValueError, match="active graph"):
        retrieve_graph_smoke(
            db_engine,
            SCOPE,
            "what evidence?",
            client,
            historical.index_version,
            RetrievalLimits(),
        )
