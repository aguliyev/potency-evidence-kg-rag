from __future__ import annotations

from dataclasses import dataclass, replace
import runpy
from typing import Sequence
from uuid import uuid4

import pytest
from sqlalchemy import text

from potency_kg.age_projection import refresh_graph
from potency_kg.contracts import Scope, ScopedSourceVersion
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.embeddings import PassageHit, embed_active_passages
from potency_kg.graph_store import write_structural_version
from potency_kg.ids import derive_scoped_id
from potency_kg.retrieval_config import HybridRetrievalConfig


@dataclass
class _Client:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity("test", "model", 3072, "float", "provider_unit_length_v1", "https://invalid", "KEY")

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch("model", tuple(tuple(1.0 for _ in range(3072)) for _ in texts), 3072, "float", "provider_unit_length_v1", 0, 0)


def test_projected_evidence_degree_excludes_hub_seed_even_when_refs_are_missing() -> None:
    from potency_kg.retrieval import _beam_candidates

    nodes = {"hub": "Product", **{f"n{number}": "Product" for number in range(21)}}
    edges = [{"edge_id": f"e{number}", "edge_type": "applies_to",
              "source_node_id": "hub", "target_node_id": f"n{number}", "is_evidence": True}
             for number in range(21)]
    node_refs = {"hub": [{"source_id": "s", "source_version": "v1", "passage_id": "p",
                          "text": "original", "text_sha256": "sha", "locator": {}, "cosine": 0.9}]}
    edge_refs = {"e0": [node_refs["hub"][0]]}
    config = HybridRetrievalConfig(vector_hits=1, seed_vector_hits=1, max_hops=1,
                                   hop_bonus=(0.05, 0.03))
    candidates, seeds, *_ = _beam_candidates(nodes, edges, node_refs, edge_refs,
                                              [PassageHit("s", "v1", "p", 0.9)], config)
    assert seeds == () and candidates == []


def test_parallel_evidence_edges_do_not_consume_distinct_neighbor_slots() -> None:
    from potency_kg.retrieval import _beam_candidates

    def ref(passage_id: str, cosine: float) -> dict[str, object]:
        return {"source_id": "source", "source_version": "v1", "passage_id": passage_id,
                "text": passage_id, "text_sha256": passage_id, "locator": {}, "cosine": cosine}

    nodes = {"seed": "Assertion", "first": "Product", "second": "Product"}
    edges = [{"edge_id": edge_id, "edge_type": "applies_to", "source_node_id": "seed",
              "target_node_id": neighbor, "is_evidence": True}
             for edge_id, neighbor in (("e1", "first"), ("e2", "first"), ("e3", "second"))]
    refs = {"seed": [ref("a", 0.95)], "first": [ref("b", 0.9)],
            "second": [ref("c", 0.8)]}
    edge_refs = {edge["edge_id"]: [ref("a", 0.95)] for edge in edges}
    config = HybridRetrievalConfig(vector_hits=1, seed_vector_hits=1,
                                   neighbors_per_entity=2, max_hops=1,
                                   hop_bonus=(0.05, 0.03))
    candidates, seeds, visited, selected_edges, diagnostics = _beam_candidates(
        nodes, edges, refs, edge_refs, [PassageHit("source", "v1", "a", 0.95)], config)
    assert seeds == ("seed",)
    assert visited == ("seed", "first", "second")
    assert selected_edges == ("e1", "e3")
    assert {candidate.passage_id for candidate in candidates} == {"a", "b", "c"}
    assert diagnostics["expansions"][0]["relations_kept"] == 2


def test_parallel_non_evidence_edge_does_not_hide_evidence_path() -> None:
    from potency_kg.retrieval import _beam_candidates

    def ref(passage_id: str) -> dict[str, object]:
        return {"source_id": "source", "source_version": "v1", "passage_id": passage_id,
                "text": passage_id, "text_sha256": passage_id, "locator": {}, "cosine": 0.8}

    nodes = {"seed": "Assertion", "target": "Product"}
    edges = [
        {"edge_id": "a-structural", "edge_type": "applies_to", "source_node_id": "seed",
         "target_node_id": "target", "is_evidence": False},
        {"edge_id": "z-evidence", "edge_type": "applies_to", "source_node_id": "seed",
         "target_node_id": "target", "is_evidence": True},
    ]
    node_refs = {"seed": [ref("a")], "target": [ref("b")]}
    edge_refs = {edge["edge_id"]: [ref("b")] for edge in edges}
    config = HybridRetrievalConfig(vector_hits=1, seed_vector_hits=1,
                                   evidence_edges_only=False, max_hops=1,
                                   hop_bonus=(0.05, 0.03))
    candidates, _, _, selected_edges, _ = _beam_candidates(
        nodes, edges, node_refs, edge_refs, [PassageHit("source", "v1", "a", 0.8)], config)
    assert selected_edges == ("z-evidence",)
    assert "b" in {candidate.passage_id for candidate in candidates}


def test_a_baseline_keeps_hub_neighbor_without_expanding_it() -> None:
    a_candidates = runpy.run_path(
        ".superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py"
    )["a_candidates"]
    nodes = {"seed": "Assertion", "hub": "Product",
             **{f"leaf{i}": "Product" for i in range(20)}}
    edges = [{"edge_id": "e0", "edge_type": "applies_to",
              "source_node_id": "seed", "target_node_id": "hub", "is_evidence": True}]
    edges.extend({"edge_id": f"e{i+1}", "edge_type": "applies_to",
                  "source_node_id": "hub", "target_node_id": f"leaf{i}", "is_evidence": True}
                 for i in range(20))
    def ref(passage_id: str) -> dict[str, object]:
        return {"source_id": "source", "source_version": "v1", "passage_id": passage_id,
                "text": passage_id, "cosine": 0.8}
    node_refs = {"seed": [ref("a")], "hub": [ref("b")],
                 **{f"leaf{i}": [ref(f"leaf{i}")] for i in range(20)}}
    edge_refs = {edge["edge_id"]: [ref("a")] for edge in edges}
    config = HybridRetrievalConfig()
    candidates, seeds, hops, expansions = a_candidates(
        nodes, edges, node_refs, edge_refs, [PassageHit("source", "v1", "a", 0.8)], config)
    assert seeds == ["seed"]
    assert "b" in {candidate["passage_id"] for candidate in candidates}
    assert hops == [1, 1]
    assert next(item for item in expansions if item["entity_id"] == "hub")["relations_kept"] == 0


def test_graph_retrieval_selects_edge_evidence_outside_vector_hits(db_engine, monkeypatch) -> None:
    from potency_kg import retrieval

    scope = Scope("tenant", "hybrid")
    run_id = uuid4()
    assertion_id = derive_scoped_id("assertion", scope, "source", "v1", "a", str(run_id), "a1")
    product_id = derive_scoped_id("entity", scope, "Product", "product")
    edge_id = derive_scoped_id("evidence", scope, "source", "v1", "a", str(run_id), "r1")
    structural_product_id = derive_scoped_id("entity", scope, "Product", "structural-only")
    structural_edge_id = derive_scoped_id("structural", scope, "source", "v1", "c")
    unreferenced_product_id = derive_scoped_id("entity", scope, "Product", "unreferenced")
    unreferenced_edge_id = derive_scoped_id("evidence", scope, "source", "v1", "b", "unreferenced")
    params = {"tenant": scope.tenant_id, "project": scope.project_id, "run_id": run_id,
              "assertion_id": assertion_id, "product_id": product_id, "edge_id": edge_id,
              "structural_product_id": structural_product_id, "structural_edge_id": structural_edge_id,
              "unreferenced_product_id": unreferenced_product_id, "unreferenced_edge_id": unreferenced_edge_id}
    with db_engine.begin() as connection:
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','title','sha',4,'final')"), params)
        for passage_id in ("a", "b", "c", "d"):
            connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) VALUES (:tenant,:project,'source','v1',:passage_id,:body,:sha)"), {**params, "passage_id": passage_id, "body": f"original {passage_id}", "sha": f"sha-{passage_id}"})
            connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1',:passage_id,'structural_only','none')"), {**params, "passage_id": passage_id})
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
        write_structural_version(connection, ScopedSourceVersion(*scope.as_tuple(), "source", "v1"))
        connection.execute(text("INSERT INTO llm_extraction_run (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,extraction_version,model,sanitized_origin,credential_env,instructor_mode,selected_bundles,input_sha256,profile_name,profile_aliases,profile_source_sha256,profile_version,alias_catalog_version,status,response) VALUES (:run_id,:tenant,:project,'source','v1','a','extraction-test','test','https://invalid','KEY','openai','[]','input','empty-v1','[]','profile-sha','profile-v1','approved-alias-catalog-v1-retrieval','succeeded','{}')"), params)
        connection.execute(text("UPDATE passage_processing SET state='extracted',reason='extracted_successfully',active_extraction_run_id=:run_id WHERE tenant_id=:tenant AND project_id=:project AND source_id='source' AND source_version='v1' AND passage_id='a'"), params)
        connection.execute(text("INSERT INTO assertion (tenant_id,project_id,assertion_id,source_id,source_version,passage_id,extraction_run_id,claim,chronology_state,review_state,original_quote) VALUES (:tenant,:project,:assertion_id,'source','v1','a',:run_id,'claim','reported','unreviewed','original a')"), params)
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:assertion_id,'Assertion','{}'),(:tenant,:project,:product_id,'Product','{}')"), params)
        connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,:edge_id,'applies_to',:assertion_id,:product_id,:assertion_id,true,'{}')"), params)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:assertion_id,'source','v1','a'),(:tenant,:project,:product_id,'source','v1','d')"), params)
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:edge_id,'source','v1','b')"), params)
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:structural_product_id,'Product','{}')"), params)
        connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,:structural_edge_id,'applies_to',:assertion_id,:structural_product_id,NULL,false,'{}')"), params)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:structural_product_id,'source','v1','c')"), params)
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:structural_edge_id,'source','v1','c')"), params)
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:unreferenced_product_id,'Product','{}')"), params)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:unreferenced_product_id,'source','v1','b')"), params)
        connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,:unreferenced_edge_id,'applies_to',:assertion_id,:unreferenced_product_id,:assertion_id,true,'{}')"), params)
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:unreferenced_edge_id,'source','v1','b')"), params)
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v0','old title','old-sha',1,'final')"), params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) VALUES (:tenant,:project,'source','v0','old','superseded private evidence','old-text-sha')"), params)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:product_id,'source','v0','old')"), params)
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:edge_id,'source','v0','old')"), params)
    client = _Client()
    index = embed_active_passages(db_engine, scope, client)
    refresh_graph(db_engine, scope, mode="semantic")
    with db_engine.begin() as connection:
        connection.execute(text("DELETE FROM kg_node_source_ref WHERE tenant_id=:tenant AND project_id=:project AND node_id=:unreferenced_product_id"), params)
        connection.execute(text("DELETE FROM kg_edge_source_ref WHERE tenant_id=:tenant AND project_id=:project AND edge_id=:unreferenced_edge_id"), params)
        other = {**params, "project": "another-project"}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','private title','private-sha',1,'final')"), other)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) VALUES (:tenant,:project,'source','v1','b','other project private text','private-text-sha')"), other)
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:product_id,'Product','{}')"), other)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:product_id,'source','v1','b')"), other)
        connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,'stale-node','Product','{}')"), params)
        connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,'stale-node','source','v1','c')"), params)
        connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,'stale-edge','applies_to',:assertion_id,'stale-node',:assertion_id,true,'{}')"), params)
        connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,'stale-edge','source','v1','c')"), params)
    config = HybridRetrievalConfig(vector_hits=1, seed_vector_hits=1, max_hops=1, hop_bonus=(0.05, 0.03))
    vector = retrieval.retrieve(db_engine, scope, "question", "vector", client, index.index_version, config)
    graph = retrieval.retrieve(db_engine, scope, "question", "graph", client, index.index_version, config)
    assert [candidate.passage_id for candidate in vector.candidates] == ["a"]
    selected = {candidate.passage_id: candidate for candidate in graph.candidates}
    assert "b" in selected and "c" not in selected and "old" not in selected
    assert selected["b"].selected_via == "graph"
    assert selected["b"].hop == 1
    assert selected["b"].via_edge_id == edge_id
    assert selected["b"].text == "original b"
    with db_engine.connect() as connection:
        assert connection.execute(text("""SELECT count(*) FROM kg_node_source_ref AS ref
            JOIN kg_node AS node ON (node.tenant_id,node.project_id,node.node_id)=
              (ref.tenant_id,ref.project_id,ref.node_id)
            WHERE ref.tenant_id=:tenant AND ref.project_id=:project
              AND ref.source_id='source' AND ref.source_version='v1' AND ref.passage_id='b'
              AND node.node_type NOT IN ('SourceDocument','SourcePassage')"""), params).scalar_one() == 0
    assert selected["d"].via_edge_id == edge_id
    assert all(candidate.text != "other project private text" for candidate in graph.candidates)
    assert selected["b"].source_metadata["title"] == "title"
    assert selected["a"].hop == 0 and selected["a"].via_node_id == assertion_id
    assert selected["a"].via_edge_id is None
    assert "stale-node" not in graph.node_ids and "stale-edge" not in graph.edge_ids
    assert unreferenced_product_id not in graph.node_ids
    assert graph.diagnostics["skipped_source_refs_without_bound_index_passage"] >= 2
    assert graph.diagnostics["skipped_nodes_without_current_provenance"] >= 1
    assert graph.diagnostics["skipped_edges_without_current_provenance"] >= 1
    assert graph.diagnostics["config"] == config.as_dict()
    assert graph.diagnostics["entities_visited_per_hop"] == [1, 1]
    assert graph.diagnostics["expansions"][0]["relations_kept"] == 1

    permissive = retrieval.retrieve(db_engine, scope, "question", "graph", client,
                                    index.index_version,
                                    replace(config, evidence_edges_only=False))
    assert structural_product_id in permissive.node_ids
    assert "c" not in {candidate.passage_id for candidate in permissive.candidates}

    original_binding = retrieval._active_graph_binding
    binding_reads = 0

    def binding_changes_between_transactions(*args, **kwargs):
        nonlocal binding_reads
        binding_reads += 1
        binding = original_binding(*args, **kwargs)
        return replace(binding, graph_version="mismatched-version") if binding_reads == 2 else binding

    monkeypatch.setattr(retrieval, "_active_graph_binding", binding_changes_between_transactions)
    with pytest.raises(ValueError, match="active graph binding changed"):
        retrieval.retrieve(db_engine, scope, "question", "graph", client, index.index_version, config)
    assert binding_reads == 2
