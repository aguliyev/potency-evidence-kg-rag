"""One-off, retrieval-only five-question A/D measurement on the active local graph."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from pydantic import SecretStr
from sqlalchemy import text

from potency_kg.contracts import Scope
from potency_kg.db import create_engine_from_env
from potency_kg.embedding_client import OpenAIEmbeddingClient
from potency_kg.embeddings import embed_query, vector_search
from potency_kg.retrieval import _active_graph_binding, _active_topology, _beam_candidates, _provenance_rows
from potency_kg.retrieval_config import HybridRetrievalConfig
from potency_kg.settings import load_embedding_settings
from potency_kg.ids import derive_embedding_version


def a_candidates(nodes, edges, node_refs, edge_refs, hits, config):
    """Traversal A on the activated AGE snapshot: bounded edge-ID BFS, hubs unexpanded."""
    excluded = set(config.excluded_node_types) | {"SourceDocument", "SourcePassage"}
    adjacency = defaultdict(list)
    degree = defaultdict(int)
    for edge in edges:
        if not edge["is_evidence"]:
            continue
        source, target = str(edge["source_node_id"]), str(edge["target_node_id"])
        if nodes.get(source) in excluded or nodes.get(target) in excluded:
            continue
        degree[source] += 1
        degree[target] += 1
        if not edge_refs.get(edge["edge_id"]):
            continue
        adjacency[source].append((target, edge))
        adjacency[target].append((source, edge))
    hit_keys = [(hit.source_id, hit.source_version, hit.passage_id) for hit in hits[:8]]
    seeds = []
    for key in hit_keys:
        for node_id in sorted(nodes):
            if nodes[node_id] in excluded or degree[node_id] > 20 or node_id in seeds:
                continue
            if any((ref["source_id"], ref["source_version"], ref["passage_id"]) == key
                   for ref in node_refs.get(node_id, ())):
                seeds.append(node_id)
                if len(seeds) == 8:
                    break
        if len(seeds) == 8:
            break
    frontier = [(node, 0, None, (node,), None, False) for node in seeds]
    visited = set(seeds)
    selected = []
    traversed_edges = set()
    per_hop = []
    expansions = []
    for hop in range(3):
        per_hop.append(len(frontier))
        following = []
        edge_options = []
        for node, _, via_edge, path, edge_type, hub_path in frontier:
            for ref in node_refs.get(node, ()):
                selected.append((ref, hop, node, via_edge, path, edge_type, hub_path or degree[node] > 20))
            if via_edge:
                for ref in edge_refs.get(via_edge, ()):
                    selected.append((ref, hop, node, via_edge, path, edge_type, hub_path or degree[node] > 20))
            if hop < 2 and degree[node] <= 20:
                for neighbor, edge in adjacency.get(node, ()):
                    edge_options.append((str(edge["edge_id"]), node, neighbor, edge,
                                         path, hub_path))
        kept_by_node = defaultdict(int)
        for edge_id, parent, neighbor, edge, path, hub_path in sorted(edge_options):
            if neighbor in visited or not node_refs.get(neighbor):
                continue
            if len(visited) >= 30 or len(traversed_edges) >= 50:
                break
            visited.add(neighbor)
            traversed_edges.add(edge_id)
            following.append((neighbor, hop + 1, edge_id,
                              path + (edge_id, neighbor), str(edge["edge_type"]),
                              hub_path or degree[neighbor] > 20))
            kept_by_node[parent] += 1
        for node, *_ in frontier:
            expansions.append({"hop": hop, "entity_id": node,
                               "relations_available": len(adjacency.get(node, ())),
                               "relations_kept": kept_by_node[node], "hub": degree[node] > 20})
        frontier = following
        if not frontier:
            break
    by_key = {}
    for ref, hop, node, edge, path, edge_type, hub_path in selected:
        key = (str(ref["source_id"]), str(ref["source_version"]), str(ref["passage_id"]))
        cosine = float(ref["cosine"])
        score = config.graph_score(cosine, hop, edge_type)
        record = {"source_id": key[0], "source_version": key[1], "passage_id": key[2],
                  "text": str(ref["text"]), "cosine": cosine, "score": score, "hop": hop,
                  "via_node_id": node, "via_edge_id": edge, "supporting_path_ids": path,
                  "hub_mediated": hub_path}
        if key not in by_key or score > by_key[key]["score"]:
            by_key[key] = record
    return sorted(by_key.values(), key=lambda row: (-row["score"], row["source_id"], row["passage_id"])), seeds, per_hop, expansions


def main():
    settings = load_embedding_settings(os.environ)
    client = OpenAIEmbeddingClient(settings, SecretStr(os.environ[settings.credential_env]))
    expected_embedding_version = derive_embedding_version(
        client.identity.model, client.identity.dimensions,
        client.identity.encoding_format, client.identity.normalization)
    engine = create_engine_from_env()
    scope = Scope("public-fda", "casgevy-potency")
    questions = {q["id"]: q["question"] for q in json.loads(Path("questions.json").read_text())}
    config = HybridRetrievalConfig()
    output = {"config": config.as_dict(), "questions": []}
    for qid in ("P1", "P2", "P4", "P5", "P6"):
        question = questions[qid]
        embedding = embed_query(client, question)
        with engine.begin() as connection:
            connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            graph = connection.execute(text("SELECT index_version FROM graph_scope WHERE tenant_id=:tenant AND project_id=:project"),
                                       {"tenant": scope.tenant_id, "project": scope.project_id}).scalar_one()
            binding = _active_graph_binding(connection, scope, str(graph), expected_embedding_version)
            hits = vector_search(connection, scope, binding.index_version, embedding.vector, config.vector_hits)
            nodes, edges = _active_topology(connection, scope, binding.graph_name)
            node_refs, edge_refs, skipped = _provenance_rows(
                connection, scope, binding.index_version, embedding.vector,
                tuple(nodes), tuple(str(edge["edge_id"]) for edge in edges))
        d, d_seeds, d_nodes, d_edges, d_diag = _beam_candidates(
            nodes, edges, node_refs, edge_refs, hits, config)
        a, a_seeds, a_hops, a_expansions = a_candidates(
            nodes, edges, node_refs, edge_refs, hits, config)
        hit_keys = {(hit.source_id, hit.source_version, hit.passage_id) for hit in hits}
        d_rows = [{"source_id": candidate.source_id,
                   "source_version": candidate.source_version,
                   "passage_id": candidate.passage_id, "text": candidate.text,
                   "cosine": candidate.cosine, "score": candidate.score,
                   "hop": candidate.hop, "via_node_id": candidate.via_node_id,
                   "via_edge_id": candidate.via_edge_id,
                   "supporting_path_ids": candidate.supporting_path_ids,
                   "hub_mediated": candidate.hub_mediated} for candidate in d]
        variants = {}
        for name, rows, seeds, hops, expansions in (("A", a, a_seeds, a_hops, a_expansions),
                                                     ("D", d_rows, d_seeds, d_diag["entities_visited_per_hop"], d_diag["expansions"])):
            nonvector = []
            for rank, row in enumerate(rows, 1):
                if (row["source_id"], row["source_version"], row["passage_id"]) in hit_keys:
                    continue
                nonvector.append({"rank": rank, "mode": name, **row,
                                  "reviewer_verdict": None, "rationale": None})
            variants[name] = {"seeds": seeds, "entities_visited_per_hop": hops,
                              "expansions": expansions, "nonvector": nonvector,
                              "nonvector_count": len(nonvector),
                              "hub_mediated_count": sum(row["hub_mediated"] for row in nonvector)}
        item = {"id": qid, "question": question, "graph_version": binding.graph_version,
                "index_version": binding.index_version, "skipped_noncurrent_provenance": skipped,
                "vector_hits": [{"source_id": hit.source_id, "source_version": hit.source_version,
                                 "passage_id": hit.passage_id, "score": hit.score} for hit in hits],
                "variants": variants}
        output["questions"].append(item)
        print(qid, "A", variants["A"]["nonvector_count"], "D", variants["D"]["nonvector_count"],
              "skipped", skipped, "hubs", variants["D"]["hub_mediated_count"], flush=True)
    Path("docs/graph-signal-candidates.json").write_text(json.dumps(output, indent=2) + "\n")
    print("saved docs/graph-signal-candidates.json", flush=True)


if __name__ == "__main__":
    main()
