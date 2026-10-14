"""Deterministic vector-seeded, bounded graph retrieval trace."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from collections import defaultdict
from typing import Literal, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .age_projection import active_graph_name, bounded_graph_traversal, _cypher, _decode_agtype, _prepare_age
from .contracts import Scope
from .embedding_client import EmbeddingClient
from .embeddings import PassageHit, QueryEmbedding, _vector_literal, embed_query, validate_index_membership, vector_search
from .ids import derive_embedding_version, derive_scoped_id
from .retrieval_config import HybridRetrievalConfig


RETRIEVER_VERSION = "vector-seeded-age-v1"
HYBRID_RETRIEVER_VERSION = "similarity-pruned-beam-v1"


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


@dataclass(frozen=True)
class RetrievalCandidate:
    key: tuple[str, str, str]
    document_id: str
    source_id: str
    source_version: str
    passage_id: str
    text: str
    text_sha256: str
    locator: dict[str, object]
    selected_via: Literal["vector", "graph", "both"]
    score: float
    cosine: float
    hop: int | None
    via_node_id: str | None
    via_edge_id: str | None
    supporting_path_ids: tuple[str, ...]
    hub_mediated: bool = False
    source_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    retriever_version: str
    query_hash: str
    embedding_version: str
    index_version: str
    graph_version: str
    graph_name: str
    config: HybridRetrievalConfig
    vector_hits: tuple[PassageHit, ...]
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    source_refs: tuple[RetrievedSourceRef, ...]
    candidates: tuple[RetrievalCandidate, ...]
    diagnostics: dict[str, object]

    @property
    def ordered_vector_hits(self) -> tuple[PassageHit, ...]:
        return self.vector_hits

    @property
    def traversed_edge_ids(self) -> tuple[str, ...]:
        return self.edge_ids

    @property
    def selected_source_refs(self) -> tuple[RetrievedSourceRef, ...]:
        return self.source_refs


def _active_topology(connection: Connection, scope: Scope, graph_name: str) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Read the activated AGE snapshot; raw kg rows may have changed since activation."""
    _prepare_age(connection)
    nodes = {
        str(_decode_agtype(node_id)): str(_decode_agtype(node_type))
        for node_id, node_type in _cypher(
            connection, graph_name, "MATCH (n) RETURN n.logical_id, n.node_type",
            "node_id agtype, node_type agtype",
        )
    }
    edges = [
        {"edge_id": str(_decode_agtype(edge_id)),
         "edge_type": str(_decode_agtype(edge_type)),
         "source_node_id": str(_decode_agtype(source_id)),
         "target_node_id": str(_decode_agtype(target_id))}
        for edge_id, edge_type, source_id, target_id in _cypher(
            connection, graph_name,
            "MATCH (source)-[edge]->(target) RETURN edge.logical_id, edge.edge_type, source.logical_id, target.logical_id",
            "edge_id agtype, edge_type agtype, source_id agtype, target_id agtype",
        )
    ]
    evidence_flags = {
        str(row["edge_id"]): bool(row["is_evidence"])
        for row in connection.execute(text("""SELECT edge_id,is_evidence FROM kg_edge
            WHERE tenant_id=:tenant AND project_id=:project AND edge_id=ANY(:edge_ids)"""),
            {"tenant": scope.tenant_id, "project": scope.project_id,
             "edge_ids": [edge["edge_id"] for edge in edges]}).mappings()
    }
    for edge in edges:
        edge["is_evidence"] = evidence_flags.get(edge["edge_id"], False)
    return nodes, edges


def _provenance_rows(
    connection: Connection, scope: Scope, index_version: str,
    query_vector: Sequence[float], node_ids: Sequence[str], edge_ids: Sequence[str],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]], int]:
    """Join only current, bound-index original passages to active graph source refs."""
    if not node_ids and not edge_ids:
        return {}, {}, 0
    rows = connection.execute(text("""
        WITH refs AS (
          SELECT 'node' AS kind, node_id AS graph_id, source_id, source_version, passage_id
          FROM kg_node_source_ref
          WHERE tenant_id=:tenant AND project_id=:project AND node_id = ANY(:node_ids)
          UNION ALL
          SELECT 'edge' AS kind, edge_id AS graph_id, source_id, source_version, passage_id
          FROM kg_edge_source_ref
          WHERE tenant_id=:tenant AND project_id=:project AND edge_id = ANY(:edge_ids)
        )
        SELECT refs.kind,refs.graph_id,refs.source_id,refs.source_version,refs.passage_id,
               passage.text,passage.text_sha256,passage.locator,
               CASE WHEN run.vector IS NOT NULL THEN 1 - (run.vector <=> CAST(:query AS vector)) END AS cosine
        FROM refs
        LEFT JOIN index_member AS member
          ON (member.tenant_id,member.project_id,member.index_version,member.source_id,
              member.source_version,member.passage_id) =
             (:tenant,:project,:index_version,refs.source_id,refs.source_version,refs.passage_id)
        LEFT JOIN embedding_run AS run ON run.embedding_run_id=member.embedding_run_id
        LEFT JOIN source_passage AS passage
          ON (passage.tenant_id,passage.project_id,passage.source_id,passage.source_version,passage.passage_id)=
             (:tenant,:project,refs.source_id,refs.source_version,refs.passage_id)
        ORDER BY refs.kind,refs.graph_id,refs.source_id,refs.source_version,refs.passage_id
    """), {"tenant": scope.tenant_id, "project": scope.project_id,
             "index_version": index_version, "query": _vector_literal(tuple(query_vector)),
             "node_ids": list(node_ids), "edge_ids": list(edge_ids)}).mappings()
    node_refs: dict[str, list[dict[str, object]]] = defaultdict(list)
    edge_refs: dict[str, list[dict[str, object]]] = defaultdict(list)
    skipped = 0
    for row in rows:
        if row["cosine"] is None or row["text"] is None:
            skipped += 1
            continue
        target = node_refs if row["kind"] == "node" else edge_refs
        target[str(row["graph_id"])].append(dict(row))
    return node_refs, edge_refs, skipped


def _beam_candidates(
    nodes: dict[str, str], edges: list[dict[str, object]],
    node_refs: dict[str, list[dict[str, object]]], edge_refs: dict[str, list[dict[str, object]]],
    hits: list[PassageHit], config: HybridRetrievalConfig,
) -> tuple[list[RetrievalCandidate], tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, object]]:
    excluded = set(config.excluded_node_types) | {"SourceDocument", "SourcePassage"}
    adjacency: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    degree: dict[str, int] = defaultdict(int)
    projected_relations: dict[str, int] = defaultdict(int)
    skipped_edges = 0
    evidence_edge_ids: set[str] = set()
    for edge in edges:
        source, target = str(edge["source_node_id"]), str(edge["target_node_id"])
        if (source not in nodes or target not in nodes or nodes[source] in excluded
                or nodes[target] in excluded):
            continue
        if edge["is_evidence"]:
            degree[source] += 1
            degree[target] += 1
            projected_relations[source] += 1
            projected_relations[target] += 1
        if not edge_refs.get(edge["edge_id"]):
            skipped_edges += 1
            continue
        if config.evidence_edges_only and not edge["is_evidence"]:
            continue
        if edge["is_evidence"]:
            evidence_edge_ids.add(str(edge["edge_id"]))
        if config.direction in ("both", "out"):
            adjacency[source].append((target, edge))
        if config.direction in ("both", "in"):
            adjacency[target].append((source, edge))
    hubs = {node_id for node_id, count in degree.items() if count > config.hub_degree_cap}
    hit_order = {(hit.source_id, hit.source_version, hit.passage_id): rank
                 for rank, hit in enumerate(hits[:config.seed_vector_hits])}
    seed_order: list[tuple[int, str]] = []
    for node_id, node_type in nodes.items():
        if node_type in excluded or node_id in hubs or not node_refs.get(node_id):
            continue
        rank = min((hit_order.get((str(ref["source_id"]), str(ref["source_version"]), str(ref["passage_id"])), 10**9)
                    for ref in node_refs[node_id]), default=10**9)
        if rank < 10**9:
            seed_order.append((rank, node_id))
    seeds = tuple(node_id for _, node_id in sorted(seed_order)[:config.max_seeds])
    # Frontier value: hop, reaching edge, path IDs, hub involvement, edge type,
    # and whether every traversed relation carries evidence.
    frontier = [(node_id, 0, None, (node_id,), False, None, True) for node_id in seeds]
    visited = set(seeds)
    visited_order: list[str] = []
    traversed_edges: list[str] = []
    candidates: dict[tuple[str, str, str], RetrievalCandidate] = {}
    hop_counts: list[int] = []
    expansion: list[dict[str, object]] = []
    hub_paths = 0

    def add_ref(ref: dict[str, object], node_id: str, hop: int, edge_id: str | None,
                path: tuple[str, ...], hub: bool, edge_type: str | None) -> None:
        key = (str(ref["source_id"]), str(ref["source_version"]), str(ref["passage_id"]))
        cosine = float(ref["cosine"])
        if config.graph_min_cosine is not None and cosine < config.graph_min_cosine:
            return
        candidate = RetrievalCandidate(key, key[0], *key, str(ref["text"]),
            str(ref["text_sha256"]), dict(ref["locator"] or {}), "graph",
            config.graph_score(cosine, hop, edge_type), cosine, hop, node_id, edge_id, path, hub)
        old = candidates.get(key)
        if old is None or (candidate.score, tuple(reversed(candidate.supporting_path_ids))) > (old.score, tuple(reversed(old.supporting_path_ids))):
            candidates[key] = candidate

    for hop in range(config.max_hops + 1):
        hop_counts.append(len(frontier))
        next_options: list[tuple[float, str, str, tuple[str, ...], bool, str, bool]] = []
        for node_id, _, via_edge, path, hub, edge_type, evidence_path in frontier:
            visited_order.append(node_id)
            if hub:
                hub_paths += 1
            if evidence_path:
                for ref in sorted(node_refs.get(node_id, ()), key=lambda ref: -float(ref["cosine"]))[:config.passages_per_entity]:
                    add_ref(ref, node_id, hop, via_edge, path, hub, edge_type)
                if via_edge in evidence_edge_ids:
                    for ref in edge_refs.get(via_edge, ()):
                        add_ref(ref, node_id, hop, via_edge, path, hub, edge_type)
            if hop == config.max_hops:
                continue
            available = adjacency.get(node_id, ())
            if node_id in hubs and not config.expand_hubs:
                expansion.append({"hop": hop, "entity_id": node_id, "relations_available": projected_relations[node_id], "relations_kept": 0, "hub": True})
                continue
            ranked_by_neighbor: dict[str, tuple[float, str, str, tuple[str, ...], bool, str, bool]] = {}
            for neighbor, edge in available:
                if neighbor in visited or not node_refs.get(neighbor):
                    continue
                best_cosine = max(float(ref["cosine"]) for ref in node_refs[neighbor])
                score = config.graph_score(best_cosine, hop + 1, str(edge["edge_type"]))
                edge_id = str(edge["edge_id"])
                option = (score, neighbor, edge_id, path + (edge_id, neighbor),
                          hub or neighbor in hubs, str(edge["edge_type"]),
                          evidence_path and bool(edge["is_evidence"]))
                previous = ranked_by_neighbor.get(neighbor)
                if (previous is None or option[6] > previous[6]
                        or (option[6] == previous[6]
                            and (score > previous[0]
                                 or (score == previous[0] and edge_id < previous[2])))):
                    ranked_by_neighbor[neighbor] = option
            ranked = list(ranked_by_neighbor.values())
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            kept = ranked[:config.neighbors_per_entity]
            expansion.append({"hop": hop, "entity_id": node_id,
                              "relations_available": projected_relations[node_id], "relations_kept": len(kept),
                              "hub": node_id in hubs,
                              "hub_edges_kept": sum(option[4] for option in kept)})
            next_options.extend(kept)
        next_options.sort(key=lambda item: (-item[0], item[1], item[2]))
        frontier = []
        for score, neighbor, edge_id, path, hub, edge_type, evidence_path in next_options:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            traversed_edges.append(edge_id)
            frontier.append((neighbor, hop + 1, edge_id, path, hub, edge_type, evidence_path))
            if len(frontier) >= config.frontier_cap_per_hop:
                break
        if not frontier:
            break
    ordered = sorted(candidates.values(), key=lambda candidate: (-candidate.score, candidate.key))[:config.candidate_cap]
    diagnostics = {"entities_visited_per_hop": hop_counts, "expansions": expansion,
                   "hub_node_ids": sorted(hubs), "hub_mediated_entities": hub_paths,
                   "skipped_edges_without_current_provenance": skipped_edges,
                   "skipped_nodes_without_current_provenance": sum(1 for n in nodes if nodes[n] not in excluded and not node_refs.get(n))}
    return ordered, seeds, tuple(visited_order), tuple(dict.fromkeys(traversed_edges)), diagnostics


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


def retrieve(
    engine: Engine,
    scope: Scope,
    question: str,
    mode: Literal["vector", "graph"],
    embedding_client: EmbeddingClient,
    index_version: str,
    config: HybridRetrievalConfig,
    query_embedding: QueryEmbedding | Sequence[float] | None = None,
) -> RetrievalResult:
    """Retrieve scoped original passages using one bound index and graph snapshot."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if mode not in ("vector", "graph"):
        raise ValueError("mode must be vector or graph")
    if config.traversal != "beam":
        raise ValueError("hybrid retrieval requires similarity-pruned beam traversal")
    expected_embedding_version = derive_embedding_version(
        embedding_client.identity.model, embedding_client.identity.dimensions,
        embedding_client.identity.encoding_format, embedding_client.identity.normalization,
    )
    if query_embedding is None:
        query_vector = embed_query(embedding_client, question).vector
    elif isinstance(query_embedding, QueryEmbedding):
        if (query_embedding.model != embedding_client.identity.model
                or query_embedding.dimensions != embedding_client.identity.dimensions
                or query_embedding.normalization != embedding_client.identity.normalization):
            raise ValueError("precomputed query embedding does not match the adapter")
        query_vector = query_embedding.vector
    else:
        query_vector = tuple(query_embedding)
    with engine.connect() as connection:
        preflight = _active_graph_binding(connection, scope, index_version, expected_embedding_version)
    with engine.begin() as connection:
        connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        current = _active_graph_binding(connection, scope, index_version, expected_embedding_version)
        if current != preflight:
            raise ValueError("active graph binding changed during retrieval")
        hits = vector_search(connection, scope, index_version, query_vector, config.vector_hits)
        if config.vector_min_cosine is not None:
            hits = [hit for hit in hits if hit.score >= config.vector_min_cosine]
        vector_refs = _source_rows(connection, scope, hits)
        vector_candidates = {
            (hit.source_id, hit.source_version, hit.passage_id): RetrievalCandidate(
                (hit.source_id, hit.source_version, hit.passage_id), hit.source_id,
                hit.source_id, hit.source_version, hit.passage_id,
                ref.text, ref.text_sha256, ref.locator, "vector", hit.score, hit.score,
                None, None, None, (), False,
            )
            for hit, ref in zip(hits, vector_refs)
        }
        seeds: tuple[str, ...] = ()
        node_ids: tuple[str, ...] = ()
        edge_ids: tuple[str, ...] = ()
        diagnostics: dict[str, object] = {"config": config.as_dict(), "mode": mode}
        if mode == "graph":
            nodes, edges = _active_topology(connection, scope, current.graph_name)
            node_refs, edge_refs, skipped = _provenance_rows(
                connection, scope, index_version, query_vector, tuple(nodes),
                tuple(edge["edge_id"] for edge in edges),
            )
            graph_candidates, seeds, node_ids, edge_ids, beam_diagnostics = _beam_candidates(
                nodes, edges, node_refs, edge_refs, hits, config,
            )
            diagnostics.update(beam_diagnostics)
            diagnostics["skipped_source_refs_without_bound_index_passage"] = skipped
            for candidate in graph_candidates:
                previous = vector_candidates.get(candidate.key)
                if previous is not None:
                    vector_candidates[candidate.key] = RetrievalCandidate(
                        candidate.key, candidate.document_id, candidate.source_id,
                        candidate.source_version, candidate.passage_id, candidate.text,
                        candidate.text_sha256, candidate.locator, "both", candidate.score,
                        candidate.cosine, candidate.hop, candidate.via_node_id,
                        candidate.via_edge_id, candidate.supporting_path_ids,
                        candidate.hub_mediated,
                    )
                else:
                    vector_candidates[candidate.key] = candidate
        document_rows = connection.execute(text("""SELECT source_id,source_version,title,status,
              authority,source_type,source_url,filename,corpus_snapshot,metadata
              FROM source_document WHERE tenant_id=:tenant AND project_id=:project
              AND source_id=ANY(:source_ids)"""),
              {"tenant": scope.tenant_id, "project": scope.project_id,
               "source_ids": sorted({candidate.source_id for candidate in vector_candidates.values()})}).mappings()
        document_metadata = {
            (str(row["source_id"]), str(row["source_version"])):
                {key: row[key] for key in ("title", "status", "authority", "source_type",
                                           "source_url", "filename", "corpus_snapshot", "metadata")}
            for row in document_rows
        }
        candidates = tuple(sorted((replace(candidate, source_metadata=document_metadata[
            (candidate.source_id, candidate.source_version)])
            for candidate in vector_candidates.values()), key=lambda item: (-item.score, item.key)))
        source_refs = tuple(RetrievedSourceRef(
            scope.tenant_id, scope.project_id, candidate.source_id,
            candidate.source_version, candidate.passage_id, candidate.text,
            candidate.text_sha256, candidate.locator,
        ) for candidate in candidates)
    return RetrievalResult(HYBRID_RETRIEVER_VERSION,
        sha256(question.encode("utf-8")).hexdigest(), expected_embedding_version,
        index_version, current.graph_version, current.graph_name, config, tuple(hits),
        seeds, node_ids, edge_ids, source_refs, candidates, diagnostics)
