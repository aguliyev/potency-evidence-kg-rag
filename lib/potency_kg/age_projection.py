"""Transactional Apache AGE projection with last-known-good activation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .contracts import ALLOWED_RELATION_ENDPOINTS, NODE_TYPES, ProjectionMode, Scope
from .embeddings import current_index_binding
from .graph_store import ProjectionSnapshot, load_current_projection
from .ids import derive_graph_name, derive_graph_version


PROJECTION_SCHEMA_VERSION = "age-projection-v1"
_GRAPH_NAME = re.compile(r"^kg_[a-f0-9]{16}_[a-f0-9]{16}$")
_FORBIDDEN_TEXT_KEYS = frozenset(
    {"text", "full_text", "passage_text", "original_quote", "supporting_quote", "exact_quote"}
)


class GraphRefreshError(RuntimeError):
    """A graph refresh was refused or failed without changing the active graph."""


@dataclass(frozen=True)
class GraphCounts:
    node_count: int
    edge_count: int

    def as_tuple(self) -> tuple[int, int]:
        return self.node_count, self.edge_count


@dataclass(frozen=True)
class BoundedGraphTraversal:
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphValidationReport:
    valid: bool
    expected_node_count: int
    actual_node_count: int
    expected_edge_count: int
    actual_edge_count: int
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "expected_node_count": self.expected_node_count,
            "actual_node_count": self.actual_node_count,
            "expected_edge_count": self.expected_edge_count,
            "actual_edge_count": self.actual_edge_count,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class GraphRefreshReport:
    refresh_run_id: UUID | None
    graph_version: str
    graph_name: str
    previous_graph_name: str | None
    mode: ProjectionMode
    allow_partial: bool
    omitted: tuple[tuple[str, str, str, str, str, str], ...]
    validation: GraphValidationReport
    reused: bool = False


FailureInjector = Callable[[str], None]


def _validated_graph_name(graph_name: str) -> str:
    if not isinstance(graph_name, str) or _GRAPH_NAME.fullmatch(graph_name) is None:
        raise ValueError("invalid internally derived AGE graph name")
    return graph_name


def _scope_lock(connection: Connection, scope: Scope, purpose: str, *, shared: bool) -> None:
    if purpose not in {"refresh", "activation"}:
        raise ValueError("invalid graph lock purpose")
    function = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    connection.execute(
        text(f"SELECT {function}(hashtextextended(:lock_key, 0))"),
        {"lock_key": f"potency-kg:{purpose}:{scope.tenant_id}:{scope.project_id}"},
    )


def _prepare_age(connection: Connection) -> None:
    connection.exec_driver_sql("LOAD 'age'")
    connection.exec_driver_sql('SET LOCAL search_path = ag_catalog, "$user", public')


def _decode_agtype(value: object) -> object:
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    rendered = str(value)
    for suffix in ("::vertex", "::edge", "::path"):
        if rendered.endswith(suffix):
            rendered = rendered[: -len(suffix)]
    try:
        return json.loads(rendered)
    except json.JSONDecodeError:
        return rendered.strip('"')


def _cypher(
    connection: Connection,
    graph_name: str,
    query: str,
    columns: str,
    parameters: Mapping[str, object] | None = None,
):
    graph = _validated_graph_name(graph_name)
    if parameters is None:
        statement = text(
            f"SELECT * FROM cypher('{graph}', $cypher${query}$cypher$) AS ({columns})"
        )
        return connection.execute(statement)
    statement = text(
        f"SELECT * FROM cypher('{graph}', $cypher${query}$cypher$, "
        f"CAST(:age_parameters AS agtype)) AS ({columns})"
    )
    return connection.execute(statement, {"age_parameters": json.dumps(parameters, sort_keys=True)})


def _graph_exists(connection: Connection, graph_name: str) -> bool:
    _validated_graph_name(graph_name)
    _prepare_age(connection)
    return connection.execute(
        text("SELECT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name=:name)"),
        {"name": graph_name},
    ).scalar_one()


def _create_graph(connection: Connection, graph_name: str) -> None:
    _validated_graph_name(graph_name)
    _prepare_age(connection)
    connection.execute(text("SELECT create_graph(:name)"), {"name": graph_name})


def _drop_graph(connection: Connection, graph_name: str) -> None:
    _validated_graph_name(graph_name)
    _prepare_age(connection)
    connection.execute(text("SELECT drop_graph(:name, true)"), {"name": graph_name})


def _clean_properties(attributes: object) -> dict[str, object]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise GraphRefreshError("graph attributes must be mappings")
    result = dict(attributes)

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_TEXT_KEYS:
                    raise GraphRefreshError("forbidden passage text property in graph projection")
                inspect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                inspect(nested)

    inspect(result)
    return result


def _node_properties(scope: Scope, row: Mapping[str, object]) -> dict[str, object]:
    attributes = _clean_properties(row.get("attributes"))
    return {
        "logical_id": str(row["node_id"]),
        "node_type": str(row["node_type"]),
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "attributes_json": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
    }


def _edge_properties(scope: Scope, row: Mapping[str, object]) -> dict[str, object]:
    attributes = _clean_properties(row.get("attributes"))
    properties = {
        "logical_id": str(row["edge_id"]),
        "edge_type": str(row["edge_type"]),
        "source_logical_id": str(row["source_node_id"]),
        "target_logical_id": str(row["target_node_id"]),
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "attributes_json": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
    }
    if row.get("assertion_id") is not None:
        properties["assertion_id"] = str(row["assertion_id"])
    return properties


def _build_vertices(connection: Connection, graph_name: str, snapshot: ProjectionSnapshot) -> None:
    _create_graph(connection, graph_name)
    for row in snapshot.nodes:
        label = str(row["node_type"])
        if label not in NODE_TYPES:
            raise GraphRefreshError("node label is not allowlisted")
        _cypher(
            connection,
            graph_name,
            (
                f"CREATE (n:{label}) SET n.logical_id=$logical_id,n.node_type=$node_type,"
                "n.tenant_id=$tenant_id,n.project_id=$project_id,"
                "n.attributes_json=$attributes_json RETURN n.logical_id"
            ),
            "logical_id agtype",
            _node_properties(snapshot.scope, row),
        ).all()


def _build_edges(connection: Connection, graph_name: str, snapshot: ProjectionSnapshot) -> None:
    node_types = {str(row["node_id"]): str(row["node_type"]) for row in snapshot.nodes}
    for row in snapshot.edges:
        edge_type = str(row["edge_type"])
        if edge_type not in ALLOWED_RELATION_ENDPOINTS:
            raise GraphRefreshError("edge type is not allowlisted")
        endpoint_types = (
            node_types.get(str(row["source_node_id"])),
            node_types.get(str(row["target_node_id"])),
        )
        if endpoint_types not in ALLOWED_RELATION_ENDPOINTS[edge_type]:
            raise GraphRefreshError(
                f"edge endpoint types are not allowlisted: {edge_type} {endpoint_types} "
                f"({row['source_node_id']} -> {row['target_node_id']})"
            )
        _cypher(
            connection,
            graph_name,
            (
                "MATCH (source {logical_id: $source_id}), (target {logical_id: $target_id}) "
                f"CREATE (source)-[edge:{edge_type}]->(target) "
                "SET edge.logical_id=$logical_id,edge.edge_type=$edge_type,"
                "edge.source_logical_id=$source_logical_id,"
                "edge.target_logical_id=$target_logical_id,edge.tenant_id=$tenant_id,"
                "edge.project_id=$project_id,edge.attributes_json=$attributes_json "
                "RETURN edge.logical_id"
            ),
            "logical_id agtype",
            {
                "source_id": str(row["source_node_id"]),
                "target_id": str(row["target_node_id"]),
                **_edge_properties(snapshot.scope, row),
            },
        ).all()


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _FORBIDDEN_TEXT_KEYS or _contains_forbidden_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _contains_full_passage(value: object, passage_texts: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_full_passage(item, passage_texts) for item in value.values())
    if isinstance(value, list):
        return any(_contains_full_passage(item, passage_texts) for item in value)
    return isinstance(value, str) and value in passage_texts


def validate_graph(
    connection: Connection,
    graph_name: str,
    snapshot: ProjectionSnapshot,
) -> GraphValidationReport:
    """Compare an AGE graph to the complete logical projection before activation."""
    _validated_graph_name(graph_name)
    _prepare_age(connection)
    node_rows = _cypher(
        connection,
        graph_name,
        "MATCH (n) RETURN n.logical_id, labels(n), properties(n)",
        "logical_id agtype, labels agtype, properties agtype",
    ).all()
    edge_rows = _cypher(
        connection,
        graph_name,
        "MATCH (source)-[edge]->(target) RETURN edge.logical_id, type(edge), source.logical_id, target.logical_id, properties(edge)",
        "logical_id agtype, edge_type agtype, source_id agtype, target_id agtype, properties agtype",
    ).all()
    errors: list[str] = []
    passage_texts = frozenset(
        str(value)
        for value in connection.execute(
            text(
                """SELECT text FROM source_passage
                   WHERE tenant_id=:tenant AND project_id=:project"""
            ),
            {"tenant": snapshot.scope.tenant_id, "project": snapshot.scope.project_id},
        ).scalars()
    )
    expected_nodes = {str(row["node_id"]): row for row in snapshot.nodes}
    expected_edges = {str(row["edge_id"]): row for row in snapshot.edges}
    actual_nodes: dict[str, tuple[object, object]] = {}
    for logical_id, labels, properties in node_rows:
        key = str(_decode_agtype(logical_id))
        actual_nodes[key] = (_decode_agtype(labels), _decode_agtype(properties))
    actual_edges: dict[str, tuple[str, str, str, object]] = {}
    for logical_id, edge_type, source_id, target_id, properties in edge_rows:
        actual_edges[str(_decode_agtype(logical_id))] = (
            str(_decode_agtype(edge_type)),
            str(_decode_agtype(source_id)),
            str(_decode_agtype(target_id)),
            _decode_agtype(properties),
        )
    if set(actual_nodes) != set(expected_nodes):
        errors.append("logical node IDs differ from relational snapshot")
    if len(node_rows) != len(actual_nodes):
        errors.append("duplicate logical node IDs")
    if set(actual_edges) != set(expected_edges):
        errors.append("logical edge IDs differ from relational snapshot")
    if len(edge_rows) != len(actual_edges):
        errors.append("duplicate logical edge IDs")
    for node_id, (labels, properties) in actual_nodes.items():
        expected = expected_nodes.get(node_id)
        label_values = set(labels) if isinstance(labels, list) else {str(labels)}
        if expected is not None and label_values != {str(expected["node_type"])}:
            errors.append(f"node label differs: {node_id}")
        if not isinstance(properties, Mapping):
            errors.append(f"node properties malformed: {node_id}")
        else:
            if properties.get("logical_id") != node_id:
                errors.append(f"node logical ID property differs: {node_id}")
            if expected is not None and properties.get("node_type") != str(expected["node_type"]):
                errors.append(f"node type property differs: {node_id}")
            if (
                properties.get("tenant_id") != snapshot.scope.tenant_id
                or properties.get("project_id") != snapshot.scope.project_id
            ):
                errors.append(f"node scope differs: {node_id}")
        if _contains_forbidden_key(properties):
            errors.append(f"forbidden text property: {node_id}")
        if isinstance(properties, Mapping) and isinstance(properties.get("attributes_json"), str):
            try:
                attributes = json.loads(str(properties["attributes_json"]))
            except json.JSONDecodeError:
                errors.append(f"node attributes malformed: {node_id}")
            else:
                if _contains_forbidden_key(attributes):
                    errors.append(f"forbidden text property: {node_id}")
                if _contains_full_passage(attributes, passage_texts):
                    errors.append(f"full passage text property: {node_id}")
    for edge_id, (edge_type, source_id, target_id, properties) in actual_edges.items():
        expected = expected_edges.get(edge_id)
        if expected is not None and (
            edge_type != str(expected["edge_type"])
            or source_id != str(expected["source_node_id"])
            or target_id != str(expected["target_node_id"])
        ):
            errors.append(f"edge endpoint or type differs: {edge_id}")
        if source_id not in actual_nodes or target_id not in actual_nodes:
            errors.append(f"dangling edge endpoint: {edge_id}")
        else:
            source_labels = actual_nodes[source_id][0]
            target_labels = actual_nodes[target_id][0]
            source_type = next(iter(source_labels), "") if isinstance(source_labels, list) else str(source_labels)
            target_type = next(iter(target_labels), "") if isinstance(target_labels, list) else str(target_labels)
            if (source_type, target_type) not in ALLOWED_RELATION_ENDPOINTS.get(edge_type, frozenset()):
                errors.append(f"edge endpoint types are not allowlisted: {edge_id}")
        if not isinstance(properties, Mapping):
            errors.append(f"edge scope differs: {edge_id}")
        else:
            if properties.get("logical_id") != edge_id:
                errors.append(f"edge logical ID property differs: {edge_id}")
            if properties.get("edge_type") != edge_type:
                errors.append(f"edge type property differs: {edge_id}")
            if (
                properties.get("source_logical_id") != source_id
                or properties.get("target_logical_id") != target_id
            ):
                errors.append(f"edge endpoint properties differ: {edge_id}")
            if (
                properties.get("tenant_id") != snapshot.scope.tenant_id
                or properties.get("project_id") != snapshot.scope.project_id
            ):
                errors.append(f"edge scope differs: {edge_id}")
        if _contains_forbidden_key(properties):
            errors.append(f"forbidden text property: {edge_id}")
        if isinstance(properties, Mapping) and isinstance(properties.get("attributes_json"), str):
            try:
                attributes = json.loads(str(properties["attributes_json"]))
            except json.JSONDecodeError:
                errors.append(f"edge attributes malformed: {edge_id}")
            else:
                if _contains_forbidden_key(attributes):
                    errors.append(f"forbidden text property: {edge_id}")
                if _contains_full_passage(attributes, passage_texts):
                    errors.append(f"full passage text property: {edge_id}")
    return GraphValidationReport(
        not errors,
        len(expected_nodes),
        len(node_rows),
        len(expected_edges),
        len(edge_rows),
        tuple(errors),
    )


def active_graph_name(connection: Connection, scope: Scope) -> str:
    # The shared lock is held to transaction end, so an activation cannot drop
    # the resolved graph between this pointer read and later Cypher statements.
    _scope_lock(connection, scope, "activation", shared=True)
    value = connection.execute(
        text(
            """SELECT active_graph_name FROM graph_scope
               WHERE tenant_id=:tenant AND project_id=:project"""
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id},
    ).scalar_one_or_none()
    if value is None:
        raise LookupError("scope has no active AGE graph")
    return _validated_graph_name(str(value))


def _counts_for_name(connection: Connection, graph_name: str) -> GraphCounts:
    _prepare_age(connection)
    nodes = _cypher(
        connection, graph_name, "MATCH (n) RETURN count(n)", "count agtype"
    ).scalar_one()
    edges = _cypher(
        connection, graph_name, "MATCH ()-[e]->() RETURN count(e)", "count agtype"
    ).scalar_one()
    return GraphCounts(int(_decode_agtype(nodes)), int(_decode_agtype(edges)))


def graph_counts(connection: Connection, scope: Scope) -> GraphCounts:
    """Resolve the pointer and read the active graph within the caller's transaction."""
    return _counts_for_name(connection, active_graph_name(connection, scope))


def _bounded_graph_nodes(
    connection: Connection,
    graph_name: str,
    node_ids: tuple[str, ...],
) -> dict[str, str]:
    unique_ids = tuple(dict.fromkeys(node_ids))
    if not unique_ids:
        return {}
    if len(unique_ids) > 16:
        raise ValueError("bounded graph node lookup exceeds the retrieval maximum")
    rows = _cypher(
        connection,
        graph_name,
        (
            "MATCH (n) WHERE n.logical_id IN $node_ids "
            "RETURN n.logical_id, n.node_type ORDER BY n.logical_id "
            f"LIMIT {len(unique_ids)}"
        ),
        "logical_id agtype, node_type agtype",
        {"node_ids": list(unique_ids)},
    )
    return {
        str(_decode_agtype(logical_id)): str(_decode_agtype(node_type))
        for logical_id, node_type in rows
    }


def _bounded_adjacent_edges(
    connection: Connection,
    graph_name: str,
    node_ids: tuple[str, ...],
    limit: int,
) -> tuple[dict[str, str], ...]:
    unique_ids = tuple(dict.fromkeys(node_ids))
    if not unique_ids or limit == 0:
        return ()
    if len(unique_ids) > 30 or not isinstance(limit, int) or not 0 <= limit <= 50:
        raise ValueError("bounded AGE traversal exceeds the retrieval maximum")
    return tuple(
        {
            "edge_id": str(_decode_agtype(edge_id)),
            "source_node_id": str(_decode_agtype(source_id)),
            "target_node_id": str(_decode_agtype(target_id)),
        }
        for edge_id, source_id, target_id in _cypher(
            connection,
            graph_name,
            (
                "MATCH (source)-[edge]->(target) "
                "WHERE source.logical_id IN $node_ids OR target.logical_id IN $node_ids "
                "RETURN edge.logical_id, source.logical_id, target.logical_id "
                f"ORDER BY edge.logical_id LIMIT {limit}"
            ),
            "edge_id agtype, source_id agtype, target_id agtype",
            {"node_ids": list(unique_ids)},
        )
    )


def bounded_graph_traversal(
    connection: Connection,
    graph_name: str,
    passage_node_ids: tuple[str, ...],
    semantic_candidate_ids: tuple[str, ...],
    *,
    semantic_seed_limit: int,
    hops: int,
    node_limit: int,
    edge_limit: int,
) -> BoundedGraphTraversal:
    """Traverse one active AGE graph without materializing its complete topology."""
    limits = (semantic_seed_limit, hops, node_limit, edge_limit)
    if any(not isinstance(value, int) or value < 0 for value in limits) or (
        semantic_seed_limit > 8 or hops > 2 or node_limit > 30 or edge_limit > 50
        or len(passage_node_ids) > 8 or len(semantic_candidate_ids) > 8
    ):
        raise ValueError("bounded AGE traversal exceeds the retrieval maximum")
    _prepare_age(connection)
    active_types = _bounded_graph_nodes(
        connection,
        graph_name,
        (*passage_node_ids, *semantic_candidate_ids),
    )
    seeds: list[str] = []
    for node_id in semantic_candidate_ids:
        if (
            active_types.get(node_id) not in {None, "SourceDocument", "SourcePassage"}
            and node_id not in seeds
        ):
            seeds.append(node_id)
            if len(seeds) >= semantic_seed_limit:
                break
    if semantic_seed_limit == 0:
        seeds.clear()
    roots = tuple(seeds) or passage_node_ids
    selected_nodes: list[str] = []
    for node_id in (*passage_node_ids, *roots):
        if node_id in active_types and node_id not in selected_nodes and len(selected_nodes) < node_limit:
            selected_nodes.append(node_id)
    selected_edges: list[str] = []
    frontier = [node_id for node_id in roots if node_id in selected_nodes]
    seen_frontier = set(frontier)
    for _hop in range(hops):
        following: list[str] = []
        for edge in _bounded_adjacent_edges(
            connection, graph_name, tuple(frontier), edge_limit
        ):
            edge_id = edge["edge_id"]
            source = edge["source_node_id"]
            target = edge["target_node_id"]
            frontier_node = source if source in frontier else target
            other = target if frontier_node == source else source
            if other not in selected_nodes and len(selected_nodes) >= node_limit:
                continue
            if edge_id not in selected_edges:
                if len(selected_edges) >= edge_limit:
                    continue
                selected_edges.append(edge_id)
            if other not in selected_nodes:
                selected_nodes.append(other)
            if other not in seen_frontier:
                seen_frontier.add(other)
                following.append(other)
        frontier = following
        if not frontier or len(selected_edges) >= edge_limit:
            break
    return BoundedGraphTraversal(tuple(seeds), tuple(selected_nodes), tuple(selected_edges))


def _projection_inputs(
    connection: Connection, scope: Scope
) -> tuple[list[tuple[str, str]], list[str], list[str], tuple[tuple[str, str, str, str, str, str], ...]]:
    params = {"tenant": scope.tenant_id, "project": scope.project_id}
    active_heads = [
        (str(row.source_id), str(row.source_version))
        for row in connection.execute(
            text(
                """SELECT source_id,source_version FROM source_head
                   WHERE tenant_id=:tenant AND project_id=:project ORDER BY source_id"""
            ),
            params,
        )
    ]
    processing = connection.execute(
        text(
            """SELECT processing.source_id,processing.source_version,processing.passage_id,
                      processing.state,processing.active_extraction_run_id
               FROM passage_processing AS processing
               JOIN source_head AS head
                 ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                    (processing.tenant_id,processing.project_id,processing.source_id,processing.source_version)
               WHERE processing.tenant_id=:tenant AND processing.project_id=:project
               ORDER BY processing.source_id,processing.source_version,processing.passage_id"""
        ),
        params,
    ).all()
    extraction_ids = sorted(
        str(row.active_extraction_run_id)
        for row in processing
        if row.active_extraction_run_id is not None and str(row.state) == "extracted"
    )
    omissions = tuple(
        (
            scope.tenant_id,
            scope.project_id,
            str(row.source_id),
            str(row.source_version),
            str(row.passage_id),
            str(row.state),
        )
        for row in processing
        if str(row.state) in {"selected", "failed"}
    )
    profile_versions = sorted(
        str(value)
        for value in connection.execute(
            text(
                """SELECT DISTINCT run.profile_version
                   FROM passage_processing AS processing
                   JOIN source_head AS head
                     ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                        (processing.tenant_id,processing.project_id,processing.source_id,processing.source_version)
                   JOIN llm_extraction_run AS run
                     ON (run.extraction_run_id,run.tenant_id,run.project_id,run.source_id,
                         run.source_version,run.passage_id) =
                        (processing.active_extraction_run_id,processing.tenant_id,processing.project_id,
                         processing.source_id,processing.source_version,processing.passage_id)
                   WHERE processing.tenant_id=:tenant AND processing.project_id=:project
                     AND processing.state='extracted'"""
            ),
            params,
        ).scalars()
    )
    return active_heads, extraction_ids, profile_versions, omissions


def _inject(failure_at: str | None, injector: FailureInjector | None, point: str) -> None:
    if failure_at == point:
        raise RuntimeError(f"injected graph refresh failure at {point}")
    if injector is not None:
        injector(point)


def _record_failed_attempt(
    engine: Engine,
    refresh_run_id: UUID,
    scope: Scope,
    graph_version: str,
    mode: ProjectionMode,
    allow_partial: bool,
    shadow: str,
    previous: str | None,
    omitted: tuple[tuple[str, str, str, str, str, str], ...],
    index_version: str | None,
    embedding_version: str | None,
    detail: str,
) -> None:
    with engine.begin() as connection:
        if _graph_exists(connection, shadow):
            active = connection.execute(
                text(
                    """SELECT active_graph_name FROM graph_scope
                       WHERE tenant_id=:tenant AND project_id=:project"""
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).scalar_one_or_none()
            if active != shadow:
                _drop_graph(connection, shadow)
        connection.execute(
            text(
                """INSERT INTO graph_refresh_run
                   (refresh_run_id,tenant_id,project_id,graph_version,mode,allow_partial,
                    index_version,embedding_version,
                    shadow_graph_name,previous_graph_name,omitted,validation,status,error_detail,finished_at)
                   VALUES (:run_id,:tenant,:project,:version,:mode,:allow_partial,
                           :index_version,:embedding_version,:shadow,:previous,
                           CAST(:omitted AS jsonb),'{}','failed',:detail,now())"""
            ),
            {
                "run_id": refresh_run_id,
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "version": graph_version,
                "mode": mode,
                "allow_partial": allow_partial,
                "index_version": index_version,
                "embedding_version": embedding_version,
                "shadow": shadow,
                "previous": previous,
                "omitted": json.dumps(omitted),
                "detail": detail,
            },
        )


def refresh_graph(
    engine: Engine,
    scope: Scope,
    mode: Literal["structural", "semantic"] = "semantic",
    allow_partial: bool = False,
    force: bool = False,
    *,
    failure_at: str | None = None,
    failure_injector: FailureInjector | None = None,
) -> GraphRefreshReport:
    """Build, validate, and atomically activate a scope-derived shadow graph."""
    if mode not in {"structural", "semantic"}:
        raise ValueError("projection mode must be structural or semantic")
    refresh_run_id = uuid4()
    previous: str | None = None
    graph_version = "graph-unresolved"
    shadow = derive_graph_name(scope, graph_version)
    omissions: tuple[tuple[str, str, str, str, str, str], ...] = ()
    index_version: str | None = None
    embedding_version: str | None = None
    try:
        with engine.begin() as connection:
            # Serialize refresh owners without blocking readers during shadow build.
            _scope_lock(connection, scope, "refresh", shared=False)
            heads, extraction_ids, profile_versions, unresolved = _projection_inputs(connection, scope)
            omissions = unresolved if mode == "semantic" else ()
            profile_version = "|".join(profile_versions) if profile_versions else "not-run"
            if mode == "semantic" and (not omissions or allow_partial):
                try:
                    binding = current_index_binding(connection, scope)
                except (LookupError, ValueError):
                    raise GraphRefreshError(
                        "semantic refresh requires exact current index membership"
                    ) from None
                index_version = binding.index_version
                embedding_version = binding.embedding_version
            graph_version = derive_graph_version(
                scope,
                PROJECTION_SCHEMA_VERSION,
                heads,
                extraction_ids if mode == "semantic" else (),
                mode,
                allow_partial,
                (item[:5] for item in omissions),
                profile_version,
                index_version,
                embedding_version,
            )
            base_name = derive_graph_name(scope, graph_version)
            current = connection.execute(
                text(
                    """SELECT active_graph_name,graph_version FROM graph_scope
                       WHERE tenant_id=:tenant AND project_id=:project"""
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).one_or_none()
            previous = str(current.active_graph_name) if current is not None else None
            if current is not None and str(current.graph_version) == graph_version and not force:
                snapshot = load_current_projection(connection, scope, mode, allow_partial)
                validation = validate_graph(connection, str(current.active_graph_name), snapshot)
                if validation.valid:
                    connection.execute(
                        text(
                            """INSERT INTO graph_refresh_run
                               (refresh_run_id,tenant_id,project_id,graph_version,mode,
                                allow_partial,index_version,embedding_version,
                                shadow_graph_name,previous_graph_name,omitted,
                                validation,status,finished_at)
                               VALUES (:run_id,:tenant,:project,:version,:mode,:allow_partial,
                                       :index_version,:embedding_version,:shadow,:previous,CAST(:omitted AS jsonb),
                                       CAST(:validation AS jsonb),'succeeded',now())"""
                        ),
                        {
                            "run_id": refresh_run_id,
                            "tenant": scope.tenant_id,
                            "project": scope.project_id,
                            "version": graph_version,
                            "mode": mode,
                            "allow_partial": allow_partial,
                            "index_version": index_version,
                            "embedding_version": embedding_version,
                            "shadow": str(current.active_graph_name),
                            "previous": previous,
                            "omitted": json.dumps(omissions),
                            "validation": json.dumps(validation.as_dict(), sort_keys=True),
                        },
                    )
                    return GraphRefreshReport(
                        refresh_run_id, graph_version, str(current.active_graph_name), previous, mode,
                        allow_partial, omissions, validation, True,
                    )
            shadow = base_name if not force and not _graph_exists(connection, base_name) else derive_graph_name(
                scope, f"{graph_version}:attempt:{refresh_run_id}"
            )
            if mode == "semantic" and omissions and not allow_partial:
                raise GraphRefreshError("semantic refresh has unresolved selected or failed passages")
            snapshot = load_current_projection(connection, scope, mode, allow_partial)
            connection.execute(
                text(
                    """INSERT INTO graph_refresh_run
                       (refresh_run_id,tenant_id,project_id,graph_version,mode,allow_partial,
                        index_version,embedding_version,
                        shadow_graph_name,previous_graph_name,omitted,status)
                       VALUES (:run_id,:tenant,:project,:version,:mode,:allow_partial,
                               :index_version,:embedding_version,:shadow,
                               :previous,CAST(:omitted AS jsonb),'running')"""
                ),
                {
                    "run_id": refresh_run_id,
                    "tenant": scope.tenant_id,
                    "project": scope.project_id,
                    "version": graph_version,
                    "mode": mode,
                    "allow_partial": allow_partial,
                    "index_version": index_version,
                    "embedding_version": embedding_version,
                    "shadow": shadow,
                    "previous": previous,
                    "omitted": json.dumps(omissions),
                },
            )
            _build_vertices(connection, shadow, snapshot)
            _inject(failure_at, failure_injector, "after_vertices")
            _build_edges(connection, shadow, snapshot)
            _inject(failure_at, failure_injector, "after_edges")
            validation = validate_graph(connection, shadow, snapshot)
            if not validation.valid:
                raise GraphRefreshError("shadow graph validation failed")
            _inject(failure_at, failure_injector, "after_validation")
            _inject(failure_at, failure_injector, "before_activation")
            # Wait for old-graph readers, then hold the exclusive boundary lock
            # through pointer replacement and exact previous-graph removal.
            _scope_lock(connection, scope, "activation", shared=False)
            connection.execute(
                text(
                    """INSERT INTO graph_scope
                       (tenant_id,project_id,active_graph_name,graph_version,index_version,embedding_version)
                       VALUES (:tenant,:project,:graph_name,:version,:index_version,:embedding_version)
                       ON CONFLICT (tenant_id,project_id) DO UPDATE
                       SET active_graph_name=EXCLUDED.active_graph_name,
                           graph_version=EXCLUDED.graph_version,
                           index_version=EXCLUDED.index_version,
                           embedding_version=EXCLUDED.embedding_version,updated_at=now()"""
                ),
                {
                    "tenant": scope.tenant_id,
                    "project": scope.project_id,
                    "graph_name": shadow,
                    "version": graph_version,
                    "index_version": index_version,
                    "embedding_version": embedding_version,
                },
            )
            _inject(failure_at, failure_injector, "after_activation")
            if previous is not None and previous != shadow:
                _drop_graph(connection, previous)
            _inject(failure_at, failure_injector, "after_cleanup")
            connection.execute(
                text(
                    """UPDATE graph_refresh_run
                       SET validation=CAST(:validation AS jsonb),status='succeeded',finished_at=now()
                       WHERE tenant_id=:tenant AND project_id=:project AND refresh_run_id=:run_id"""
                ),
                {
                    "tenant": scope.tenant_id,
                    "project": scope.project_id,
                    "run_id": refresh_run_id,
                    "validation": json.dumps(validation.as_dict(), sort_keys=True),
                },
            )
        return GraphRefreshReport(
            refresh_run_id, graph_version, shadow, previous, mode, allow_partial,
            omissions, validation, False,
        )
    except Exception as error:
        detail = str(error) if isinstance(error, (GraphRefreshError, RuntimeError)) else "graph refresh failed"
        try:
            _record_failed_attempt(
                engine, refresh_run_id, scope, graph_version, mode, allow_partial, shadow,
                previous, omissions, index_version, embedding_version, detail,
            )
        except Exception:
            raise GraphRefreshError("graph refresh failed and failure history could not be recorded") from None
        if isinstance(error, GraphRefreshError):
            raise
        raise GraphRefreshError(detail) from None
