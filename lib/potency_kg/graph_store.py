"""Relational structural graph writer and scoped current-projection reader."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .contracts import ProjectionMode, Scope, ScopedSourceVersion
from .ids import derive_scoped_id


@dataclass(frozen=True)
class GraphWriteReport:
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class ProjectionSnapshot:
    scope: Scope
    mode: ProjectionMode
    allow_partial: bool
    nodes: tuple[dict[str, object], ...]
    edges: tuple[dict[str, object], ...]


def write_structural_version(connection: Connection, source: ScopedSourceVersion) -> GraphWriteReport:
    """Write source/document topology inside the caller-owned import transaction."""
    scope = {"tenant_id": source.tenant_id, "project_id": source.project_id, "source_id": source.source_id, "source_version": source.source_version}
    document_node = derive_scoped_id("document", source.scope, source.source_id, source.source_version)
    document_result = connection.execute(text("""
        INSERT INTO kg_node (tenant_id, project_id, node_id, node_type, attributes)
        VALUES (:tenant_id, :project_id, :node_id, 'SourceDocument', CAST(:attributes AS jsonb))
        ON CONFLICT DO NOTHING
    """), {**scope, "node_id": document_node, "attributes": '{"system_generated":true}'})
    node_count = document_result.rowcount
    edge_count = 0
    passages = connection.execute(text("""
        SELECT passage_id FROM source_passage
        WHERE tenant_id=:tenant_id AND project_id=:project_id AND source_id=:source_id AND source_version=:source_version
        ORDER BY passage_id
    """), scope).scalars().all()
    for passage_id in passages:
        passage_node = derive_scoped_id("passage", source.scope, source.source_id, source.source_version, passage_id)
        edge_id = derive_scoped_id("contains", source.scope, source.source_id, source.source_version, passage_id)
        node_result = connection.execute(text("""
            INSERT INTO kg_node (tenant_id, project_id, node_id, node_type, attributes)
            VALUES (:tenant_id, :project_id, :node_id, 'SourcePassage', CAST(:attributes AS jsonb)) ON CONFLICT DO NOTHING
        """), {**scope, "node_id": passage_node, "attributes": '{"system_generated":true}'})
        edge_result = connection.execute(text("""
            INSERT INTO kg_edge (tenant_id, project_id, edge_id, edge_type, source_node_id, target_node_id, is_evidence, attributes)
            VALUES (:tenant_id, :project_id, :edge_id, 'contains_passage', :document_node, :passage_node, false, CAST(:attributes AS jsonb)) ON CONFLICT DO NOTHING
        """), {**scope, "edge_id": edge_id, "document_node": document_node, "passage_node": passage_node, "attributes": '{"system_generated":true}'})
        node_count += node_result.rowcount
        edge_count += edge_result.rowcount
        connection.execute(text("""
            INSERT INTO kg_node_source_ref (tenant_id, project_id, node_id, source_id, source_version, passage_id)
            VALUES (:tenant_id, :project_id, :node_id, :source_id, :source_version, :passage_id) ON CONFLICT DO NOTHING
        """), {**scope, "node_id": passage_node, "passage_id": passage_id})
        connection.execute(text("""
            INSERT INTO kg_edge_source_ref (tenant_id, project_id, edge_id, source_id, source_version, passage_id)
            VALUES (:tenant_id, :project_id, :edge_id, :source_id, :source_version, :passage_id) ON CONFLICT DO NOTHING
        """), {**scope, "edge_id": edge_id, "passage_id": passage_id})
    return GraphWriteReport(node_count=node_count, edge_count=edge_count)


def load_current_projection(connection: Connection, scope: Scope, mode: ProjectionMode, allow_partial: bool) -> ProjectionSnapshot:
    if mode not in {"structural", "semantic"}:
        raise ValueError("projection mode must be structural or semantic")
    params = {"tenant_id": scope.tenant_id, "project_id": scope.project_id}
    active_heads = connection.execute(
        text("""
            SELECT source_id, source_version FROM source_head
            WHERE tenant_id=:tenant_id AND project_id=:project_id
            ORDER BY source_id
        """),
        params,
    ).all()
    document_ids = [
        derive_scoped_id("document", scope, row.source_id, row.source_version)
        for row in active_heads
    ]
    if not document_ids:
        return ProjectionSnapshot(scope, mode, allow_partial, (), ())
    mode_node_filter = "AND n.node_type IN ('SourceDocument', 'SourcePassage')" if mode == "structural" else ""
    mode_edge_filter = "AND e.edge_type = 'contains_passage'" if mode == "structural" else ""
    query_params = {**params, "document_ids": document_ids}
    nodes = tuple(
        dict(row)
        for row in connection.execute(
            text(f"""
                SELECT DISTINCT n.node_id, n.node_type, n.attributes
                FROM kg_node AS n
                LEFT JOIN kg_node_source_ref AS ref
                  ON (ref.tenant_id, ref.project_id, ref.node_id) = (n.tenant_id, n.project_id, n.node_id)
                LEFT JOIN source_head AS head
                  ON (head.tenant_id, head.project_id, head.source_id, head.source_version) =
                     (ref.tenant_id, ref.project_id, ref.source_id, ref.source_version)
                WHERE n.tenant_id=:tenant_id AND n.project_id=:project_id
                  AND (n.node_id = ANY(:document_ids) OR head.source_id IS NOT NULL)
                  {mode_node_filter}
                ORDER BY n.node_id
            """),
            query_params,
        ).mappings()
    )
    edges = tuple(
        dict(row)
        for row in connection.execute(
            text(f"""
                SELECT DISTINCT e.edge_id, e.edge_type, e.source_node_id, e.target_node_id, e.assertion_id, e.attributes
                FROM kg_edge AS e
                JOIN kg_edge_source_ref AS ref
                  ON (ref.tenant_id, ref.project_id, ref.edge_id) = (e.tenant_id, e.project_id, e.edge_id)
                JOIN source_head AS head
                  ON (head.tenant_id, head.project_id, head.source_id, head.source_version) =
                     (ref.tenant_id, ref.project_id, ref.source_id, ref.source_version)
                WHERE e.tenant_id=:tenant_id AND e.project_id=:project_id
                  {mode_edge_filter}
                ORDER BY e.edge_id
            """),
            params,
        ).mappings()
    )
    return ProjectionSnapshot(scope, mode, allow_partial, nodes, edges)
