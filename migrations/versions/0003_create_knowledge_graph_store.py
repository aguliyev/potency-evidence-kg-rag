"""Create scoped canonical graph and structural-projection records.

Revision ID: 0003_knowledge_graph_store
Revises: 0002_evidence_store
"""

from alembic import op

revision = "0003_knowledge_graph_store"
down_revision = "0002_evidence_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE kg_node (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, node_id TEXT NOT NULL,
      node_type TEXT NOT NULL CHECK (node_type IN ('SourceDocument','SourcePassage','Product','Application','Indication','PotencyStrategy','Assay','MeasuredAttribute','RegulatoryInteraction','ReviewConclusion','Assertion')),
      attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
      PRIMARY KEY (tenant_id, project_id, node_id)
    );
    CREATE TABLE kg_edge (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, edge_id TEXT NOT NULL, edge_type TEXT NOT NULL, source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL,
      assertion_id TEXT, is_evidence BOOLEAN NOT NULL DEFAULT false, attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
      PRIMARY KEY (tenant_id, project_id, edge_id),
      CONSTRAINT ck_kg_edge_type CHECK (edge_type IN ('contains_passage','supported_by','applies_to','measures','used_as','proposed_in','requested_in','revised_by','concluded_in','cites','supersedes_or_supplements')),
      CONSTRAINT fk_kg_edge_source FOREIGN KEY (tenant_id, project_id, source_node_id) REFERENCES kg_node(tenant_id, project_id, node_id),
      CONSTRAINT fk_kg_edge_target FOREIGN KEY (tenant_id, project_id, target_node_id) REFERENCES kg_node(tenant_id, project_id, node_id),
      CONSTRAINT fk_kg_edge_assertion FOREIGN KEY (tenant_id, project_id, assertion_id) REFERENCES assertion(tenant_id, project_id, assertion_id),
      CONSTRAINT ck_kg_evidence_has_assertion CHECK ((NOT is_evidence) OR assertion_id IS NOT NULL)
    );
    CREATE TABLE kg_node_source_ref (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, node_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      PRIMARY KEY (tenant_id, project_id, node_id, source_id, source_version, passage_id),
      CONSTRAINT fk_node_ref_node FOREIGN KEY (tenant_id, project_id, node_id) REFERENCES kg_node(tenant_id, project_id, node_id),
      CONSTRAINT fk_node_ref_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE kg_edge_source_ref (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, edge_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      PRIMARY KEY (tenant_id, project_id, edge_id, source_id, source_version, passage_id),
      CONSTRAINT fk_edge_ref_edge FOREIGN KEY (tenant_id, project_id, edge_id) REFERENCES kg_edge(tenant_id, project_id, edge_id),
      CONSTRAINT fk_edge_ref_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE graph_scope (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, active_graph_name TEXT NOT NULL CHECK (active_graph_name ~ '^kg_[a-f0-9]{16}_[a-f0-9]{16}$'), graph_version TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id)
    );
    CREATE TABLE graph_refresh_run (
      refresh_run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, graph_version TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('structural','semantic')),
      allow_partial BOOLEAN NOT NULL,
      shadow_graph_name TEXT NOT NULL CHECK (shadow_graph_name ~ '^kg_[a-f0-9]{16}_[a-f0-9]{16}$'),
      previous_graph_name TEXT CHECK (previous_graph_name IS NULL OR previous_graph_name ~ '^kg_[a-f0-9]{16}_[a-f0-9]{16}$'),
      omitted JSONB NOT NULL DEFAULT '[]'::jsonb,
      validation JSONB NOT NULL DEFAULT '{}'::jsonb, status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')), error_detail TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
      PRIMARY KEY (tenant_id, project_id, refresh_run_id)
    );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS graph_refresh_run, graph_scope, kg_edge_source_ref, kg_node_source_ref, kg_edge, kg_node CASCADE")
