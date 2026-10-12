"""Bind current semantic graphs to exact indexes and resolved mention provenance.

Revision ID: 0004_current_semantic_state
Revises: 0003_knowledge_graph_store
"""

from alembic import op


revision = "0004_current_semantic_state"
down_revision = "0003_knowledge_graph_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE current_index (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          index_version TEXT NOT NULL,
          embedding_version TEXT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, project_id),
          CONSTRAINT fk_current_index_snapshot
            FOREIGN KEY (tenant_id, project_id, index_version)
            REFERENCES index_snapshot(tenant_id, project_id, index_version)
        );

        ALTER TABLE graph_scope
          ADD COLUMN index_version TEXT,
          ADD COLUMN embedding_version TEXT,
          ADD CONSTRAINT ck_graph_scope_semantic_binding
            CHECK ((index_version IS NULL) = (embedding_version IS NULL)),
          ADD CONSTRAINT fk_graph_scope_index
            FOREIGN KEY (tenant_id, project_id, index_version)
            REFERENCES index_snapshot(tenant_id, project_id, index_version);

        ALTER TABLE graph_refresh_run
          ADD COLUMN index_version TEXT,
          ADD COLUMN embedding_version TEXT,
          ADD CONSTRAINT ck_graph_refresh_semantic_binding
            CHECK ((index_version IS NULL) = (embedding_version IS NULL)),
          ADD CONSTRAINT fk_graph_refresh_index
            FOREIGN KEY (tenant_id, project_id, index_version)
            REFERENCES index_snapshot(tenant_id, project_id, index_version);

        ALTER TABLE passage_processing
          ADD COLUMN validated_extraction_run_id UUID,
          ADD CONSTRAINT fk_processing_validated_extraction
            FOREIGN KEY (validated_extraction_run_id, tenant_id, project_id, source_id,
                         source_version, passage_id)
            REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id,
                                          source_id, source_version, passage_id);

        ALTER TABLE entity_mention
          ADD CONSTRAINT uq_entity_mention_extraction
            UNIQUE (extraction_run_id, tenant_id, project_id, source_id,
                    source_version, passage_id, mention_id);

        CREATE TABLE kg_node_mention_ref (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          node_id TEXT NOT NULL,
          source_id TEXT NOT NULL,
          source_version TEXT NOT NULL,
          passage_id TEXT NOT NULL,
          mention_id TEXT NOT NULL,
          extraction_run_id UUID NOT NULL,
          PRIMARY KEY (tenant_id, project_id, node_id, source_id, source_version,
                       passage_id, mention_id, extraction_run_id),
          CONSTRAINT fk_node_mention_ref_node
            FOREIGN KEY (tenant_id, project_id, node_id)
            REFERENCES kg_node(tenant_id, project_id, node_id),
          CONSTRAINT fk_node_mention_ref_mention
            FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id,
                         source_version, passage_id, mention_id)
            REFERENCES entity_mention(extraction_run_id, tenant_id, project_id,
                                      source_id, source_version, passage_id, mention_id)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS kg_node_mention_ref;
        ALTER TABLE entity_mention DROP CONSTRAINT IF EXISTS uq_entity_mention_extraction;
        ALTER TABLE passage_processing
          DROP CONSTRAINT IF EXISTS fk_processing_validated_extraction,
          DROP COLUMN IF EXISTS validated_extraction_run_id;
        ALTER TABLE graph_refresh_run
          DROP CONSTRAINT IF EXISTS fk_graph_refresh_index,
          DROP CONSTRAINT IF EXISTS ck_graph_refresh_semantic_binding,
          DROP COLUMN IF EXISTS embedding_version,
          DROP COLUMN IF EXISTS index_version;
        ALTER TABLE graph_scope
          DROP CONSTRAINT IF EXISTS fk_graph_scope_index,
          DROP CONSTRAINT IF EXISTS ck_graph_scope_semantic_binding,
          DROP COLUMN IF EXISTS embedding_version,
          DROP COLUMN IF EXISTS index_version;
        DROP TABLE IF EXISTS current_index;
        """
    )
