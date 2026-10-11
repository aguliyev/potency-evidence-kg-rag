"""Create immutable evidence inputs, processing history, and current state.

Revision ID: 0002_evidence_store
Revises: 0001_extensions
"""

from alembic import op

revision = "0002_evidence_store"
down_revision = "0001_extensions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE processing_state AS ENUM ('not_run', 'structural_only', 'selected', 'extracted', 'failed')")
    op.execute("""
    CREATE TABLE import_run (
      run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL,
      descriptor_sha256 TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
      started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
      PRIMARY KEY (tenant_id, project_id, run_id)
    );
    CREATE TABLE source_document (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL,
      title TEXT NOT NULL, source_sha256 TEXT NOT NULL, expected_passage_count INTEGER NOT NULL CHECK (expected_passage_count >= 0),
      status TEXT NOT NULL CHECK (length(trim(status)) > 0), authority TEXT, source_type TEXT, source_url TEXT, filename TEXT, corpus_snapshot TEXT,
      metadata JSONB NOT NULL DEFAULT '{}'::jsonb, imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, source_id, source_version)
    );
    CREATE TABLE source_head (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL,
      activated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, source_id),
      CONSTRAINT fk_source_head_document FOREIGN KEY (tenant_id, project_id, source_id, source_version)
        REFERENCES source_document (tenant_id, project_id, source_id, source_version)
    );
    CREATE TABLE source_passage (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      text TEXT NOT NULL, text_sha256 TEXT NOT NULL, locator JSONB NOT NULL DEFAULT '{}'::jsonb,
      subject_scope JSONB NOT NULL DEFAULT '{}'::jsonb, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
      PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_source_passage_document FOREIGN KEY (tenant_id, project_id, source_id, source_version)
        REFERENCES source_document (tenant_id, project_id, source_id, source_version)
    );
    CREATE TABLE import_error (
      error_id UUID NOT NULL, run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL,
      source_id TEXT, source_version TEXT, passage_id TEXT, code TEXT NOT NULL, detail TEXT NOT NULL CHECK (length(trim(detail)) > 0), created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, error_id),
      CONSTRAINT fk_import_error_run FOREIGN KEY (tenant_id, project_id, run_id)
        REFERENCES import_run(tenant_id, project_id, run_id)
    );
    CREATE TABLE embedding_run (
      embedding_run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      embedding_version TEXT NOT NULL, model TEXT NOT NULL, sanitized_origin TEXT NOT NULL, credential_env TEXT NOT NULL,
      dimensions INTEGER NOT NULL CHECK (dimensions > 0), encoding_format TEXT NOT NULL DEFAULT 'float', normalization TEXT NOT NULL,
      input_sha256 TEXT NOT NULL, vector vector, usage JSONB NOT NULL DEFAULT '{}'::jsonb, attempts INTEGER NOT NULL DEFAULT 1 CHECK (attempts > 0),
      status TEXT NOT NULL CHECK (status IN ('succeeded','failed')), error_detail TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, embedding_run_id),
      UNIQUE (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_embedding_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id)
        REFERENCES source_passage (tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE passage_embedding (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      embedding_run_id UUID NOT NULL, PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id, embedding_run_id),
      CONSTRAINT fk_passage_embedding_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id)
        REFERENCES source_passage (tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_passage_embedding_run FOREIGN KEY (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id)
        REFERENCES embedding_run (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE index_snapshot (
      index_version TEXT NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, embedding_version TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (tenant_id, project_id, index_version)
    );
    CREATE TABLE index_member (
      index_version TEXT NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL, embedding_run_id UUID NOT NULL,
      PRIMARY KEY (index_version, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_index_member_snapshot FOREIGN KEY (tenant_id, project_id, index_version) REFERENCES index_snapshot(tenant_id, project_id, index_version),
      CONSTRAINT fk_index_member_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_index_member_embedding FOREIGN KEY (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id)
        REFERENCES embedding_run(embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE jev_response (
      response_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      jev_response_version TEXT NOT NULL, model TEXT NOT NULL, request_id TEXT, credential_env TEXT NOT NULL DEFAULT 'TYPESAFE_API_KEY',
      questions JSONB NOT NULL, criteria JSONB NOT NULL DEFAULT '{}'::jsonb, probabilities JSONB NOT NULL,
      normalization TEXT NOT NULL DEFAULT 'redaction-v1', prompt_hash TEXT, bundle_hash TEXT, input_sha256 TEXT NOT NULL,
      profile_name TEXT, profile_aliases JSONB NOT NULL DEFAULT '{}'::jsonb, profile_source_sha256 TEXT, profile_version TEXT NOT NULL,
      usage JSONB NOT NULL DEFAULT '{}'::jsonb, attempts INTEGER NOT NULL DEFAULT 1 CHECK (attempts > 0),
      status TEXT NOT NULL CHECK (status IN ('succeeded','failed')), error_detail TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, response_id),
      UNIQUE (response_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_jev_response_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE jev_selection (
      selection_id UUID NOT NULL, response_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      thresholds JSONB NOT NULL, selected_bundles JSONB NOT NULL, jev_selection_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, selection_id),
      UNIQUE (selection_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_jev_selection_response FOREIGN KEY (response_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES jev_response(response_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE llm_extraction_run (
      extraction_run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      extraction_version TEXT NOT NULL, model TEXT NOT NULL, sanitized_origin TEXT NOT NULL, credential_env TEXT NOT NULL,
      instructor_mode TEXT NOT NULL DEFAULT 'openai', selected_bundles JSONB NOT NULL, prompt_hash TEXT, schema_hash TEXT,
      normalization TEXT NOT NULL DEFAULT 'redaction-v1', input_sha256 TEXT NOT NULL,
      profile_name TEXT, profile_aliases JSONB NOT NULL DEFAULT '{}'::jsonb, profile_source_sha256 TEXT, profile_version TEXT NOT NULL,
      usage JSONB NOT NULL DEFAULT '{}'::jsonb, attempts INTEGER NOT NULL DEFAULT 1 CHECK (attempts > 0),
      status TEXT NOT NULL CHECK (status IN ('succeeded','failed')), response JSONB, error_detail TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, extraction_run_id),
      UNIQUE (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_extraction_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE validation_failure (
      validation_failure_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, stage TEXT NOT NULL, immutable_run_id UUID NOT NULL, code TEXT NOT NULL, json_path TEXT, detail TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, validation_failure_id)
    );
    CREATE TABLE materialization_run (
      materialization_run_id UUID NOT NULL, extraction_run_id UUID NOT NULL, tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      previous_active_extraction_run_id UUID, status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
      counts JSONB NOT NULL DEFAULT '{}'::jsonb, error_detail TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
      PRIMARY KEY (tenant_id, project_id, materialization_run_id),
      CONSTRAINT fk_materialization_extraction FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_materialization_previous_extraction FOREIGN KEY (previous_active_extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE passage_processing (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      state processing_state NOT NULL, reason TEXT NOT NULL CHECK (length(trim(reason)) > 0), active_jev_selection_id UUID, active_extraction_run_id UUID, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_processing_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_processing_selection FOREIGN KEY (active_jev_selection_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES jev_selection(selection_id, tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_processing_extraction FOREIGN KEY (active_extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE redaction_mention (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL, mention_id TEXT NOT NULL,
      original_start INTEGER NOT NULL, original_end INTEGER NOT NULL, model_start INTEGER NOT NULL, model_end INTEGER NOT NULL, PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id, mention_id),
      CONSTRAINT fk_redaction_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE entity_mention (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL, mention_id TEXT NOT NULL, entity_type TEXT NOT NULL, text TEXT NOT NULL,
      extraction_run_id UUID, original_start INTEGER, original_end INTEGER, model_start INTEGER, model_end INTEGER, exact_quote TEXT,
      PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id, mention_id),
      CONSTRAINT fk_entity_mention_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_entity_mention_extraction FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE alias_candidate (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL, alias_id TEXT NOT NULL, candidate TEXT NOT NULL,
      extraction_run_id UUID, proposer TEXT, extraction_version TEXT, original_quote TEXT,
      status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
      PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id, alias_id),
      CONSTRAINT fk_alias_candidate_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_alias_candidate_extraction FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    CREATE TABLE assertion (
      tenant_id TEXT NOT NULL, project_id TEXT NOT NULL, assertion_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL, passage_id TEXT NOT NULL,
      extraction_run_id UUID, claim TEXT NOT NULL CHECK (length(trim(claim)) > 0), chronology_state TEXT NOT NULL, review_state TEXT NOT NULL,
      graph_representation JSONB NOT NULL DEFAULT '{}'::jsonb, authority TEXT, status TEXT, original_quote TEXT,
      original_start INTEGER, original_end INTEGER, model_start INTEGER, model_end INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, project_id, assertion_id),
      CONSTRAINT fk_assertion_passage FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
      CONSTRAINT fk_assertion_extraction FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
    );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS assertion, alias_candidate, entity_mention, redaction_mention, passage_processing, materialization_run, validation_failure, llm_extraction_run, jev_selection, jev_response, index_member, index_snapshot, passage_embedding, embedding_run, import_error, source_passage, source_head, source_document, import_run CASCADE")
    op.execute("DROP TYPE IF EXISTS processing_state")
