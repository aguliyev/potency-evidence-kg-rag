"""Create the tenant/project-scoped alias registry.

Revision ID: 0005_scoped_alias_registry
Revises: 0004_current_semantic_state
"""

from alembic import op


revision = "0005_scoped_alias_registry"
down_revision = "0004_current_semantic_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE aliases (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          alias_id TEXT NOT NULL,
          alias_value TEXT NOT NULL,
          normalized_alias TEXT NOT NULL,
          entity_type TEXT,
          canonical_key TEXT,
          canonical_name TEXT,
          status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected')),
          managed_by TEXT NOT NULL CHECK (managed_by IN ('static_config','review')),
          normalization_version TEXT NOT NULL,
          confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          decided_at TIMESTAMPTZ,
          decided_by TEXT,
          decision_note TEXT,
          PRIMARY KEY (tenant_id, project_id, alias_id),
          CONSTRAINT ck_aliases_nonempty_values CHECK (
            length(trim(alias_value)) > 0 AND length(trim(normalized_alias)) > 0
          ),
          CONSTRAINT ck_aliases_entity_type CHECK (
            entity_type IS NULL OR entity_type IN ('Product','Assay')
          ),
          CONSTRAINT ck_aliases_approved_target CHECK (
            status <> 'approved' OR
            (entity_type IS NOT NULL AND length(trim(canonical_key)) > 0
             AND length(trim(canonical_name)) > 0)
          ),
          CONSTRAINT ck_aliases_static_status CHECK (
            managed_by <> 'static_config' OR status IN ('approved','rejected')
          ),
          CONSTRAINT ck_aliases_static_target_and_retirement CHECK (
            managed_by <> 'static_config' OR
            (
              entity_type IS NOT NULL AND length(trim(canonical_key)) > 0
              AND length(trim(canonical_name)) > 0
              AND (
                status = 'approved' OR
                (status = 'rejected' AND decided_at IS NOT NULL
                 AND decided_by = 'static_config')
              )
            )
          ),
          CONSTRAINT ck_aliases_rejected_decision CHECK (
            status <> 'rejected' OR (decided_at IS NOT NULL AND decided_by IS NOT NULL)
          )
        );

        CREATE UNIQUE INDEX uq_aliases_one_approved_target
          ON aliases (tenant_id, project_id, entity_type, normalized_alias)
          WHERE status = 'approved';

        CREATE TABLE alias_discovery_run (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          discovery_run_id UUID NOT NULL,
          method TEXT NOT NULL CHECK (method IN ('deterministic','llm')),
          source_id TEXT NOT NULL,
          source_version TEXT NOT NULL,
          input_sha256 TEXT NOT NULL,
          detector_version TEXT NOT NULL,
          model TEXT,
          sanitized_origin TEXT,
          credential_env TEXT,
          prompt_hash TEXT,
          schema_hash TEXT,
          status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
          usage JSONB NOT NULL DEFAULT '{}'::jsonb,
          attempts INTEGER NOT NULL DEFAULT 1 CHECK (attempts > 0),
          counts JSONB NOT NULL DEFAULT '{}'::jsonb,
          error_detail TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ,
          PRIMARY KEY (tenant_id, project_id, discovery_run_id),
          UNIQUE (discovery_run_id, tenant_id, project_id, source_id, source_version),
          CONSTRAINT fk_alias_discovery_run_document FOREIGN KEY
            (tenant_id, project_id, source_id, source_version)
            REFERENCES source_document(tenant_id, project_id, source_id, source_version)
        );

        CREATE TABLE alias_evidence (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          evidence_id TEXT NOT NULL,
          alias_id TEXT NOT NULL,
          source_type TEXT NOT NULL CHECK (source_type IN
            ('static_config','deterministic_rule','llm_mined','extraction')),
          source_id TEXT,
          source_version TEXT,
          passage_id TEXT,
          extraction_run_id UUID,
          discovery_run_id UUID,
          exact_quote TEXT,
          original_start INTEGER,
          original_end INTEGER,
          detector_name TEXT NOT NULL,
          detector_version TEXT NOT NULL,
          model TEXT,
          prompt_hash TEXT,
          schema_hash TEXT,
          input_sha256 TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, project_id, evidence_id),
          CONSTRAINT fk_alias_evidence_alias FOREIGN KEY (tenant_id, project_id, alias_id)
            REFERENCES aliases(tenant_id, project_id, alias_id),
          CONSTRAINT fk_alias_evidence_passage FOREIGN KEY
            (tenant_id, project_id, source_id, source_version, passage_id)
            REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
          CONSTRAINT ck_alias_evidence_source_identity CHECK (
            (source_id IS NULL AND source_version IS NULL AND passage_id IS NULL) OR
            (source_id IS NOT NULL AND source_version IS NOT NULL AND passage_id IS NOT NULL)
          ),
          CONSTRAINT fk_alias_evidence_extraction FOREIGN KEY
            (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
            REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id),
          CONSTRAINT ck_alias_evidence_extraction_scope CHECK (
            extraction_run_id IS NULL OR
            (source_id IS NOT NULL AND source_version IS NOT NULL AND passage_id IS NOT NULL)
          ),
          CONSTRAINT fk_alias_evidence_discovery FOREIGN KEY
            (tenant_id, project_id, discovery_run_id)
            REFERENCES alias_discovery_run(tenant_id, project_id, discovery_run_id)
        );

        ALTER TABLE jev_response
          ADD COLUMN alias_catalog_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
        ALTER TABLE jev_response
          ALTER COLUMN alias_catalog_version DROP DEFAULT;
        ALTER TABLE llm_extraction_run
          ADD COLUMN alias_catalog_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
        ALTER TABLE llm_extraction_run
          ALTER COLUMN alias_catalog_version DROP DEFAULT;

        INSERT INTO aliases (
          tenant_id, project_id, alias_id, alias_value, normalized_alias,
          entity_type, canonical_key, canonical_name, status, managed_by,
          normalization_version
        )
        SELECT tenant_id, project_id, alias_id, candidate, lower(trim(candidate)),
               NULL, NULL, NULL, 'pending', 'review', 'legacy-unversioned'
        FROM alias_candidate;

        INSERT INTO alias_evidence (
          tenant_id, project_id, evidence_id, alias_id, source_type,
          source_id, source_version, passage_id, extraction_run_id, exact_quote,
          detector_name, detector_version, input_sha256
        )
        SELECT candidate.tenant_id, candidate.project_id,
               'legacy-' || md5(concat_ws('|', candidate.tenant_id, candidate.project_id,
                 candidate.source_id, candidate.source_version, candidate.passage_id,
                 candidate.alias_id)),
               candidate.alias_id, 'extraction', candidate.source_id,
               candidate.source_version, candidate.passage_id,
               candidate.extraction_run_id, candidate.original_quote,
               COALESCE(candidate.proposer, 'legacy_alias_candidate'),
               COALESCE(candidate.extraction_version, 'legacy-unversioned'),
               COALESCE(run.input_sha256, passage.text_sha256)
        FROM alias_candidate AS candidate
        JOIN source_passage AS passage
          ON (passage.tenant_id, passage.project_id, passage.source_id,
              passage.source_version, passage.passage_id) =
             (candidate.tenant_id, candidate.project_id, candidate.source_id,
              candidate.source_version, candidate.passage_id)
        LEFT JOIN llm_extraction_run AS run
          ON (run.extraction_run_id, run.tenant_id, run.project_id, run.source_id,
              run.source_version, run.passage_id) =
             (candidate.extraction_run_id, candidate.tenant_id, candidate.project_id,
              candidate.source_id, candidate.source_version, candidate.passage_id);

        DO $$
        DECLARE
          legacy_count BIGINT;
          alias_count BIGINT;
          evidence_count BIGINT;
        BEGIN
          SELECT count(*) INTO legacy_count FROM alias_candidate;
          SELECT count(*) INTO alias_count FROM aliases WHERE normalization_version = 'legacy-unversioned';
          SELECT count(*) INTO evidence_count FROM alias_evidence WHERE source_type = 'extraction';
          IF legacy_count <> alias_count OR legacy_count <> evidence_count THEN
            RAISE EXCEPTION 'legacy alias migration count mismatch: candidates %, aliases %, evidence %',
              legacy_count, alias_count, evidence_count;
          END IF;
        END $$;

        DROP TABLE alias_candidate;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE TABLE alias_candidate (
          tenant_id TEXT NOT NULL,
          project_id TEXT NOT NULL,
          source_id TEXT NOT NULL,
          source_version TEXT NOT NULL,
          passage_id TEXT NOT NULL,
          alias_id TEXT NOT NULL,
          candidate TEXT NOT NULL,
          extraction_run_id UUID,
          proposer TEXT,
          extraction_version TEXT,
          original_quote TEXT,
          status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
          PRIMARY KEY (tenant_id, project_id, source_id, source_version, passage_id, alias_id),
          CONSTRAINT fk_alias_candidate_passage FOREIGN KEY
            (tenant_id, project_id, source_id, source_version, passage_id)
            REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id),
          CONSTRAINT fk_alias_candidate_extraction FOREIGN KEY
            (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
            REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)
        );

        INSERT INTO alias_candidate (
          tenant_id, project_id, source_id, source_version, passage_id, alias_id,
          candidate, extraction_run_id, proposer, extraction_version, original_quote, status
        )
        SELECT DISTINCT ON (evidence.tenant_id, evidence.project_id, evidence.alias_id)
          evidence.tenant_id, evidence.project_id, evidence.source_id,
          evidence.source_version, evidence.passage_id, alias.alias_id,
          alias.alias_value, evidence.extraction_run_id, evidence.detector_name,
          evidence.detector_version, evidence.exact_quote, alias.status
        FROM alias_evidence AS evidence
        JOIN aliases AS alias
          ON (alias.tenant_id, alias.project_id, alias.alias_id) =
             (evidence.tenant_id, evidence.project_id, evidence.alias_id)
        WHERE evidence.source_type = 'extraction'
          AND evidence.source_id IS NOT NULL
          AND evidence.source_version IS NOT NULL
          AND evidence.passage_id IS NOT NULL
        ORDER BY evidence.tenant_id, evidence.project_id, evidence.alias_id, evidence.evidence_id;

        DROP TABLE alias_evidence;
        DROP TABLE alias_discovery_run;
        DROP INDEX IF EXISTS uq_aliases_one_approved_target;
        DROP TABLE aliases;
        ALTER TABLE llm_extraction_run DROP COLUMN alias_catalog_version;
        ALTER TABLE jev_response DROP COLUMN alias_catalog_version;
        """
    )
