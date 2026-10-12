from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def _source(connection, tenant: str, project: str) -> None:
    connection.execute(text("INSERT INTO source_document (tenant_id, project_id, source_id, source_version, title, source_sha256, expected_passage_count, status) VALUES (:tenant, :project, 'same-source', 'same-version', 'title', 'hash', 1, 'final')"), {"tenant": tenant, "project": project})
    connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES (:tenant, :project, 'same-source', 'same-version', 'same-passage', 'text', 'hash')"), {"tenant": tenant, "project": project})


def test_required_schema_tables_exist(db_engine) -> None:
    """Dropping a planned evidence/graph table must break this test."""
    expected = {
        "import_run", "import_error", "source_document", "source_head", "source_passage",
        "embedding_run", "passage_embedding", "index_snapshot", "index_member", "jev_response",
        "jev_selection", "llm_extraction_run", "validation_failure", "materialization_run",
        "passage_processing", "redaction_mention", "entity_mention", "assertion",
        "aliases", "alias_evidence", "alias_discovery_run",
        "kg_node", "kg_edge", "kg_node_source_ref", "kg_node_mention_ref", "kg_edge_source_ref",
        "current_index", "graph_scope", "graph_refresh_run",
    }
    with db_engine.connect() as connection:
        actual = set(connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars())
        states = connection.execute(
            text("SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_type.oid=enumtypid WHERE typname='processing_state' ORDER BY enumsortorder")
        ).scalars().all()
    assert expected <= actual
    assert states == ["not_run", "structural_only", "selected", "extracted", "failed"]


def test_processing_tracks_the_exact_validated_extraction_candidate(db_engine) -> None:
    """Materialization needs a durable pointer to the run validated by the extraction stage."""
    with db_engine.connect() as connection:
        columns = set(
            connection.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='passage_processing'"
                )
            ).scalars()
        )
    assert "validated_extraction_run_id" in columns


def test_composite_provenance_and_scoped_endpoints_are_database_enforced(db_engine) -> None:
    """Weakening a composite foreign key to local identifiers must break this test."""
    with db_engine.begin() as connection:
        _source(connection, "tenant", "one")
        _source(connection, "tenant", "two")
        connection.execute(text("INSERT INTO kg_node (tenant_id, project_id, node_id, node_type, attributes) VALUES ('tenant', 'one', 'n1', 'SourceDocument', '{}'), ('tenant', 'two', 'n1', 'SourceDocument', '{}')"))
        connection.execute(text("INSERT INTO assertion (tenant_id, project_id, assertion_id, source_id, source_version, passage_id, claim, chronology_state, review_state) VALUES ('tenant', 'one', 'a1', 'same-source', 'same-version', 'same-passage', 'claim', 'reported', 'unreviewed')"))
        connection.execute(text("INSERT INTO assertion (tenant_id, project_id, assertion_id, source_id, source_version, passage_id, claim, chronology_state, review_state) VALUES ('tenant', 'two', 'a2', 'same-source', 'same-version', 'same-passage', 'claim', 'reported', 'unreviewed')"))
    with pytest.raises(IntegrityError):
        with db_engine.begin() as connection:
            connection.execute(text("INSERT INTO assertion (tenant_id, project_id, assertion_id, source_id, source_version, passage_id, claim, chronology_state, review_state) VALUES ('tenant', 'two', 'bad', 'same-source', 'same-version', 'missing', 'claim', 'reported', 'unreviewed')"))
    with pytest.raises(IntegrityError):
        with db_engine.begin() as connection:
            connection.execute(text("INSERT INTO kg_edge (tenant_id, project_id, edge_id, edge_type, source_node_id, target_node_id, assertion_id, is_evidence) VALUES ('tenant', 'one', 'bad', 'supported_by', 'n1', 'n1', 'a2', true)"))


def test_every_passage_reference_and_current_pointer_rejects_cross_project_rows(db_engine) -> None:
    """Omitting scope from any provenance or current-pointer foreign key must break this test."""
    with db_engine.begin() as connection:
        _source(connection, "tenant", "one")
        _source(connection, "tenant", "two")
        connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES ('tenant', 'two', 'same-source', 'same-version', 'only-two', 'text', 'hash')"))
        connection.execute(text("INSERT INTO kg_node (tenant_id, project_id, node_id, node_type) VALUES ('tenant', 'one', 'n1', 'SourceDocument'), ('tenant', 'two', 'n1', 'SourceDocument'), ('tenant', 'two', 'n2', 'SourcePassage')"))
        connection.execute(text("INSERT INTO kg_edge (tenant_id, project_id, edge_id, edge_type, source_node_id, target_node_id) VALUES ('tenant', 'one', 'e1', 'contains_passage', 'n1', 'n1')"))
        connection.execute(text("INSERT INTO jev_response (response_id, tenant_id, project_id, source_id, source_version, passage_id, jev_response_version, model, questions, probabilities, input_sha256, profile_version, alias_catalog_version, status) VALUES ('00000000-0000-0000-0000-000000000101', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', 'jv', 'jev', '{}', '{}', 'hash', 'profile', 'catalog-v1', 'succeeded')"))
        connection.execute(text("INSERT INTO jev_selection (selection_id, response_id, tenant_id, project_id, source_id, source_version, passage_id, thresholds, selected_bundles, jev_selection_version) VALUES ('00000000-0000-0000-0000-000000000102', '00000000-0000-0000-0000-000000000101', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', '{}', '[]', 'sv')"))
        connection.execute(text("INSERT INTO llm_extraction_run (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id, extraction_version, model, sanitized_origin, credential_env, selected_bundles, input_sha256, profile_version, alias_catalog_version, status) VALUES ('00000000-0000-0000-0000-000000000103', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', 'ev', 'model', 'https://example.test/v1', 'LLM_API_KEY', '[]', 'hash', 'profile', 'catalog-v1', 'succeeded')"))

    invalid_statements = [
        "INSERT INTO assertion (tenant_id, project_id, assertion_id, source_id, source_version, passage_id, claim, chronology_state, review_state) VALUES ('tenant', 'three', 'a', 'same-source', 'same-version', 'same-passage', 'claim', 'reported', 'unreviewed')",
        "INSERT INTO entity_mention (tenant_id, project_id, source_id, source_version, passage_id, mention_id, entity_type, text) VALUES ('tenant', 'three', 'same-source', 'same-version', 'same-passage', 'm', 'Product', 'name')",
        "INSERT INTO alias_evidence (tenant_id, project_id, evidence_id, alias_id, source_type, passage_id, detector_name, detector_version, input_sha256) VALUES ('tenant', 'three', 'bad', 'a', 'extraction', 'same-passage', 'detector', 'v1', 'hash')",
        "INSERT INTO kg_node_source_ref (tenant_id, project_id, node_id, source_id, source_version, passage_id) VALUES ('tenant', 'one', 'n1', 'same-source', 'same-version', 'only-two')",
        "INSERT INTO kg_edge_source_ref (tenant_id, project_id, edge_id, source_id, source_version, passage_id) VALUES ('tenant', 'one', 'e1', 'same-source', 'same-version', 'only-two')",
        "INSERT INTO kg_edge (tenant_id, project_id, edge_id, edge_type, source_node_id, target_node_id) VALUES ('tenant', 'one', 'cross-endpoint', 'contains_passage', 'n1', 'n2')",
        "INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason, active_jev_selection_id) VALUES ('tenant', 'one', 'same-source', 'same-version', 'same-passage', 'selected', 'jev_bundle_selected', '00000000-0000-0000-0000-000000000102')",
        "INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason, active_extraction_run_id) VALUES ('tenant', 'one', 'same-source', 'same-version', 'same-passage', 'extracted', 'extracted_successfully', '00000000-0000-0000-0000-000000000103')",
        "INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason, validated_extraction_run_id) VALUES ('tenant', 'one', 'same-source', 'same-version', 'same-passage', 'selected', 'extraction_validated', '00000000-0000-0000-0000-000000000103')",
    ]
    for statement in invalid_statements:
        with pytest.raises(IntegrityError):
            with db_engine.begin() as connection:
                connection.execute(text(statement))


def test_index_membership_requires_same_scope_embedding_run(db_engine) -> None:
    """An index member bound to another project's embedding run must break this test."""
    with db_engine.begin() as connection:
        _source(connection, "tenant", "one")
        _source(connection, "tenant", "two")
        connection.execute(text("INSERT INTO embedding_run (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id, embedding_version, model, sanitized_origin, credential_env, dimensions, normalization, input_sha256, status) VALUES ('00000000-0000-0000-0000-000000000201', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', 'embedding-v', 'model', 'https://example.test/v1', 'EMBEDDING_LLM_API_KEY', 3072, 'provider_unit_length_v1', 'hash', 'succeeded')"))
        connection.execute(text("INSERT INTO index_snapshot (index_version, tenant_id, project_id, embedding_version) VALUES ('index-v', 'tenant', 'one', 'embedding-v')"))
    with pytest.raises(IntegrityError):
        with db_engine.begin() as connection:
            connection.execute(text("INSERT INTO index_member (index_version, tenant_id, project_id, source_id, source_version, passage_id, embedding_run_id) VALUES ('index-v', 'tenant', 'one', 'same-source', 'same-version', 'same-passage', '00000000-0000-0000-0000-000000000201')"))


def test_project_owned_version_keys_allow_same_local_id_in_different_projects(db_engine) -> None:
    """Using an unscoped primary key for an index version must break this test."""
    with db_engine.begin() as connection:
        connection.execute(text("INSERT INTO index_snapshot (index_version, tenant_id, project_id, embedding_version) VALUES ('same-index', 'tenant', 'one', 'embedding-v'), ('same-index', 'tenant', 'two', 'embedding-v')"))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM index_snapshot WHERE index_version='same-index'")).scalar_one() == 2


def test_scoped_alias_registry_enforces_schema_and_review_invariants(db_engine) -> None:
    """Removing registry checks, scoped provenance, or catalog versions must break this test."""
    with db_engine.connect() as connection:
        tables = set(connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")).scalars())
        registry_columns = {
            table: set(connection.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:table"
            ), {"table": table}).scalars())
            for table in ("aliases", "alias_evidence", "alias_discovery_run")
        }
        semantic_columns = {
            table: set(connection.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:table AND is_nullable='NO'"
            ), {"table": table}).scalars())
            for table in ("jev_response", "llm_extraction_run")
        }
        semantic_defaults = {
            table: connection.execute(text(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:table "
                "AND column_name='alias_catalog_version'"
            ), {"table": table}).scalar_one()
            for table in ("jev_response", "llm_extraction_run")
        }
        foreign_keys = set(connection.execute(text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE contype='f' AND conrelid IN "
            "('alias_evidence'::regclass, 'alias_discovery_run'::regclass)"
        )).scalars())
        evidence_checks = set(connection.execute(text(
            "SELECT conname FROM pg_constraint WHERE conrelid='alias_evidence'::regclass AND contype='c'"
        )).scalars())
    assert {"aliases", "alias_evidence", "alias_discovery_run"} <= tables
    assert registry_columns == {
        "aliases": {
            "tenant_id", "project_id", "alias_id", "alias_value", "normalized_alias",
            "entity_type", "canonical_key", "canonical_name", "status", "managed_by",
            "normalization_version", "confidence", "created_at", "updated_at", "decided_at",
            "decided_by", "decision_note",
        },
        "alias_evidence": {
            "tenant_id", "project_id", "evidence_id", "alias_id", "source_type", "source_id",
            "source_version", "passage_id", "extraction_run_id", "discovery_run_id", "exact_quote",
            "original_start", "original_end", "detector_name", "detector_version", "model",
            "prompt_hash", "schema_hash", "input_sha256", "created_at",
        },
        "alias_discovery_run": {
            "tenant_id", "project_id", "discovery_run_id", "method", "source_id", "source_version",
            "input_sha256", "detector_version", "model", "sanitized_origin", "credential_env",
            "prompt_hash", "schema_hash", "status", "usage", "attempts", "counts", "error_detail",
            "created_at", "finished_at",
        },
    }
    assert all("alias_catalog_version" in columns for columns in semantic_columns.values())
    assert all(default is None for default in semantic_defaults.values())
    assert {
        "FOREIGN KEY (tenant_id, project_id, alias_id) REFERENCES aliases(tenant_id, project_id, alias_id)",
        "FOREIGN KEY (tenant_id, project_id, source_id, source_version, passage_id) REFERENCES source_passage(tenant_id, project_id, source_id, source_version, passage_id)",
        "FOREIGN KEY (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id) REFERENCES llm_extraction_run(extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id)",
        "FOREIGN KEY (tenant_id, project_id, discovery_run_id) REFERENCES alias_discovery_run(tenant_id, project_id, discovery_run_id)",
        "FOREIGN KEY (tenant_id, project_id, source_id, source_version) REFERENCES source_document(tenant_id, project_id, source_id, source_version)",
    } <= foreign_keys
    assert {"ck_alias_evidence_source_identity", "ck_alias_evidence_extraction_scope"} <= evidence_checks

    with db_engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO aliases "
            "(tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,normalization_version) "
            "VALUES ('tenant','one','approved-a','Alias A','alias a','Product','product-a','Product A','approved','review','v1')"
        ))
        connection.execute(text(
            """INSERT INTO alias_evidence
               (tenant_id,project_id,evidence_id,alias_id,source_type,detector_name,detector_version,input_sha256)
               VALUES ('tenant','one','static-without-passage','approved-a','static_config','seed','v1','hash')"""
        ))
    invalid_statements = [
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version) VALUES ('tenant','one','approved-missing','Alias A','alias a','approved','review','v1')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version) VALUES ('tenant','one','blank-alias','  ','alias','pending','review','v1')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,status,managed_by,normalization_version) VALUES ('tenant','one','bad-type','Alias','alias','Application','pending','review','v1')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version,confidence) VALUES ('tenant','one','bad-confidence','Alias','alias','pending','review','v1',1.1)",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version) VALUES ('tenant','one','static-pending','Alias B','alias b','pending','static_config','v1')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version,decided_at,decided_by) VALUES ('tenant','one','static-rejected-missing-target','Alias B','alias b','rejected','static_config','v1',now(),'static_config')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,normalization_version,decided_at,decided_by) VALUES ('tenant','one','static-rejected-wrong-auditor','Alias B','alias b','Product','product-b','Product B','rejected','static_config','v1',now(),'reviewer')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version) VALUES ('tenant','one','rejected-undecided','Alias C','alias c','rejected','review','v1')",
        "INSERT INTO aliases (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,normalization_version) VALUES ('tenant','one','approved-conflict','Alias A again','alias a','Product','product-b','Product B','approved','review','v1')",
        "INSERT INTO alias_evidence (tenant_id,project_id,evidence_id,alias_id,source_type,passage_id,detector_name,detector_version,input_sha256) VALUES ('tenant','one','partial-passage','approved-a','deterministic_rule','same-passage','detector','v1','hash')",
        "INSERT INTO alias_evidence (tenant_id,project_id,evidence_id,alias_id,source_type,extraction_run_id,detector_name,detector_version,input_sha256) VALUES ('tenant','one','unscoped-extraction','approved-a','extraction','00000000-0000-0000-0000-000000000501','detector','v1','hash')",
        "INSERT INTO alias_evidence (tenant_id,project_id,evidence_id,alias_id,source_type,source_id,source_version,passage_id,detector_name,detector_version,input_sha256) VALUES ('tenant','two','cross-project-evidence','approved-a','extraction','same-source','same-version','same-passage','detector','v1','hash')",
    ]
    for statement in invalid_statements:
        with pytest.raises(IntegrityError):
            with db_engine.begin() as connection:
                connection.execute(text(statement))


def test_legacy_alias_candidates_migrate_without_invented_targets(migrated_0004_engine) -> None:
    """Migration must retain legacy observations as unresolved review proposals."""
    engine, upgrade_to_head = migrated_0004_engine
    upgrade_to_head()
    with engine.connect() as connection:
        aliases = connection.execute(text(
            "SELECT alias_id,alias_value,status,entity_type,canonical_key,canonical_name "
            "FROM aliases ORDER BY alias_value"
        )).all()
        evidence = connection.execute(text(
            "SELECT source_type,extraction_run_id FROM alias_evidence ORDER BY evidence_id"
        )).all()
        legacy_table = connection.execute(text(
            "SELECT to_regclass('public.alias_candidate')"
        )).scalar_one()
        catalog_columns = connection.execute(text(
            "SELECT table_name,column_default FROM information_schema.columns "
            "WHERE table_schema='public' "
            "AND table_name IN ('jev_response','llm_extraction_run') "
            "AND column_name='alias_catalog_version' ORDER BY table_name"
        )).all()
        migrated_catalog_version = connection.execute(text(
            "SELECT alias_catalog_version FROM llm_extraction_run "
            "WHERE extraction_run_id='00000000-0000-0000-0000-000000000501'"
        )).scalar_one()
    assert aliases == [
        ("legacy-cgv", "CGV-X", "pending", None, None, None),
        ("legacy-exa", "exa-cel", "pending", None, None, None),
    ]
    assert len(evidence) == 2
    assert all(row.source_type == "extraction" and row.extraction_run_id for row in evidence)
    assert legacy_table is None
    assert catalog_columns == [
        ("jev_response", None),
        ("llm_extraction_run", None),
    ]
    assert migrated_catalog_version == "legacy-unversioned"


def test_alias_registry_downgrade_recreates_representable_legacy_candidates(migrated_0004_engine) -> None:
    """Downgrade must retain one extraction observation for each scoped alias."""
    engine, upgrade_to_head = migrated_0004_engine
    upgrade_to_head()
    from alembic import command
    from alembic.config import Config

    command.downgrade(Config("alembic.ini"), "0004_current_semantic_state")
    with engine.connect() as connection:
        candidates = connection.execute(text(
            "SELECT alias_id,candidate,extraction_run_id,status "
            "FROM alias_candidate ORDER BY alias_id"
        )).all()
        aliases_table = connection.execute(text(
            "SELECT to_regclass('public.aliases')"
        )).scalar_one()
    assert candidates == [
        ("legacy-cgv", "CGV-X", UUID("00000000-0000-0000-0000-000000000501"), "pending"),
        ("legacy-exa", "exa-cel", UUID("00000000-0000-0000-0000-000000000501"), "pending"),
    ]
    assert aliases_table is None
