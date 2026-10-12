from __future__ import annotations

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
        "passage_processing", "redaction_mention", "entity_mention", "alias_candidate", "assertion",
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
        connection.execute(text("INSERT INTO jev_response (response_id, tenant_id, project_id, source_id, source_version, passage_id, jev_response_version, model, questions, probabilities, input_sha256, profile_version, status) VALUES ('00000000-0000-0000-0000-000000000101', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', 'jv', 'jev', '{}', '{}', 'hash', 'profile', 'succeeded')"))
        connection.execute(text("INSERT INTO jev_selection (selection_id, response_id, tenant_id, project_id, source_id, source_version, passage_id, thresholds, selected_bundles, jev_selection_version) VALUES ('00000000-0000-0000-0000-000000000102', '00000000-0000-0000-0000-000000000101', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', '{}', '[]', 'sv')"))
        connection.execute(text("INSERT INTO llm_extraction_run (extraction_run_id, tenant_id, project_id, source_id, source_version, passage_id, extraction_version, model, sanitized_origin, credential_env, selected_bundles, input_sha256, profile_version, status) VALUES ('00000000-0000-0000-0000-000000000103', 'tenant', 'two', 'same-source', 'same-version', 'same-passage', 'ev', 'model', 'https://example.test/v1', 'LLM_API_KEY', '[]', 'hash', 'profile', 'succeeded')"))

    invalid_statements = [
        "INSERT INTO assertion (tenant_id, project_id, assertion_id, source_id, source_version, passage_id, claim, chronology_state, review_state) VALUES ('tenant', 'three', 'a', 'same-source', 'same-version', 'same-passage', 'claim', 'reported', 'unreviewed')",
        "INSERT INTO entity_mention (tenant_id, project_id, source_id, source_version, passage_id, mention_id, entity_type, text) VALUES ('tenant', 'three', 'same-source', 'same-version', 'same-passage', 'm', 'Product', 'name')",
        "INSERT INTO alias_candidate (tenant_id, project_id, source_id, source_version, passage_id, alias_id, candidate) VALUES ('tenant', 'three', 'same-source', 'same-version', 'same-passage', 'a', 'name')",
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
