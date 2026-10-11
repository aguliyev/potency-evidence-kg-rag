from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from potency_kg.db import connection_scope


def test_connection_scope_commits_real_database_work(db_engine) -> None:
    """Removing the high-level transaction boundary must break this test."""
    with connection_scope(db_engine) as connection:
        connection.execute(text("INSERT INTO import_run (run_id, tenant_id, project_id, descriptor_sha256, status) VALUES ('00000000-0000-0000-0000-000000000001', 'tenant', 'project', 'hash', 'running')"))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM import_run")).scalar_one() == 1


def test_processing_reason_is_required_and_active_pointer_is_nullable(db_engine) -> None:
    """Allowing blank reasons or requiring an active run must break this test."""
    with db_engine.begin() as connection:
        connection.execute(text("INSERT INTO source_document (tenant_id, project_id, source_id, source_version, title, source_sha256, expected_passage_count, status) VALUES ('t', 'p', 's', 'v', 'title', 'hash', 1, 'final')"))
        connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES ('t', 'p', 's', 'v', 'x', 'text', 'hash')"))
        connection.execute(text("INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason) VALUES ('t', 'p', 's', 'v', 'x', 'not_run', 'inactive_source_version')"))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT active_extraction_run_id FROM passage_processing")).scalar_one() is None
    with pytest.raises(IntegrityError):
        with db_engine.begin() as connection:
            connection.execute(text("UPDATE passage_processing SET reason='   '"))


def test_processing_has_no_partial_active_pointer_index(db_engine) -> None:
    """Reintroducing a contradictory partial active-pointer index must break this test."""
    with db_engine.connect() as connection:
        indexes = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='passage_processing'")
        ).scalars().all()
    assert not any(" WHERE " in definition.upper() for definition in indexes)
