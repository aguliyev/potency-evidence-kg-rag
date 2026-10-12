from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from potency_kg.db import create_engine_from_env


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_models: opt-in tests that call configured external model providers",
    )


@pytest.fixture
def db_engine(monkeypatch: pytest.MonkeyPatch):
    database = f"kg_test_{uuid4().hex}"
    admin_url = f"postgresql+psycopg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/postgres"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database}"'))
    monkeypatch.setenv("POSTGRES_DB", database)
    command.upgrade(Config("alembic.ini"), "head")
    engine = create_engine_from_env()
    yield engine
    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :database AND pid <> pg_backend_pid()"), {"database": database})
        connection.execute(text(f'DROP DATABASE "{database}"'))
        assert connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname=:database"), {"database": database}
        ).scalar() is None
    admin.dispose()


@pytest.fixture
def migrated_0004_engine(monkeypatch: pytest.MonkeyPatch):
    """A disposable legacy database plus an explicit upgrade-to-head action."""
    database = f"kg_migration_test_{uuid4().hex}"
    admin_url = f"postgresql+psycopg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/postgres"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database}"'))
    monkeypatch.setenv("POSTGRES_DB", database)
    config = Config("alembic.ini")
    command.upgrade(config, "0004_current_semantic_state")
    engine = create_engine_from_env()
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO source_document "
            "(tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) "
            "VALUES ('legacy-tenant','legacy-project','legacy-source','v1','legacy title','legacy-sha',1,'final')"
        ))
        connection.execute(text(
            "INSERT INTO source_passage "
            "(tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) "
            "VALUES ('legacy-tenant','legacy-project','legacy-source','v1','legacy-passage','legacy text','passage-sha')"
        ))
        connection.execute(text(
            "INSERT INTO llm_extraction_run "
            "(extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,extraction_version,model,sanitized_origin,credential_env,selected_bundles,input_sha256,profile_version,status) "
            "VALUES ('00000000-0000-0000-0000-000000000501','legacy-tenant','legacy-project','legacy-source','v1','legacy-passage','legacy-extraction','model','https://example.test/v1','LLM_API_KEY','[]','input-sha','profile','succeeded')"
        ))
        connection.execute(text(
            "INSERT INTO alias_candidate "
            "(tenant_id,project_id,source_id,source_version,passage_id,alias_id,candidate,extraction_run_id,proposer,extraction_version,original_quote) "
            "VALUES "
            "('legacy-tenant','legacy-project','legacy-source','v1','legacy-passage','legacy-cgv','CGV-X','00000000-0000-0000-0000-000000000501','legacy-detector','legacy-extraction','CGV-X'),"
            "('legacy-tenant','legacy-project','legacy-source','v1','legacy-passage','legacy-exa','exa-cel','00000000-0000-0000-0000-000000000501','legacy-detector','legacy-extraction','exa-cel')"
        ))

    def upgrade_to_head() -> None:
        command.upgrade(config, "head")

    yield engine, upgrade_to_head
    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :database AND pid <> pg_backend_pid()"), {"database": database})
        connection.execute(text(f'DROP DATABASE "{database}"'))
    admin.dispose()
