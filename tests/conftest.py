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
