"""Database factory and transaction boundary used by high-level stages."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine, URL


def create_engine_from_env() -> Engine:
    url = URL.create(
        "postgresql+psycopg",
        username=os.environ.get("POSTGRES_USER", "notebook"),
        password=os.environ.get("POSTGRES_PASSWORD", "notebook-local-dev"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "notebook"),
    )
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def connection_scope(engine: Engine) -> Iterator[Connection]:
    """Open the single transaction owned by a named high-level stage."""
    with engine.begin() as connection:
        yield connection
