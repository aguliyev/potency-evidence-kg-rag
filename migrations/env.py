from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import URL

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def database_url() -> URL:
    return URL.create(
        drivername="postgresql+psycopg",
        username=os.environ.get("POSTGRES_USER", "notebook"),
        password=os.environ.get("POSTGRES_PASSWORD", "notebook-local-dev"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "notebook"),
    )


def run_migrations_offline() -> None:
    url = database_url()
    context.configure(
        url=url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = database_url().render_as_string(hide_password=False)
    # Alembic's ConfigParser treats percent signs as interpolation markers.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
