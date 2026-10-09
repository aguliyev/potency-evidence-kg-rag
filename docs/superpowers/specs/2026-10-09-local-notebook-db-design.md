# Local Notebook and Graph Database Design

## Goal

Provide a reproducible local development stack with JupyterLab and PostgreSQL that supports both vector search through pgvector and graph storage through Apache AGE. Keep notebook code and notebooks editable on the host, and apply schema migrations as part of normal stack startup.

## Design

Docker Compose will define three services: `db`, `migrate`, and `notebook`. `db` uses the custom database image and a named volume. `migrate` is a one-shot container using the notebook Python image; it waits for the database health check and runs `alembic upgrade head`. `notebook` waits for the migration service to complete successfully, then serves JupyterLab. Thus `docker compose up` and `bin/start` bring up the database, apply pending migrations, and then start JupyterLab.

There will be exactly two Dockerfiles: one builds PostgreSQL with Apache AGE and pgvector, and one builds the Python/Jupyter environment with the project dependencies and Alembic. PostgreSQL 17 with an AGE release published for PostgreSQL 17 and a pgvector release supporting PostgreSQL 17 is the initial compatibility target; both extension versions will be pinned. Database schema and extension enablement belong in versioned Alembic migrations, using SQLAlchemy operations where suitable and explicit SQL for PostgreSQL extensions and AGE graph commands. The initial migration enables both extensions; application tables and named AGE graphs can be added in later revisions.

The notebook service mounts host `lib/` and `notebooks/` into its workspace. The `lib/` directory is importable from notebook kernels, and edits on the host are immediately reflected in the running container. Database and notebook ports and database credentials are configurable through Compose environment variables with local development defaults.

## Commands and lifecycle

- `bin/install` verifies that Docker and the Compose plugin are available, prepares local configuration if needed, and builds the images. It does not install host packages or Docker.
- `bin/start` builds as needed and runs the full Compose stack in the background. Compose health/completion dependencies ensure migrations finish before JupyterLab starts.
- `bin/migrate` runs the one-shot Alembic service on demand and returns its exit status.
- `bin/stop` stops/removes Compose containers while preserving the named database volume.

The Alembic environment and revision files live under `migrations/`. `alembic.ini`, dependency declarations, a Compose file, a local environment template, and concise setup documentation complete the interface.

## Alternatives considered

1. **Alembic in a one-shot Compose service (recommended):** consistent with the Python notebook environment, provides ordered/versioned migrations, and can mix SQLAlchemy operations with raw SQL. It keeps database startup focused on PostgreSQL while making migrations an explicit dependency.
2. **Versioned SQL scripts:** fewer Python abstractions and natural for extension-specific SQL, but requires choosing and maintaining a separate migration runner/state mechanism.
3. **Run migrations inside the PostgreSQL entrypoint:** makes the database image responsible for application schema and Python tooling, and couples database restarts to application migration logic. This is not selected.

## Scope and acceptance

The work adds the Compose file, two Dockerfiles, `bin/install`, `bin/start`, `bin/stop`, `bin/migrate`, an Alembic environment under `migrations/`, and the `lib/` and `notebooks/` mount points. Setup documentation will state how to connect from the host and from notebooks and how to override local defaults. No production deployment, host Docker installation, data ingestion, or application-specific schema is included.
