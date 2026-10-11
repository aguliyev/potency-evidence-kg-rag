# Local Notebook and Graph Database

This optional local stack runs JupyterLab with PostgreSQL 17, pgvector, and Apache AGE. Docker Compose builds the database and Python images, starts PostgreSQL, applies pending Alembic migrations, and then starts JupyterLab. The database health check gates migrations, and successful migrations gate the notebook service.

Dockerfiles and the Compose definition are under `docker/`.

## Setup

Docker and its Compose plugin must already be installed. Run:

```bash
bin/install
bin/start
```

`bin/install` checks Docker and Compose, creates `etc/secrets/.env` from `etc/secrets/.env.example` if it does not exist, and builds the images. `bin/start` starts the stack in the background. Open JupyterLab at <http://localhost:8888> and use the `JUPYTER_TOKEN` value in `etc/secrets/.env` (the local default is `local-development-token`).

You can also start Compose directly after `bin/install`:

```bash
docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env up --build --detach
```

To stop the containers while keeping database data, run `bin/stop`. Run `bin/migrate` to apply migrations on demand. The named `deffai_postgres_data` volume is preserved by `bin/stop`.

## Environment configuration

`etc/.env` contains shared local configuration. `etc/secrets/.env` contains credentials and tokens, and is ignored by Git. The tracked `etc/secrets/.env.example` supplies development-only values when `bin/install` creates the private file. Replace those values for your local use and do not commit the private file.

Every Compose service loads both files through `env_file`. The lifecycle scripts also pass both files to Compose so values used for port bindings and the Jupyter command are available during Compose interpolation.

The default host connection is `postgresql+psycopg://notebook:notebook-local-dev@localhost:5432/notebook`. Notebook code connects to host `db`, port `5432`, using `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`. Change the host ports in `etc/.env`, and database credentials or the Jupyter token in `etc/secrets/.env`.

## Stack details

- PostgreSQL 17 with pinned pgvector 0.8.7 and Apache AGE 1.7.0.
- Python 3.12 with pinned notebook and migration dependencies.
- Alembic migrations enable the `vector` and `age` extensions.
- The `db` health check gates the one-shot `migrate` service; `notebook` waits until migration succeeds.
- The database uses the named `deffai_postgres_data` volume so its data persists across container stops.
- No application tables, named AGE graphs, ingestion, or production deployment configuration are included.

Put importable Python modules in `lib/` and notebooks in `notebooks/`. Both directories are mounted into `/workspace` in the notebook container, and `lib/` is on `PYTHONPATH`, so host edits are available to notebook kernels immediately.
