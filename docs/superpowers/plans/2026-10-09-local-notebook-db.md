# Local Notebook and Graph Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible local Docker Compose stack with PostgreSQL 17, pgvector, Apache AGE, Alembic migrations, and JupyterLab.

**Architecture:** Compose builds a custom PostgreSQL image with pinned pgvector and AGE releases, and a Python image with pinned notebook and migration dependencies. A health-checked database gates a one-shot Alembic service, whose successful completion gates JupyterLab. Host scripts validate prerequisites, prepare `.env`, and drive Compose lifecycle commands.

**Tech Stack:** Docker Compose, PostgreSQL 17, pgvector 0.8.7, Apache AGE 1.7.0 for PG17, Python 3.12, SQLAlchemy, Alembic, psycopg 3, JupyterLab.

**Spec:** `docs/superpowers/specs/2026-10-09-local-notebook-db-design.md`

## Global Constraints

- PostgreSQL 17 is the initial compatibility target.
- Pin both PostgreSQL extension versions.
- `docker compose -f docker/compose.yaml up` and `bin/start` must run migrations before JupyterLab.
- `bin/install` must not install host packages or Docker.
- `bin/stop` must preserve the named database volume.
- Notebook code and notebooks remain editable on the host through `lib/` and `notebooks/` mounts.
- Do not add application tables, named AGE graphs, ingestion, or production deployment configuration.

---

### Task 1: Build images and define Compose lifecycle

**Files:**
- Create: `docker/Dockerfile.db`
- Create: `docker/Dockerfile.notebook`
- Create: `requirements.txt`
- Create: `docker/compose.yaml`
- Create: `.env.example`
- Create: `.dockerignore`
- Create: `.gitignore` (only if absent; preserve existing ignore rules if present)

**Interfaces:**
- Compose services: `db`, `migrate`, `notebook`.
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` are passed to migration and notebook containers; Alembic builds a properly escaped SQLAlchemy URL from them.
- Host ports, credentials, database name, and Jupyter token come from Compose environment interpolation.

- [ ] Add a PostgreSQL 17 image with pinned pgvector 0.8.7 and Apache AGE 1.7.0 installed.
- [ ] Add a Python 3.12 notebook image with pinned Alembic, SQLAlchemy, psycopg, and JupyterLab dependencies.
- [ ] Define database health checking, migration completion dependency, localhost port bindings, and the persistent PostgreSQL volume.
- [ ] Mount `lib/` and `notebooks/` into the notebook workspace and add the `lib/` mount to `PYTHONPATH`.
- [ ] Add local development defaults to `.env.example`, excluding real credentials.

### Task 2: Add versioned extension migration

**Files:**
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/0001_enable_extensions.py`

**Interfaces:**
- Alembic reads `DATABASE_URL` from its environment.
- Revision `0001_enable_extensions` enables `vector` and `age`; downgrade removes them in reverse order.

- [ ] Configure Alembic for a migration directory under `migrations/` and a runtime-provided connection URL.
- [ ] Configure online and offline migration execution with SQLAlchemy.
- [ ] Create the initial revision using explicit SQL for the PostgreSQL extensions.

### Task 3: Add local lifecycle scripts and setup documentation

**Files:**
- Create: `bin/install`
- Create: `bin/start`
- Create: `bin/migrate`
- Create: `bin/stop`
- Create: `lib/.gitkeep`
- Create: `notebooks/.gitkeep`
- Modify: `README.md`

**Interfaces:**
- `bin/install`: prerequisite checks, copy `.env.example` to `.env` only if absent, then build images.
- `bin/start`: build as needed and start the complete stack in the background.
- `bin/migrate`: run the one-shot migration service and return Compose's status.
- `bin/stop`: stop and remove Compose containers without removing volumes.

- [ ] Implement executable shell scripts with clear prerequisite and failure messages.
- [ ] Document install, start, stop, host connection details, notebook connection details, and configuration overrides.
- [ ] Keep README assignment instructions intact while adding a focused local stack section.

### Task 4: Review the resulting configuration

**Files:**
- Review: all files created or modified above.

- [ ] Inspect diffs for requirements coverage, accidental credential inclusion, and preservation of existing user changes.
- [ ] Check shell script syntax and inspect Compose/Alembic configuration statically; do not run test suites or start services.
