# Potency evidence knowledge graph

## Project summary

- The pipeline imports the supplied regulatory and precedent sources into a
  tenant/project-scoped PostgreSQL store and preserves source versions and passage
  provenance.
- Jev is the **System One model**. It classifies passages with a fixed seven-question
  Noul bundle; saved probabilities and thresholds determine which extraction
  bundles run.
- Structured extraction creates evidence-backed entities and relationships.
  Product and Assay mentions resolve through the approved, scoped alias catalog.
- The system builds a validated graph and supports vector and graph retrieval with
  answers that cite passages. Traces preserve the exact evidence sent to the model.
- [`HybridRetrievalConfig`](../lib/potency_kg/retrieval_config.py) centralizes
  retrieval settings—such as hit counts, graph traversal, scoring, and evidence
  budgets—so tuning it changes how evidence is selected.
- Alias coverage is a key opportunity to improve recall. Broader automatic alias
  discovery paired with human review and approval would help keep canonical entities
  accurate as the corpus grows.
- NER models could improve the detection of Products, Assays, and other domain
  entities during extraction.
- The best next step is a stronger evaluation loop: measure retrieval and answer
  quality, then use the same cases to compare changes to configuration, aliases,
  prompts, and NER models.

## Implementation layout
>
> | Directory | Contents |
> | --- | --- |
> | [`bin/`](bin/) | Supported local setup, service, migration, and Docker-backed runner commands. |
> | [`docker/`](docker/) | Compose services and container build configuration for PostgreSQL, migrations, Jupyter, and the runner. |
> | [`docs/`](docs/) | Method notes, retrieval comparison, delivery map, etc. |
> | [`etc/`](etc/) | Non-secret environment defaults and private-environment templates. |
> | [`lib/`](lib/) | Python implementation: ingestion, graph materialization, retrieval, answering, evaluation, and delivery validation. |
> | [`migrations/`](migrations/) | Alembic database schema migrations. |
> | [`notebooks/`](notebooks/) | Build, inspection, and read-only retrieval/answering example notebooks. |
> | [`results/`](results/) | Generated graph exports, answer/trace artifacts, evaluation, architecture, and delivery-audit outputs. |
> | [`tests/`](tests/) | Unit, integration, and real-model-gated regression coverage. |

--------------------------------
--------------------------

## Data

| Path | Purpose |
|---|---|
| `corpus/records.jsonl` | Mechanical convenience extraction: 791 normalized passages. |
| `corpus/manifest.json` | Frozen snapshot, source inventory, hashes, and versions. |
| `corpus/raw/regulatory/` | Official FDA guidance/policy PDFs and dated eCFR XML. |
| `corpus/raw/precedent/` | Official ZYNTEGLO and LYFGENIA SBRAs. |
| `corpus/raw/casgevy-125785/` | Supplied public TDT approval materials. |
| `corpus/raw/casgevy-125787/` | SCD approval letter and SBRA. |
| `questions.json` | Six required questions; P3 requires vector and graph modes. |
| `schema/graph-core.json` | Adjustable reference schema; document mappings to required behavior. |
| `schema/output-contract.json` | Minimum artifact and field expectations. |
| `configs/casgevy-potency.json` | Corpus scope, case identity, and relative resource paths. |
| `configs/model-access.template.json` | Team-completed access configuration; contains no credentials. |
| `MODEL_ACCESS.md` | Pre-start checklist, two access smoke calls and outage handling. |
| `fixtures/` | Project-isolation collision inputs plus correction/conflict scenarios. |

### Objective

- Process all records in `corpus/records.jsonl`; do not pre-filter to documents known to answer the questions.
- Treat files under `corpus/raw/` as the source of truth. The JSONL is a mechanical extraction provided for convenience.
- Use the raw files to inspect provenance and page context; PDF/XML parsing is not required.
- Use only the supplied corpus for factual answers. URLs are provenance references.
- `(b)(4)` denotes public redaction and must not be filled by inference.
- The CASGEVY CMC review memorandum, approval letter, and SBRA are separate documents.
- The configured application is TDT/STN 125785, but the shared exa-cel potency history spans TDT and SCD/STN 125787. Preserve application and indication scope.

The record identity is:

```text
(tenant_id, project_id, source_id, source_version, passage_id)
```

Do not treat `passage_id` alone as globally unique. The fixture inputs intentionally collide on local source and passage identifiers across project scopes.


## Local architecture

The Docker Compose stack contains:

- **`db`** — PostgreSQL 17 with pgvector and Apache AGE.
- **`migrate`** — a one-shot Alembic migration service.
- **`notebook`** — Python 3.12 and JupyterLab with the repository mounted at
  `/workspace`.
- **`runner`** — the supported command and test environment used by `bin/runner`.

The database is stored in the named `potency_kg_rag_postgres_data` Docker volume. Normal
stops preserve it. The repository's `configs/`, `corpus/`, `fixtures/`, and
`schema/` directories are protected inputs and are mounted read-only in containers.

## Prerequisites

Install the following before setup:

- Git.
- Docker Desktop, or Docker Engine with the Docker Compose plugin.
- Enough disk space to build the PostgreSQL and Python images and retain the local
  database volume.

Confirm Docker is running:

```bash
docker version
docker compose version
```

Run all remaining commands from the repository root.

## First-time setup

Build the local images and create the private environment file:

```bash
bin/install
```

`bin/install` performs three actions:

1. Verifies that Docker and the Compose plugin are available.
2. Copies `etc/secrets/.env.example` to `etc/secrets/.env` when the private file
   does not already exist.
3. Builds the database and Python images.

Review `etc/secrets/.env` after the first installation. It is ignored by Git and
must not be committed.

The default local-only values are sufficient for the structural workflow:

```dotenv
POSTGRES_PASSWORD=notebook-local-dev
JUPYTER_TOKEN=local-development-token
LLM_API_KEY=
EMBEDDING_LLM_API_KEY=
TYPESAFE_API_KEY=
```

Shared, non-secret settings live in `etc/.env`. This file controls the PostgreSQL
database name and ports, the Jupyter host port, and model endpoint/model names.

## Start the stack

Start PostgreSQL, apply migrations, and launch JupyterLab:

```bash
bin/start
```

The command builds changed images, starts the services in the background, waits for
PostgreSQL to become healthy, applies pending migrations, and starts Jupyter only
after migration succeeds.

Check service state with:

```bash
docker compose -f docker/compose.yaml \
  --env-file etc/.env --env-file etc/secrets/.env ps
```

Apply migrations again without restarting the stack when needed:

```bash
bin/migrate
```

## Stop the stack

Stop and remove the containers while retaining database data:

```bash
bin/stop
```

The `potency_kg_rag_postgres_data` volume is intentionally preserved, so the next
`bin/start` continues from the same database. Removing that volume is destructive
and is not part of the normal stop procedure.

## Use JupyterLab

After `bin/start`, open:

<http://localhost:8888/?token=q>

Enter the `JUPYTER_TOKEN` from `etc/secrets/.env`. The development default is
`local-development-token`. If `JUPYTER_HOST_PORT` is changed in `etc/.env`, use the
new port instead of `8888`.

The repository is mounted at `/workspace`, and `lib/` is already on `PYTHONPATH`.
Edits made on the host are therefore visible to notebook kernels without copying
files into the container.

Use the notebooks in this order:

1. **`notebooks/01_build_knowledge_graph.ipynb`** — runs the pipeline as explicit,
   independently rerunnable stages. Each write stage is followed by a read-only
   inspection cell showing scoped counts, versions, statuses, and sanitized failure
   totals.
2. **`notebooks/02_inspect_knowledge_graph.ipynb`** — inspects the active graph and
   published artifacts. Its optional retrieval smoke test remains disabled unless
   `KG_ENABLE_RETRIEVAL_SMOKE=true` is set explicitly.

The notebooks call the same public stage functions as the CLI. They intentionally
contain no separate pipeline implementation and do not automatically approve alias
candidates.

## Run from the command line

`bin/runner` executes commands inside the same image and environment used by
Jupyter:

```bash
bin/runner python -m potency_kg.cli run-kg \
  --config configs/casgevy-potency.json \
  --output results/kg
```

With no real-model gate, this runs the credential-free structural path through
import, alias preparation, structural graph refresh, inspection, and export. Model
stages are reported as `not_run`.

To run an individual stage, use the corresponding command:

```bash
bin/runner python -m potency_kg.cli preflight \
  --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli inspect-kg \
  --config configs/casgevy-potency.json
```

The detailed command sequence is documented in
[Knowledge graph generation and operation](kg-generation.md).

## Pipeline stages in detail

### 1. Preflight

Preflight validates the project descriptor, corpus manifest, normalized records,
raw-file checksums, source counts, passage counts, replacement declarations, and
path containment. It reads no model settings and writes no database state.

This stage establishes that the supplied inputs are complete and internally
consistent before any import begins.

```bash
bin/runner python -m potency_kg.cli preflight \
  --config configs/casgevy-potency.json
```

### 2. Structural import

Import writes immutable source documents and passages, advances explicit source
heads, and creates document/passage structural graph records. Each source is
transactional: an invalid or failed source does not leave partially active rows.

An unchanged import is idempotent. A newer source version replaces the current head
only when the descriptor declares the replacement; prior versions remain available
as history.

```bash
bin/runner python -m potency_kg.cli import \
  --config configs/casgevy-potency.json
```

### 3. Static alias loading

The seed loader reconciles the source-controlled alias configuration into the
tenant/project-scoped alias registry. Static mappings are trusted application data,
load as approved, and carry immutable configuration evidence.

Removing or changing a static mapping retires the old row without deleting its
history. Human review commands cannot modify static rows.

```bash
bin/runner python -m potency_kg.cli load-alias-seeds \
  --config configs/casgevy-potency.json
```

### 4. Alias discovery and review

Deterministic discovery scans active passages for supported alias patterns without
constructing a model client. Repeated observations add evidence to the same scoped
candidate. New candidates remain pending and never affect entity identity until a
reviewer explicitly approves them.

```bash
bin/runner python -m potency_kg.cli discover-aliases-deterministic \
  --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli list-aliases \
  --config configs/casgevy-potency.json --status pending
bin/runner python -m potency_kg.cli list-alias-evidence \
  --config configs/casgevy-potency.json --alias-id <alias-id>
```

Approval requires a reviewer and a complete Product or Assay target. Rejection
preserves the candidate and all evidence. A later reconsideration is audited.

```bash
bin/runner python -m potency_kg.cli review-alias \
  --config configs/casgevy-potency.json \
  --alias-id <alias-id> --decision approved --reviewer <operator> \
  --entity-type Product --canonical-key casgevy --canonical-name CASGEVY
```

The optional LLM discovery command is a separate, explicit operation. It is
disabled by default and does not approve its proposals automatically.

```bash
bin/runner python -m potency_kg.cli discover-aliases-llm \
  --config configs/casgevy-potency.json
```

#### Why alias curation is the main recall and quality lever

Alias coverage is the single largest factor in the recall and overall quality of
the knowledge graph. Product and Assay mentions are resolved exclusively through
exact normalized matches against the approved scoped alias catalog, and the
extraction input is built from that same catalog snapshot. Every synonym missing
from the catalog — `CTX001`, `exa-cel`, `VX-290` for CASGEVY, for example —
splits one real-world entity into several graph nodes or misses its mentions
entirely. Pending, rejected, cross-type, or merely similar aliases never merge
entities, so unreviewed candidates do not help until a human approves them.

Candidates come from two generators:

- **Deterministic rules** (`document-alias-patterns`,
  `document-alias-patterns-v1` in `lib/potency_kg/alias_discovery.py`) scan each
  active passage for explicit synonym patterns: parenthetical forms in both
  directions (`exagamglogene autotemcel (exa-cel)`, `lovotibeglogene autotemcel
  (lovo-cel)`) and explicit phrases (`also known as`, `abbreviated as`,
  `formerly called`). Pairs pass an acronym check — the
  short form must be an ordered letter-subsequence of the long form — before
  they are stored as `deterministic_rule` evidence.
- **Document-wide LLM discovery** (`structured-document-alias-discovery`,
  `structured-document-alias-discovery-v1`) sends windows of a document's
  passages to a structured provider and grounds every proposal back to an exact
  quote in its original passage before anything is written. These proposals are
  stored as `llm_mined` evidence with per-proposal confidence. They carry no
  `canonical_key`, so approving them always requires a human to supply the full
  Product or Assay target.

Human curation is therefore mandatory, not optional: a reviewer inspects each
candidate's evidence quotes (`notebooks/03_review_aliases.ipynb`, or
`list-alias-evidence` / `review-alias` on the CLI) and approves it into a
canonical target such as `Product/casgevy/CASGEVY`. Approval is what tells the
graph that `CTX001`, `exa-cel`, and `CASGEVY` are the same entity, and each
decision is audited with reviewer identity. Only one row per
`(entity_type, normalized_alias)` can be approved, so competing proposals for
the same synonym must be resolved to a single winner.

One way to reduce the human workload would be semantic pre-checking of
candidates — comparing alias and proposal embeddings, or running NER over the
supporting quotes to confirm both forms denote the same entity — but that is
out of scope for this project stage. Review stays explicit and human-driven.

### 5. Embeddings and index snapshot

The embedding stage generates or reuses 3072-dimensional vectors for active
passages and records an immutable index snapshot. Vector reuse depends on normalized
content and provider identity; index identity also includes source membership.

```bash
bin/runner python -m potency_kg.cli embed \
  --config configs/casgevy-potency.json
```

### 6. Jev classification and selection

Jev sends each passage through the fixed seven-question Noul classification bundle.
All probabilities are persisted before inclusive thresholds select extraction
bundles. A passage with no selected bundle makes no extraction request, and a failed
classification never falls back to unrestricted extraction.

```bash
bin/runner python -m potency_kg.cli classify-jev \
  --config configs/casgevy-potency.json
```

### 7. Structured extraction

Extraction builds a response schema from only the selected bundles plus universal
assertion and provenance fields. It validates local identifiers and requires exact
supporting quotes from the source passage. Invalid or ungrounded output cannot
become the active extraction.

The extraction snapshot records the exact approved alias catalog used to construct
the model input.

```bash
bin/runner python -m potency_kg.cli extract \
  --config configs/casgevy-potency.json
```

### 8. Resolution and materialization

Materialization converts a validated extraction into canonical entities,
assertions, relationships, and source references. Product and Assay mentions use
exact normalized matches from the approved scoped alias catalog. Pending, rejected,
cross-type, or merely similar aliases do not merge entities.

The stage verifies that the extraction's catalog snapshot still matches the live
catalog and performs semantic writes atomically. If an alias decision changes the
catalog, rerun the semantic stages before materializing or publishing a replacement
graph.

```bash
bin/runner python -m potency_kg.cli materialize \
  --config configs/casgevy-potency.json
```

### 9. Graph refresh

Refresh projects relational state into a shadow Apache AGE graph, validates its
counts, labels, endpoints, provenance, and text-leakage constraints, and then
activates it atomically. Failure preserves the prior active graph.

Use structural mode when model stages have not run, or semantic mode after
successful extraction and materialization:

```bash
bin/runner python -m potency_kg.cli refresh-graph \
  --config configs/casgevy-potency.json --mode structural
```

### 10. Export

Export writes a versioned directory beneath `results/kg/history/`, validates the
manifest and artifact hashes, and then atomically updates the relative
`results/kg/current` symlink. Existing history is immutable, and a failed export
does not replace the last-known-good current version.

```bash
bin/runner python -m potency_kg.cli export-kg \
  --config configs/casgevy-potency.json --output results/kg
```

The export includes its manifest, import report, extraction coverage, graph
validation, graph nodes, and graph edges.

### 11. Inspection

Inspection is read-only. It reports the scoped catalog version, import and
processing counts, stage statuses and failures, current embedding/index/graph
bindings, and the validated current export version. Provider exception details and
credential values are not exposed.

```bash
bin/runner python -m potency_kg.cli inspect-kg \
  --config configs/casgevy-potency.json
```

## Enable real model stages

Structural operation requires no model credentials. To enable the semantic stages,
set private values in `etc/secrets/.env` and explicitly enable the gate:

```dotenv
KG_RUN_REAL_MODELS=true
LLM_API_KEY=<generation-key>
EMBEDDING_LLM_API_KEY=<embedding-key>
TYPESAFE_API_KEY=<jev-key>
```

The endpoint and model names are read from `etc/.env`. A complete semantic
`run-kg` requires all three setting groups to validate. If the gate is absent or a
group is incomplete, the composite command remains structural and reports the model
stages as `not_run`; it does not silently substitute a provider or credential.

Changing the approved alias set does not rewrite an active graph automatically.
After approving a pending/rejected alias or rejecting an approved alias, explicitly
rerun Jev, extraction, materialization, semantic refresh, and export.

## Run fixtures and tests

Run the supplied behavior fixtures with offline deterministic adapters:

```bash
bin/runner python -m potency_kg.cli run-kg-fixtures \
  --output results/kg/fixtures
```

Run the offline test suite with:

```bash
bin/runner pytest -q -m "not real_models"
```

Real-provider tests are opt-in and report `not_run` when the explicit gate is not
enabled:

```bash
bin/runner pytest -q -rs tests/real/test_real_model_stages.py -m real_models
```

## Troubleshooting

### Jupyter does not start

Check Compose service state and migration logs:

```bash
docker compose -f docker/compose.yaml \
  --env-file etc/.env --env-file etc/secrets/.env ps
docker compose -f docker/compose.yaml \
  --env-file etc/.env --env-file etc/secrets/.env logs migrate notebook
```

Jupyter waits for a healthy database and a successful migration, so a database or
migration failure prevents the notebook service from starting.

### A port is already in use

Change `POSTGRES_HOST_PORT` or `JUPYTER_HOST_PORT` in `etc/.env`, then restart the
stack with `bin/stop` followed by `bin/start`.

### A command cannot connect to PostgreSQL

Run `bin/start` first and verify that `potency_kg_rag-db` is healthy. Commands should be run
through `bin/runner`, which uses the container hostname and environment expected by
the project.

### A semantic stage is `not_run`

Confirm that `KG_RUN_REAL_MODELS=true` is present in `etc/secrets/.env` and that all
three credential groups are complete. Structural success is not evidence that real
embedding, Jev, or extraction calls ran.

### Existing local data cannot migrate or import cleanly

Do not rewrite immutable rows to force a migration. A database populated by an old,
uncommitted implementation can contain state that no released migration produced.
Preserve it for investigation and verify the current code against a separately
named, freshly migrated database.

## Further documentation

- [Knowledge graph generation and operation](kg-generation.md) — detailed runtime,
  safety, versioning, and export contracts.
- [Question-independent extraction review](kg-extraction-review.md) — extraction and
  alias-review procedure.
- [Knowledge graph glossary](kg-glossary.md) — project terminology.
- [Corpus documentation](../corpus/README.md) — supplied source inventory and
  normalization notes.
- [Fixture documentation](../fixtures/README.md) — behavior cases and fixture
  execution.

### Implemented workflow

After the graph is built, the workflow runs a frozen hybrid retriever,
locally validates citations against the exact evidence sent to the model, and writes
main, extra-mode, and fixture answer/trace artifacts. The examples notebook is
read-only: it renders saved evidence and history without querying a database or a
provider.

```bash
bin/runner python -m potency_kg.cli answer \
  --config configs/casgevy-potency.json --output results
bin/runner python -m potency_kg.cli run-fixture-answers \
  --config configs/casgevy-potency.json --output results
bin/runner python -m potency_kg.cli evaluate \
  --results results --cases results/evaluation/cases.jsonl
bin/runner jupyter nbconvert --to notebook --execute \
  notebooks/06_retrieval_and_answering_examples.ipynb \
  --output /tmp/potency-retrieval-examples.ipynb
```

The frozen retrieval policy, prompt identity, snapshot-binding rule, and P3
vector-versus-graph comparison are documented in
[Retrieval method and configuration](retrieval-method.md) and
[P3 vector versus graph comparison](p3-vector-graph-comparison.md).

### Recorded outcome and limits

- Real runs produced seven required main answer/trace pairs, five extra vector-mode
  pairs, and three fixture answer/trace pairs.
- The final offline suite recorded 353 passing tests. Saved generation usage is
  54,886 main tokens plus 7,323 fixture tokens; provider dollar rates were not
  supplied, so no actual dollar cost is claimed.
- Evaluation preserves five frozen E-cases as `unscorable`, rather than scoring
  them as failures, because no saved run matches their questions.
- Final delivery verification is intentionally blocked: the preserved real fixture
  report lacks the contract-required `scenarios` field, so no delivery manifest was
  produced. The graph-signal measurement is also marked `not_run` because external
  embedding approval for its corpus-derived questions was denied.

Next steps are to repair the fixture report through its provenance-preserving
producer, deliberately run and save the E-case attempts, then rerun
`verify-deliverables`. The root `README.md` remains unchanged; this document is the
project's documentation-local handoff README.
