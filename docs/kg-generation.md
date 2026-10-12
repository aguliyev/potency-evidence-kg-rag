# Knowledge graph generation and operation

This workflow creates the evidence knowledge graph only. Its published contract is
`results/kg/current/`; a later hybrid-RAG phase may consume that directory, but query
answering, UI, alias approval, and expert adjudication are outside this stage.

## Protected inputs and runtime boundary

Run commands from the repository root. `configs/`, `corpus/`, `fixtures/`, and
`schema/` are supplied, read-only inputs. Compose mounts all four directories
read-only. Snapshot them before work and verify the identical path/hash set after
work:

```bash
bin/runner python -m potency_kg.cli hash-inputs --output /tmp/kg-protected-before.json
bin/runner python -m potency_kg.cli hash-inputs --verify /tmp/kg-protected-before.json
```

The snapshot contains relative paths and SHA-256 digests, never source contents.
`run-kg`, individual config-driven commands, fixture runs, and inspection also apply
their own pre/post protection checks. A mismatch fails the operation.

Build, start, and migrate the local stack with:

```bash
docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env build
bin/start
bin/runner alembic upgrade head
```

`bin/runner` is the supported executable boundary. It uses the same Compose images,
database, environment files, `PYTHONPATH`, and read-only supplied-input mounts as the
notebook service.

### Local pre-commit database residue

The committed scoped-ID algorithm was introduced together with the graph-store
migration and has not changed. A developer database populated by an earlier,
uncommitted implementation can therefore contain structural IDs that no repository
migration ever produced. Current import correctly refuses those rows as an
incomplete immutable structural graph; this is not a supported in-place upgrade
case and is not evidence that a clean-checkout migration failed. Preserve such a
database for investigation and use a separately named, freshly migrated database
for verification. Do not rewrite or silently accept its immutable rows.

## Settings and credential gates

Settings are loaded lazily by the stage that needs them. Structural work constructs
no model adapter and needs none of the model variables.

| Stage | Variables | Contract |
|---|---|---|
| Embedding | `EMBEDDING_LLM_URI`, `EMBEDDING_LLM_MODEL`, `EMBEDDING_LLM_API_KEY` | Official origin `https://api.openai.com/v1`, model `text-embedding-3-large`, float encoding, 3072 dimensions |
| Extraction | `LLM_URI`, `LLM_MODEL`, `LLM_API_KEY` | Instructor model identifier is derived as `openai/<LLM_MODEL>` |
| Jev classification | `TYPESAFE_API_KEY` | One TypeSafe/Jev request carries all seven fixed Noul questions |

No stage reads or remaps `OPENAI_API_KEY`. Real adapters are disabled unless
`KG_RUN_REAL_MODELS=true`; a complete `run-kg` semantic run additionally requires
all three stage-specific setting groups to validate. With the gate absent or any
group missing/invalid, `run-kg` performs the structural workflow and reports
`semantic_status: not_run`. It never substitutes another provider or credential.
The inspection notebook has a second, independent
`KG_ENABLE_RETRIEVAL_SMOKE=true` gate for its single query-embedding call and trace
write.

Endpoints are rejected if they contain userinfo, query parameters, or fragments.
Stored and printed reports contain sanitized origins and credential-variable names,
not credential values, authorization headers, endpoint query strings, exception
messages from providers, or full raw model responses. CLI failures return nonzero
and emit only a generic, JSON-formatted diagnostic.

## Commands and stage order

Every version label is derived from declared inputs; the CLI accepts no caller-
controlled embedding, Jev, extraction, index, or graph version label.

```bash
bin/runner python -m potency_kg.cli preflight --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli import --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli embed --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli classify-jev --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli extract --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli materialize --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli refresh-graph --config configs/casgevy-potency.json --mode semantic
bin/runner python -m potency_kg.cli export-kg --config configs/casgevy-potency.json --output results/kg
bin/runner python -m potency_kg.cli inspect-kg --config configs/casgevy-potency.json
```

The one-command workflow runs `preflight`, `import`, optional `embed`,
`classify-jev`, `extract`, and `materialize`, then explicitly `refresh-graph` and
`export-kg`:

```bash
bin/runner python -m potency_kg.cli run-kg \
  --config configs/casgevy-potency.json \
  --output results/kg
```

Without the real-model gate, refresh mode is `structural`. With the gate and every
valid setting group, refresh mode is `semantic`. Reports include run identifiers,
derived versions, counts, reuse decisions, sanitized model audits, graph validation,
and artifact paths. A stage failure stops later stages and returns nonzero; export
and current promotion occur only after graph validation.

For a manual structural-only run:

```bash
bin/runner python -m potency_kg.cli preflight --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli import --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli refresh-graph --config configs/casgevy-potency.json --mode structural
bin/runner python -m potency_kg.cli export-kg --config configs/casgevy-potency.json --output results/kg
```

Structural exports set `index_version`, `embedding_version`, and semantic-version
collections to `null`, and semantic statuses to `not_run`. Every valid active
passage still has document/passage structural nodes and their structural edge.

## Selection, extraction, and reuse tradeoffs

Jev persists every probability before selection. The inclusive default thresholds
are `0.50` for `product_context`, `alias_identity`, `potency_strategy`,
`assay_evidence`, and `regulatory_history`, and `0.35` for `review_conclusion` and
`redaction_uncertainty`. Lower thresholds improve recall but schedule more generation
calls and can add cost/noise. Higher thresholds reduce calls and noise but increase
false-negative risk. Threshold-only changes reuse the immutable Jev response and
derive a new selection version. No selected bundle means no generation call. A Jev
failure never falls back to unrestricted extraction.

Embeddings are cached by normalized content and provider identity. Unchanged text
reuses vectors, while source membership participates in the separately derived index
identity. Jev responses, selections, extractions, materializations, and graph
versions are independently derived from their declared inputs. An unchanged import
is idempotent after reconnect and reports unchanged source versions. Source
correction requires an explicit replacement declaration; history remains immutable.

The extraction schema is built from only the selected bundles plus universal
assertion/provenance fields. Local references and exact supporting quotes must
validate. A malformed or ungrounded response creates no partial active extraction.

## Transaction and last-known-good guarantees

- Import is manifest-first and source-transactional. A failed source activation does
  not leave partial passage or structural rows.
- Embedding batches and index snapshots commit atomically; failed batches do not
  claim success.
- Jev response/selection history is immutable. The mutable processing row exposes a
  machine-readable current state and reason.
- A validated extraction is resolved and materialized with its current pointer in
  one owner transaction. Assertion, mention, alias-candidate, node, and edge
  provenance is scope constrained by the database.
- AGE refresh builds a shadow graph, validates counts, labels, endpoints,
  provenance, and absence of passage-text leakage, then atomically activates it.
  Any failure preserves the previous active graph. The former graph is cleaned only
  after activation succeeds.
- Semantic refresh refuses unresolved active passages unless the explicit
  `--allow-partial` override is supplied; omissions are then recorded exactly.
- Export writes an immutable `results/kg/history/<graph-version>/`, validates its
  manifest and file hashes, then atomically promotes the relative
  `results/kg/current` symlink. A failed build or validation leaves the last-known-
  good current target unchanged. Re-export of an identical valid version reuses its
  history and repoints current without rewriting history.

Each immutable export contains `manifest.json`, `import-report.json`,
`extraction-coverage.json`, `graph-validation.json`, `graph/nodes.jsonl`, and
`graph/edges.jsonl`. The manifest carries scope, active source heads, source hashes,
projection mode, derived versions, profile/prompt provenance, statuses, counts, and
artifact hashes.

## Fixtures, notebooks, and later handoff

Run supplied behavior fixtures with deterministic offline adapters only:

```bash
bin/runner python -m potency_kg.cli run-kg-fixtures \
  --output results/kg/fixtures
```

The output root must be exactly the repository's `results/kg/fixtures`; generated
manifests, normalized records, graphs, exports, and traces stay there. Input paths
are resolved through the single fixture catalog translation path. Fixture questions
and expected answers are not imported as evidence. Rerunning requires a clean output
location because immutable scenario directories are created with refusal on
collision; the underlying source/import/version operations themselves are
idempotent and reconnect is verified.

`notebooks/01_build_knowledge_graph.ipynb` is a thin orchestrator over
`run_kg_workflow`; it owns no pipeline implementation and verifies refresh/export are
the final stages. `notebooks/02_inspect_knowledge_graph.ipynb` reads counts and active
graph state by default. Only an explicit retrieval-smoke gate performs one query
embedding and writes a bounded, sanitized trace containing the query hash, versions,
hit/seed/edge identifiers, limits, and exact selected source references.

Downstream hybrid RAG must consume the validated target of `results/kg/current/`,
not a staging directory or database work-in-progress. It must preserve the export's
scope and provenance and treat the graph as an intermediate evidence contract, not
as an answer-generation result.

## Deferred alias correction

Trusted aliases come only from the application-owned project profile and carry
descriptor/profile provenance. Model-extracted alias candidates retain exact
evidence provenance and remain pending. Human approval, redirects, UI, and review-
triggered rematerialization are deliberately deferred. A later correction workflow
must approve a candidate, create an auditable redirect, derive affected versions,
rematerialize impacted evidence, rebuild/validate a shadow graph, and atomically
publish a new export while preserving all prior history.
