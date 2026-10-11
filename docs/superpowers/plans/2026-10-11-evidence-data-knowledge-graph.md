# Evidence Data and Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import and account for the complete supplied corpus, selectively extract a provenance-grounded semantic knowledge graph, preserve all source and extraction history in PostgreSQL, and refresh a project-isolated Apache AGE projection that can be inspected from Jupyter notebooks.

**Architecture:** Python modules under `lib/potency_kg/` contain all behavior. Two thin notebooks call those modules: one builds the graph stage by stage and explicitly calls `refresh_graph(...)`; the other inspects graph coverage, provenance, and a vector-seeded traversal smoke test. Relational tables retain every valid source version and are authoritative, while AGE is a disposable projection of complete active source heads and atomically activated extraction runs. All executable commands run inside Docker Compose, and integration tests use a disposable PostgreSQL database.

**Tech Stack:** Python 3.12, Pydantic 2, Instructor, TypeSafe/Jev, SQLAlchemy 2, Psycopg 3, Alembic, PostgreSQL 17, pgvector, Apache AGE 1.7, pytest, JupyterLab, nbconvert.

**Spec:** `ASSIGNMENT.md` sections 1–2 and the graph-retrieval portion of section 3; `MODEL_ACCESS.md`; `corpus/README.md`; `corpus/manifest.json`; `schema/graph-core.json`; `schema/output-contract.json`; and `fixtures/BEHAVIOR_CASES.md` for correction/conflict semantics.

## Global Constraints

- This plan covers KG generation, inspection, export, and a retrieval smoke test only. It does not implement P1–P6 answers, answer-generation traces, formal evaluation, AWS architecture/costing, or final submission handoff.
- Put all new documentation in `docs/`. Keep the root `README.md` unchanged.
- Do not read or modify `docs/scratchpad.md`.
- Keep all application behavior in importable modules under `lib/potency_kg/`; notebooks contain orchestration and presentation only.
- Run migrations, tests, CLIs, and notebook execution inside Docker Compose. Do not require a configured host Python environment.
- Treat `(tenant_id, project_id, source_id, source_version, passage_id)` as source-passage identity. Preserve the supplied `passage_id` for citations and call the scoped graph identifier `passage_node_id`.
- Import and reconcile all 791 normalized records and create an inspectable `SourceDocument` representation for all 24 manifest sources.
- Treat files under `corpus/raw/` as source of truth, `corpus/manifest.json` as the document catalog, and `corpus/records.jsonl` as the supplied mechanical passage extraction.
- Do not ingest assignment text, questions, schemas, fixture contracts, model-access files, or generated outputs as evidence.
- Preserve every source version, model run, extraction run, assertion, and graph export. Never rewrite historical provenance.
- A source version becomes active only after all expected passages for that source validate and import successfully.
- Keep tenant and project in every project-owned relational key, graph ID, cache key, trace, run record, graph name, and graph-selection operation.
- Do not reuse embedding-cache entries across projects. A separately authorized shared-public cache is outside this prototype.
- Never traverse from one project AGE graph into another project graph.
- Every valid imported source version and passage receives relational structural representation. Every passage in a complete active source version receives an embedding; retained embeddings are not deleted when a version is later superseded. Only relevance-selected active passages receive semantic extraction.
- Every passage records one semantic-processing outcome and reason: `structural_only`, `selected`, `extracted`, or `failed`.
- Relevance selection is corpus-wide and may not use P1–P6 IDs, fixture names, expected answers, or a hand-picked document allowlist.
- Automatically extracted assertions and relationships remain `unreviewed`; extracted aliases remain `pending_review`. Model agreement is not human review.
- Keep source authority/status, assertion regulatory state, and extraction review state independent.
- Treat document-level `subject_scope` only as an entity-resolution hint. Create `applies_to` edges only from explicit passage evidence.
- Trusted configured aliases may merge entities. Pending or rejected aliases may not affect resolution.
- Assertions are atomic. Every evidence-bearing relation references the local assertion that supports it.
- Every atomic claim has one immutable relational assertion and exactly one graph representation: `Assertion` or, for an explicit reviewer/agency conclusion, `ReviewConclusion`.
- Preserve equivalent-looking assertions from different passages as separate evidence records. Represent proposed, requested, revised, accepted, rejected, and unresolved chronology as separate assertions rather than mutating one assertion.
- Do not create `contradicts` edges unless a passage explicitly states the contradiction. Independently supported opposing assertions remain separate and unresolved.
- Do not merge redacted markers across passages. Redaction identity includes full scoped passage identity and occurrence index.
- Do not infer concealed `(b)(4)` content.
- Use original source text for citations and exported evidence. Redaction-substituted text is stored separately and used only as model input.
- Keep full passage text in relational PostgreSQL. AGE stores only identifiers, locators, compact attributes, and provenance needed for traversal.
- Make unchanged import, embedding, relevance classification, extraction, materialization, export, and AGE refresh idempotent.
- Use parameterized SQL for values. AGE graph names and labels come only from internal allowlisted functions and are validated before identifier interpolation.
- Derive index, relevance, extractor, and graph versions from model IDs, prompt/schema hashes, normalization settings, and active inputs; callers may not label new work with arbitrary versions.
- `refresh_graph()` projects only active source heads and atomically activated completed extraction runs. Superseded history remains relational and never enters default AGE traversal.
- Semantic refresh fails by default if any selected passage has an unresolved extraction failure. `allow_partial=True` is an explicit, reported override.
- Keep the implementation small: no LangChain, LlamaIndex, NetworkX, generic ontology framework, PDF parser, OCR pipeline, graph visualization, or incremental AGE synchronization.

---

## Accepted Design Decisions

### Canonical storage and AGE projection

Relational `kg_node`, `kg_edge`, and normalized source-reference tables are authoritative. Apache AGE lives in the same PostgreSQL instance but is a query projection, not a second source of truth. Refresh drops and rebuilds only the internally generated graph for that scope, then validates logical IDs and endpoint integrity. A failed refresh leaves relational data intact and is recoverable by rerunning it.

`refresh_graph(engine, scope, mode="semantic", allow_partial=False, force=False)` derives its graph version from projection-schema version, active source heads, active extraction runs, projection mode, partial-policy flag, and the exact omitted/failed passage set. The version is reused only when that same version is currently installed in the scope's AGE graph and still validates; `force=True` rebuilds the same derived version. Both reused and rebuilt attempts create refresh-run records.

### Current state and history

`source_head` selects the active complete version of each logical source. Relational structural nodes and edges represent every valid imported version, including incomplete and superseded versions. `passage_processing.active_extraction_run_id` selects the single extraction used for current materialization of a passage and changes only in the transaction that completely materializes that extraction. Relational history is immutable. The current AGE projection includes only structural records from active source heads and semantic records supported by active passages and active extraction runs. Shared semantic nodes are projected only when at least one current edge or current source reference needs them; projected source references exclude superseded passages.

### Structural and semantic coverage

The manifest creates every document record before passage import. Every valid passage receives a relational structural node and `contains_passage` edge. Complete active versions receive embeddings; those embeddings remain inspectable after later supersession. Empty or page-number-only passages are deterministically `structural_only`; passages rejected by the relevance policy also become `structural_only` with a different recorded reason. Remaining passages receive the same batched relevance policy. Selected passages receive one passage-local semantic extraction call, with bounded concurrency and versioned caching.

### Entity and assertion identity

- `SourceDocument`: scope + source ID + source version.
- `SourcePassage`: full five-part source identity; the graph ID is `passage_node_id`, while `SourceRef.passage_id` remains the supplied passage ID.
- `Application`: scope + normalized authoritative identifier such as `STN 125785`.
- `Product`: scope + configured product ID such as `casgevy`.
- Redacted entity: full scoped passage identity + redaction occurrence index.
- Other semantic entities: scope + entity type + compatible product/application context + normalized name after configured/approved alias lookup.
- Assertion record: scope + full passage identity + extractor version + passage-local assertion ID + normalized atomic claim.
- Structural edge: scope + type + endpoints.
- Evidence edge: scope + type + endpoints + assertion ID.

Configured aliases map `CASGEVY`, `exagamglogene autotemcel`, and `exa-cel` to the configured product node. Extracted alias claims are inspectable but inactive until approved. If a future approved merge joins materialized entities, retain the deprecated ID in `entity_redirect`; do not rewrite historical extraction responses.

Every atomic claim is stored in the relational `assertion` table. Its graph node type is exactly one of `Assertion` or `ReviewConclusion`. The extraction model may propose `ReviewConclusion` as the claim's graph representation, but deterministic validation accepts it only for an explicit reviewer/agency conclusion or recommendation grounded in the supporting quote and compatible source authority. A grounded but misclassified conclusion is conservatively materialized as `Assertion` with a validation warning; a missing or invalid quote fails extraction. Assay roles are attributes on evidence-bearing `used_as` edges from an Assay to its explicitly supported Product, Application, or PotencyStrategy context.

### Exact spans and redactions

Extraction uses `model_text`, where supported `(b)(4)` spacing variants are replaced by passage-scoped opaque tokens. Each assertion and relation provides an exact supporting quote. Validation must find that quote deterministically in `model_text`; zero matches or ambiguous matches fail the extraction unless a valid occurrence index disambiguates it. Materialization maps the validated span back to original source offsets and original text before storing provenance.

Semantic entity nodes aggregate source references that establish observed names or existence. Assertion nodes, ReviewConclusion nodes, and evidence-bearing edges retain the exact original quotes that support their claims. The extraction contract includes an allowlisted source/edge/target matrix; Pydantic validation rejects relationships outside that matrix.

### Canonical terminology

`docs/kg-glossary.md` defines Source Version, Active Source Head, Assertion, ReviewConclusion, Regulatory State, Review State, Potency Strategy, Assay Role, Extraction Run, Current Graph, Processing Version, and Graph Version. Prompts, contracts, notebooks, and exports use those terms consistently.

### Notebook and CLI roles

`notebooks/01_build_knowledge_graph.ipynb` exposes the stages and reports counts, failures, selection coverage, and the final explicit `refresh_graph(...)` call. `notebooks/02_inspect_knowledge_graph.ipynb` performs read-only relational/AGE checks and a retrieval smoke test. A minimal CLI calls the same module interfaces for headless reproducibility. Full machine-readable artifacts are written under `results/kg/`; committed notebook output is limited to concise summaries.

## Planned File Structure

```text
requirements.txt
docker/compose.yaml
migrations/versions/
  0002_create_evidence_store.py
  0003_create_knowledge_graph_store.py
lib/potency_kg/
  __init__.py
  config.py
  contracts.py
  ids.py
  redactions.py
  db.py
  ingest.py
  graph_store.py
  embeddings.py
  relevance.py
  extraction.py
  aliases.py
  resolution.py
  age_projection.py
  fixture_runs.py
  retrieval.py
  export.py
  cli.py
prompts/
  relevance-v1.md
  extraction-v1.md
notebooks/
  01_build_knowledge_graph.ipynb
  02_inspect_knowledge_graph.ipynb
tests/
  conftest.py
  unit/
    test_contracts.py
    test_ids.py
    test_redactions.py
    test_relevance.py
    test_extraction.py
    test_aliases.py
  integration/
    test_migrations.py
    test_import.py
    test_embeddings.py
    test_model_processing.py
    test_resolution.py
    test_age_refresh.py
    test_retrieval_smoke.py
    test_exports.py
    test_cli.py
    test_behavior_graphs.py
docs/
  kg-generation.md
  kg-glossary.md
fixtures/configs/
  source-correction-before.json
  source-correction-after.json
  unresolved-conflict.json
results/kg/                  # generated, gitignored
  import-report.json
  extraction-coverage.json
  graph-validation.json
  graph/nodes.jsonl
  graph/edges.jsonl
  fixtures/<scenario>/<stage>/
```

## Relational Data Model

The migrations create the following project-scoped tables and constraints:

- `import_run`: run ID, scope, corpus snapshot, manifest/records hashes, status, timestamps, and reconciled counts.
- `import_error`: run ID, optional source identity, optional supplied passage ID, stage, code, and sanitized detail.
- `source_document`: scope, source/version identity, manifest metadata, raw path/hash, expected passage count, import completeness, and creating run.
- `source_head`: one active complete version per `(tenant_id, project_id, source_id)`.
- `source_passage`: full five-part source identity, `passage_node_id`, exact text/hash, locator, authority, status, subject scope, and metadata.
- `embedding_cache`: scope, provider, model, dimensions, input hash, vector, usage, and timestamps.
- `passage_embedding`: full passage identity, cache row, index version, and active state.
- `model_run`: scope, provider/model, task, derived processing version, input hash, sanitized usage/error, elapsed time, status, sanitized raw response text/JSON when available, validation errors, and validated response JSON when successful.
- `passage_processing`: full passage identity, deterministic/relevance decisions, outcome/reasons, relevance run, extraction runs, and active extraction run.
- `redaction_mention`: full passage identity, occurrence, raw marker, opaque token, and original/model offsets.
- `entity_mention`: extraction run, passage-local ID, exact original span/text, entity type/context, and resolved node ID.
- `entity_alias`: normalized alias, target node, relation type, status, proposer, exact source reference, evidence quote, and review fields.
- `entity_redirect`: deprecated node ID, canonical node ID, and approved merge provenance.
- `assertion`: assertion ID, extraction run, full passage identity, local assertion ID, original supporting quote/offsets, atomic claim, state, authority/status, event date, actor/version/time, and review state.
- `kg_node`: scoped logical node ID, type, label, attributes, optional assertion ID, actor/version/time, and review state where evidence-bearing.
- `kg_node_source_ref`: composite node key plus full source reference and original quote/offsets where applicable.
- `kg_edge`: scoped edge ID, type, composite endpoint keys, status, assertion ID when evidence-bearing, actor/version/time, and review state.
- `kg_edge_source_ref`: composite edge key plus full source reference and original quote/offsets.
- `graph_scope`: scope, generated AGE graph name, last successful graph version, mode, and refresh timestamp.
- `graph_refresh_run`: scope, derived graph version, mode, partial/force flags, omitted/failed passage set, reused/rebuilt status, expected/projected counts, validation result, errors, and timestamps.

Use composite foreign keys `(tenant_id, project_id, node_id)` for both edge endpoints. Do not rely on application validation alone for endpoint scope isolation. Partial unique indexes enforce one active passage embedding per index version and one active extraction run per passage.

---

### Task 1: Make the Container the Reproducible Runtime

**Files:**
- Modify: `requirements.txt`
- Modify: `docker/compose.yaml`
- Modify: `docker/Dockerfile.notebook`
- Create: `bin/runner`
- Create: `tests/conftest.py`
- Create: `tests/test_runtime.py`
- Create: `lib/potency_kg/__init__.py`

**Interfaces:**
- Produces: a shared Python image and `/workspace` repository mount used by Jupyter, migrations, tests, and CLI commands; `bin/runner <command...>` executes arbitrary headless commands in the runner container.
- Consumes: existing `etc/.env`, `etc/secrets/.env`, PostgreSQL health check, and Alembic configuration.

- [ ] **Step 1: Add a failing runtime smoke test**

Create `tests/conftest.py` with a session-scoped disposable-database fixture. It connects to the maintenance database, generates and prints a name matching `^potency_kg_test_[a-f0-9]{12}$`, creates that database, runs Alembic against it, yields its URL, terminates only connections to that exact validated database during teardown, and drops it even when tests fail. Create `tests/test_runtime.py` with a smoke test that imports `potency_kg` and asserts that `Path("/workspace/ASSIGNMENT.md").is_file()`.

- [ ] **Step 2: Run the test in the current image and verify failure**

Run: `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env run --rm notebook python -c 'from pathlib import Path; assert Path("/workspace/ASSIGNMENT.md").is_file()'`

Expected: failure because the current image/mount does not expose the complete repository root.

- [ ] **Step 3: Add the root mount and headless runner**

Add the direct pin `nbconvert==7.16.6` to `requirements.txt`. Bind-mount `../:/workspace` read-write for `notebook`, `migrate`, and a new `runner` service so newly created migrations are visible without rebuilding the image. Give `runner` the same image, environment, healthy-database dependency, working directory, and `PYTHONPATH=/workspace/lib`; do not make it depend on the persistent `migrate` service because tests create and migrate their own database. Create `bin/runner` as a strict Bash wrapper around `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env run --rm runner "$@"`. Document in a Compose comment that this trusted local-development mount exposes `.git` and `etc/secrets` and is not a production pattern.

- [ ] **Step 4: Verify imports and test discovery inside the runner**

Run: `bin/runner python -c "import potency_kg; print(potency_kg.__file__)"`

Expected: the printed path begins `/workspace/lib/potency_kg/`.

Run: `bin/runner pytest tests/test_runtime.py -v`

Expected: the runtime smoke test passes inside the container.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt docker/compose.yaml docker/Dockerfile.notebook bin/runner lib/potency_kg/__init__.py tests/conftest.py tests/test_runtime.py
git commit -m "chore: add reproducible KG runner"
```

### Task 2: Define Contracts, Scoped IDs, and Redaction Spans

**Files:**
- Create: `lib/potency_kg/contracts.py`
- Create: `lib/potency_kg/ids.py`
- Create: `lib/potency_kg/redactions.py`
- Create: `docs/kg-glossary.md`
- Create: `tests/unit/test_contracts.py`
- Create: `tests/unit/test_ids.py`
- Create: `tests/unit/test_redactions.py`

**Interfaces:**
- Produces: `Scope`, `SourceRef`, `SourcePassageRecord`, `EntityMentionDraft`, `AliasClaimDraft`, `AssertionDraft`, `RelationDraft`, `PassageExtraction`, `ResolvedExtraction`, `ProjectionSnapshot`, `EmbeddingPolicy`, `RelevancePolicy`, `ExtractionPolicy`, `GraphNodeRecord`, `GraphEdgeRecord`, and `ALLOWED_RELATION_ENDPOINTS`.
- Produces: `document_id`, `passage_node_id`, `application_id`, `configured_product_id`, `name_entity_id`, `assertion_id`, `edge_id`, `graph_name_for_scope`, and `derive_processing_version`.
- Produces: `substitute_redactions(source_ref, text) -> RedactionSubstitution` and `restore_model_span(substitution, start, end) -> OriginalSpan`.

- [ ] **Step 1: Write failing contract and identity tests**

Test unknown types, atomic non-empty assertion claims, mutually exclusive `Assertion`/`ReviewConclusion` graph representation, relation `assertion_local_id` linkage, the allowed source/edge/target matrix, `used_as.role` and context targets, local endpoint validation, supporting-quote occurrence validation, default `review_state="unreviewed"`, deterministic scoped IDs, STN/BLA normalization, deterministic processing versions, and graph names matching `^kg_[a-f0-9]{16}$`.

- [ ] **Step 2: Write failing redaction tests**

Cover `(b)(4)`, `(b) (4)`, whitespace variants, multiple occurrences, scope-separated tokens, immutable original text, model-to-original offset restoration, and rejection of concealed-value reconstruction.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_contracts.py tests/unit/test_ids.py tests/unit/test_redactions.py -v`

Expected: import failures for the missing modules.

- [ ] **Step 4: Implement strict contracts and canonical IDs**

Use explicit enums derived from the supplied graph core plus `mentions`, `contradicts`, `supersedes`, and `supplements`. Define `ALLOWED_RELATION_ENDPOINTS` as exact source-type/edge-type/target-type tuples and require `used_as` edges to carry an assay-role enum and target Product, Application, or PotencyStrategy. Serialize digest inputs as canonical JSON arrays. Keep `SourceRef.passage_id` unchanged and use `passage_node_id` only for the scoped graph identifier. Derive processing versions from canonical model ID, prompt/schema hashes, and normalization settings.

- [ ] **Step 5: Implement reversible redaction substitution**

Store ordered original/model spans for every replacement. `restore_model_span` must return the exact original substring and offsets and must fail when a requested span crosses an unmappable region.

- [ ] **Step 6: Define canonical KG terminology**

Create `docs/kg-glossary.md` with concise, implementation-free definitions for Source Version, Active Source Head, Assertion, ReviewConclusion, Regulatory State, Review State, Potency Strategy, Assay Role, Extraction Run, Current Graph, Processing Version, and Graph Version. Define ReviewConclusion as an explicit reviewer/agency conclusion, not a synonym for any accepted assertion.

- [ ] **Step 7: Run focused tests**

Run: `bin/runner pytest tests/unit/test_contracts.py tests/unit/test_ids.py tests/unit/test_redactions.py -v`

Expected: all focused tests pass.

- [ ] **Step 8: Commit**

```bash
git add lib/potency_kg/contracts.py lib/potency_kg/ids.py lib/potency_kg/redactions.py docs/kg-glossary.md tests/unit
git commit -m "feat: define scoped KG contracts and evidence spans"
```

### Task 3: Create the Evidence and Graph Schema

**Files:**
- Create: `migrations/versions/0002_create_evidence_store.py`
- Create: `migrations/versions/0003_create_knowledge_graph_store.py`
- Create: `lib/potency_kg/db.py`
- Create: `tests/integration/test_migrations.py`

**Interfaces:**
- Consumes: `0001_enable_extensions.py`.
- Produces: `create_engine_from_env()`, `transaction(engine)`, and the tables and composite constraints listed in this plan.

- [ ] **Step 1: Write a failing migration smoke test**

Use the disposable database fixture, upgrade the empty database to `head`, assert every planned table exists, reject duplicate full passage identity, reject a `kg_edge` whose endpoint belongs to another project, and reject two active extraction pointers for the same passage. Assert the printed database name has the required prefix and the database is absent after fixture teardown.

- [ ] **Step 2: Run the migration test and verify failure**

Run: `bin/runner pytest tests/integration/test_migrations.py -v`

Expected: failure because migrations `0002` and `0003` do not exist.

- [ ] **Step 3: Implement the evidence-store migration**

Create run, error, source, source-head, passage, scoped embedding, model-run, processing, redaction, mention, alias, redirect, and assertion tables with PostgreSQL `JSONB`, timezone-aware timestamps, explicit status checks, and composite keys. Model runs store sanitized raw response text/JSON, structured validation errors, validated output when successful, and a derived processing version. Assertions store graph representation type and validation warnings.

- [ ] **Step 4: Implement the canonical-graph migration**

Create graph nodes, edges, normalized source references, graph scope, and graph refresh runs. Use composite endpoint foreign keys and indexes for scope/type, active heads, active extraction runs, normalized aliases, source-reference joins, and embedding lookups.

- [ ] **Step 5: Implement connection helpers**

Set the relational `search_path` explicitly per connection and register pgvector with Psycopg 3. AGE-specific setup remains inside `age_projection.py` so ordinary relational callers do not depend on AGE session state.

- [ ] **Step 6: Run migrations and tests**

Run: `bin/runner alembic upgrade head`

Run: `bin/runner pytest tests/integration/test_migrations.py -v`

Expected: migration succeeds and all schema isolation tests pass.

- [ ] **Step 7: Commit**

```bash
git add migrations/versions/0002_create_evidence_store.py migrations/versions/0003_create_knowledge_graph_store.py lib/potency_kg/db.py tests/integration/test_migrations.py
git commit -m "feat: add scoped evidence and graph schema"
```

### Task 4: Validate Configuration and Import the Manifest First

**Files:**
- Create: `lib/potency_kg/config.py`
- Create: `lib/potency_kg/ingest.py`
- Create: `lib/potency_kg/graph_store.py`
- Modify: `configs/casgevy-potency.json`
- Create: `tests/integration/test_import.py`

**Interfaces:**
- Produces: `load_project_config(path) -> ProjectConfig`, `preflight_corpus(config) -> CorpusPreflightReport`, and `import_corpus(engine, config) -> ImportReport`.
- Produces from `graph_store.py`: `write_structural_version(engine, source_id, source_version) -> GraphWriteReport`, `write_semantic_extraction(engine, passage, run_id, resolved) -> GraphWriteReport`, and `load_current_projection(engine, scope, mode, allow_partial) -> ProjectionSnapshot`.
- Consumes: manifest document counts/hashes, source records, configured aliases, and scoped identity functions.

- [ ] **Step 1: Write failing preflight and import tests**

For the official corpus, assert manifest/records snapshot and scope agreement, records-file SHA-256, raw-file SHA-256, 24 documents, 791 passages, manifest per-source counts, passage text hashes, and rejection of paths escaping the package root. For synthetic fixtures without a manifest, require an explicit `synthetic=true` descriptor and derive expected per-source counts from only the input JSONL files declared by that fixture. Assert a document row exists even when one of its passages is invalid.

- [ ] **Step 2: Write source-activation and idempotency tests**

Assert the first import reports imported rows; the second reports unchanged rows; incomplete source versions never become heads; complete explicit replacements advance only their declared source head; historical rows remain; and totals reconcile as `imported + unchanged + failed + omitted = input`.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_import.py -v`

Expected: missing import/configuration functions.

- [ ] **Step 4: Implement model and project configuration validation**

Resolve project paths against the project-config directory and fixture input paths against the package root. Validate package edition, scope, corpus snapshot, source records, graph schema, output contract, configured product aliases, and model-access file location. Require the manifest for the official corpus; allow its absence only for explicitly synthetic fixture descriptors. Add explicit configured aliases for `CASGEVY`, `exagamglogene autotemcel`, and `exa-cel` to `configs/casgevy-potency.json`. Never include credential values in validated dumps or logs.

- [ ] **Step 5: Implement manifest-first streaming import**

Create document versions from the manifest before reading JSONL. Validate each passage independently, retain explicit failures, and update `source_head` only after imported/unchanged valid passage counts equal the manifest expectation for that source. Preserve `subject_scope` as metadata and a resolution hint; do not create `applies_to` relationships from it. For fixture replacements, require the declared replacement relationship rather than using issue dates.

- [ ] **Step 6: Materialize structural graph records**

Implement `graph_store.py` as a deep domain interface, not a generic node/edge repository. `write_structural_version` upserts document and passage nodes plus `contains_passage` edges for every valid imported version, including incomplete and superseded versions. Structural edges use actor `system`, version `structural-v1`, review state `unreviewed`, and empty source references because they describe storage structure rather than scientific claims. `write_semantic_extraction` and `load_current_projection` centralize identity, provenance, active-run, and active-source selection rules used by later tasks.

- [ ] **Step 7: Run focused and full-corpus preflight tests**

Run: `bin/runner pytest tests/integration/test_import.py -v`

Expected: all tests pass; the read-only preflight reports 24 sources and 791 valid normalized records.

- [ ] **Step 8: Commit**

```bash
git add configs/casgevy-potency.json lib/potency_kg/config.py lib/potency_kg/ingest.py lib/potency_kg/graph_store.py tests/integration/test_import.py
git commit -m "feat: import manifest-backed evidence"
```

### Task 5: Embed Every Active Passage with Scoped Caching

**Files:**
- Create: `lib/potency_kg/embeddings.py`
- Create: `tests/integration/test_embeddings.py`

**Interfaces:**
- Produces: `EmbeddingClient.embed(texts: list[str]) -> EmbeddingBatch`, `embed_active_passages(engine, scope, client, policy) -> EmbeddingReport`, and `vector_search(engine, scope, index_version, query_vector, limit) -> list[PassageHit]`. `EmbeddingReport.index_version` is derived from the client model identity, vector dimension, and normalization policy.
- Consumes: completed model-access configuration for real clients; tests use a deterministic fake.

- [ ] **Step 1: Write failing cache and retrieval tests**

Verify finite vectors, consistent dimensions, batching at configured limits, no repeated call for the same scoped model/input hash, new calls after text/model/project changes, current-source filtering, explicit index-version filtering, and deterministic fake-client ranking.

- [ ] **Step 2: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_embeddings.py -v`

Expected: missing embedding module.

- [ ] **Step 3: Implement the client seam and model-access gate**

Real clients refuse to run unless setup status, model IDs, limits, both verification timestamps, acknowledged constraints, and the required credential environment-variable name are present. Fake clients identify themselves as `test-double` and cannot write a real-run status.

- [ ] **Step 4: Implement scoped caching and exact search**

Cache keys include scope, provider, model, dimensions, and input hash. Derive the index version; never accept a caller-provided label for newly embedded work. Embed passages from complete active source versions and retain their cached vectors and passage bindings if those versions are later superseded. Store vectors without a dimension-specific ANN index; for 791 passages, filter current scoped rows by selected existing index version before exact cosine ordering.

- [ ] **Step 5: Run tests**

Run: `bin/runner pytest tests/integration/test_embeddings.py -v`

Expected: all embedding tests pass without external calls.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/embeddings.py tests/integration/test_embeddings.py
git commit -m "feat: embed passages with scoped caching"
```

### Task 6: Classify Relevance and Extract Validated Atomic Claims

**Files:**
- Create: `prompts/relevance-v1.md`
- Create: `prompts/extraction-v1.md`
- Create: `lib/potency_kg/relevance.py`
- Create: `lib/potency_kg/extraction.py`
- Create: `tests/unit/test_relevance.py`
- Create: `tests/unit/test_extraction.py`
- Create: `tests/integration/test_model_processing.py`

**Interfaces:**
- Produces: `RelevanceClient.classify(batch) -> RelevanceBatch`, `ExtractionClient.extract(passage) -> PassageExtraction`, `classify_pending_passages(engine, scope, client, policy) -> RelevanceReport`, `extract_selected_passages(engine, scope, client, policy) -> ExtractionReport`, and `extract_explicit_alias_patterns(text) -> list[AliasClaimDraft]`.
- Consumes: an injected real or fake model-client adapter, model-access limits, scoped model runs, extraction contracts, reversible redaction mappings, and derived processing versions. The two high-level functions own run lookup, batching/concurrency, retries, response validation, and persistence.

- [ ] **Step 1: Write failing relevance tests**

Cover deterministic page-number/empty skips, batched multi-label decisions, a persisted reason for every passage, no question IDs or document allowlist in classifier input, and retry behavior only for declared transient failures.

- [ ] **Step 2: Write failing extraction tests**

Cover atomic claims, mutually exclusive Assertion/ReviewConclusion representation, explicit conclusion qualification, conservative downgrade with validation warning, valid allowed-matrix endpoints, evidence-bearing `used_as` roles, relation-to-assertion linkage, exact unique supporting quotes, occurrence-index disambiguation, original-span restoration, parenthetical abbreviations, `formerly` aliases, redaction non-inference, and rejection of partial invalid output.

- [ ] **Step 3: Write failing persistent-processing tests**

In `tests/integration/test_model_processing.py`, verify completed-run reuse, derived relevance/extractor versions, transient-only retries, sanitized invalid raw-response storage, validation-error storage, no materialization from failed output, and explicit `test-double` provenance for fake clients.

- [ ] **Step 4: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_relevance.py tests/unit/test_extraction.py tests/integration/test_model_processing.py -v`

Expected: missing relevance/extraction modules.

- [ ] **Step 5: Write versioned prompts**

The relevance prompt asks fixed typed questions about potency strategy, assays, measured attributes, regulatory interactions, proposal/request/revision/conclusion, conflicts, and redactions. The extraction prompt requires allowed enums, passage-local IDs, atomic assertions, exactly one graph representation per claim, explicit ReviewConclusion evidence, relation `assertion_local_id`, exact quotes plus occurrence, explicit alias types, and no reconstruction of redacted text. Both prompts separate source authority/status, assertion regulatory state, and extraction review state.

- [ ] **Step 6: Implement persistent relevance and extraction processing**

`classify_pending_passages` uses configurable batches; `extract_selected_passages` extracts one selected passage per call with bounded concurrency. Derive processing versions and compute input hashes from scope, full passage identity, model text, prompt/schema/normalization hashes, and model ID. Reuse matching completed runs. Persist sanitized invalid raw responses and structured validation errors as failed runs. Grounded claims misclassified as ReviewConclusion become Assertion drafts with warnings; invalid quotes fail the passage. Never materialize partial invalid output.

- [ ] **Step 7: Run focused tests**

Run: `bin/runner pytest tests/unit/test_relevance.py tests/unit/test_extraction.py tests/integration/test_model_processing.py -v`

Expected: all tests pass with deterministic fake clients.

- [ ] **Step 8: Commit**

```bash
git add prompts lib/potency_kg/relevance.py lib/potency_kg/extraction.py tests/unit/test_relevance.py tests/unit/test_extraction.py tests/integration/test_model_processing.py
git commit -m "feat: select and extract grounded KG claims"
```

### Task 7: Resolve Aliases and Materialize the Current Semantic Graph

**Files:**
- Create: `lib/potency_kg/aliases.py`
- Create: `lib/potency_kg/resolution.py`
- Create: `tests/unit/test_aliases.py`
- Create: `tests/integration/test_resolution.py`

**Interfaces:**
- Produces: `bootstrap_configured_entities_and_aliases(engine, config) -> AliasBootstrapReport`, `record_alias_candidates(engine, passage, run_id, extraction) -> AliasRecordReport`, `resolve_mention(engine, scope, mention, context) -> ResolvedEntity`, and `materialize_and_activate_extraction(engine, passage, run_id) -> MaterializationReport`.
- Consumes: configured CASGEVY aliases, exact original spans, `write_semantic_extraction`, allowed relation endpoints, and the distinction between resolution hints and evidence-bearing context.

- [ ] **Step 1: Write failing alias tests**

Verify `CASGEVY`, `exagamglogene autotemcel`, and `exa-cel` resolve to one configured product node; extracted aliases remain inactive while pending; rejected aliases remain inspectable; and similarity alone never merges entities.

- [ ] **Step 2: Write failing materialization tests**

Verify resolution priority, STN/BLA normalization, context-separated generic assay names, deterministic passage-distinct assertion IDs, separate chronological assertion states, exactly one Assertion/ReviewConclusion node per relational assertion, source/claim/review state separation, no `applies_to` edge from `subject_scope` alone, atomic transaction behavior, evidence edges linked to assertions, immutable prior extraction runs, and a single active extraction run per passage.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_aliases.py tests/integration/test_resolution.py -v`

Expected: missing alias/resolution modules.

- [ ] **Step 4: Implement configured and candidate aliases**

Create configured Product/Application anchor nodes before alias rows, then bootstrap trusted aliases from project configuration with status `configured`. Store pattern/model claims as `pending_review` with exact source reference, original quote, proposer, and extractor version. Provide approve/reject functions for future use, but no review UI.

- [ ] **Step 5: Implement two-stage materialization**

Resolve and persist all mentions first, then call `write_semantic_extraction` to materialize assertions, exactly one graph claim node per assertion, entity provenance, and evidence relations using resolved endpoints. One database transaction validates and commits the complete passage extraction and only then changes `active_extraction_run_id`. A failed transaction leaves the new run inactive and never deletes prior rows.

- [ ] **Step 6: Run tests**

Run: `bin/runner pytest tests/unit/test_aliases.py tests/integration/test_resolution.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/aliases.py lib/potency_kg/resolution.py tests/unit/test_aliases.py tests/integration/test_resolution.py
git commit -m "feat: resolve and materialize semantic KG records"
```

### Task 8: Rebuild and Validate the Current AGE Graph

**Files:**
- Create: `lib/potency_kg/age_projection.py`
- Create: `tests/integration/test_age_refresh.py`

**Interfaces:**
- Produces: `refresh_graph(engine, scope, mode="semantic", allow_partial=False, force=False) -> GraphRefreshReport`, `validate_graph(engine, scope) -> GraphValidationReport`, and `graph_counts(engine, scope) -> GraphCounts`.
- Consumes: `load_current_projection(...)` and a fixed projection-schema version. `mode="structural"` excludes semantic records.

- [ ] **Step 1: Write failing refresh tests**

Assert distinct generated graph names per project, fixed-name validation, logical IDs as AGE properties, no full passage text in AGE, no dangling endpoints, deterministic versions, unchanged validated reuse, `force=True` rebuilding, no duplicates after repeated refresh, structural-only behavior, semantic failure on unresolved selected passages, partial versions containing the exact omitted/failed set, current-head filtering, and project-isolation traversal.

- [ ] **Step 2: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_age_refresh.py -v`

Expected: missing AGE projection module.

- [ ] **Step 3: Implement safe graph rebuilding**

Load AGE and set its required session `search_path` inside the adapter. Resolve graph names only through `graph_name_for_scope` and revalidate the strict regex before interpolation. Derive the graph version from projection schema, active heads/runs, mode, partial policy, and exact omitted/failed identities. Reuse only when the requested version equals `graph_scope.last_graph_version` and the currently installed graph validates; an older refresh-run record is not proof that its graph is still installed. On rebuild, drop only that resolved graph, recreate it, create vertices by logical `node_id`, and create edges by logical `edge_id`. Parameterize every property value and omit full passage text.

- [ ] **Step 4: Implement current-record selection and validation**

Select structural records through `source_head` and semantic records through active complete extraction runs. Filter projected source references to active passages. Compare expected/projected logical ID sets, endpoint counts, scope properties, and relational/AGE totals. Do not advance `graph_scope.last_graph_version` after failed validation.

- [ ] **Step 5: Run tests**

Run: `bin/runner pytest tests/integration/test_age_refresh.py -v`

Expected: all AGE refresh tests pass against the Compose database.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/age_projection.py tests/integration/test_age_refresh.py
git commit -m "feat: refresh current project graphs in AGE"
```

### Task 9: Export the KG and Prove Vector-Seeded Traversal

**Files:**
- Create: `lib/potency_kg/retrieval.py`
- Create: `lib/potency_kg/export.py`
- Create: `tests/integration/test_retrieval_smoke.py`
- Create: `tests/integration/test_exports.py`

**Interfaces:**
- Produces: `retrieve_graph_smoke(engine, scope, index_version, query_vector, limits) -> RetrievalTrace`, `export_import_report(engine, run_id, output_path)`, `export_extraction_coverage(engine, scope, output_path)`, `export_graph_validation(engine, scope, output_path)`, `export_graph(engine, scope, output_dir, view="history")`, and `validate_kg_artifacts(output_dir) -> ExportValidationReport`.
- Writes only under caller-supplied output directory, normally `results/kg/`.
- `validate_kg_artifacts` enforces the import/node/edge/source-reference/coverage/graph-validation subset of `schema/output-contract.json`; it documents excluded answer, trace, evaluation, architecture, and handoff artifacts without modifying the supplied contract.

- [ ] **Step 1: Write failing retrieval smoke tests**

Verify scoped current vector hits map to `SourcePassage` vertices, semantic seeds come from adjacent current nodes, traversal obeys explicit direction/hop/node/edge caps, selected evidence comes only from traversed source references, exact original passages are reloaded relationally, no full text is read from AGE, and another project never appears.

- [ ] **Step 2: Write failing export tests**

Validate reconciled totals, 24 document nodes, all valid historical passage nodes in relational export, current-only AGE membership, extraction coverage per source, required KG-phase node/edge fields, independent source/assertion/review states, exact supplied passage IDs, evidence quotes as exact substrings of the referenced source version, no dangling endpoints, deterministic ordering, and relational-versus-AGE current logical ID counts. Assert missing answer/evaluation/architecture artifacts do not fail KG-phase validation.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_retrieval_smoke.py tests/integration/test_exports.py -v`

Expected: missing retrieval/export modules.

- [ ] **Step 4: Implement bounded smoke retrieval**

Use top-eight scoped vector passage hits, up to eight adjacent semantic seeds, at most two explicitly recorded hops, 30 nodes, and 50 edges. Return vector hits, seed IDs, traversed edge IDs, graph paths, and exact original source passages. Do not generate an answer.

- [ ] **Step 5: Implement deterministic atomic exports**

Write `import-report.json`, `extraction-coverage.json`, `graph-validation.json`, `graph/nodes.jsonl`, and `graph/edges.jsonl` through temporary files followed by atomic rename. The history view exports every relational graph version; the current view applies the same active-source/extraction selection as AGE. Sort all records by scoped logical identity and serialize timestamps in UTC ISO 8601.

- [ ] **Step 6: Run tests**

Run: `bin/runner pytest tests/integration/test_retrieval_smoke.py tests/integration/test_exports.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/retrieval.py lib/potency_kg/export.py tests/integration/test_retrieval_smoke.py tests/integration/test_exports.py
git commit -m "feat: export and inspect the generated KG"
```

### Task 10: Execute Isolation, Correction, and Conflict KG Fixtures

**Files:**
- Create: `lib/potency_kg/fixture_runs.py`
- Modify: `fixtures/configs/project-one.json`
- Modify: `fixtures/configs/project-two.json`
- Create: `fixtures/configs/source-correction-before.json`
- Create: `fixtures/configs/source-correction-after.json`
- Create: `fixtures/configs/unresolved-conflict.json`
- Create: `tests/integration/test_behavior_graphs.py`

**Interfaces:**
- Produces: `run_kg_fixtures(engine, fixture_configs, output_root, embedding_client, relevance_client, extraction_client) -> FixtureGraphReport`.
- Consumes: the ordinary import, processing, materialization, refresh, and export interfaces. Fixture identifiers and expected answers never enter extraction prompts or evidence.

- [ ] **Step 1: Write failing fixture-configuration tests**

Require every fixture config to declare `synthetic=true`, scope, snapshot, exact input JSONL paths, and stage. Require correction-after to declare the exact superseded source reference. Reject any fixture configuration that includes expected attributes, expected answer text, questions, or checks as evidence inputs.

- [ ] **Step 2: Write failing fixture graph tests**

Import both isolation projects into the same disposable database and verify containment after a new connection. For source correction, import/refresh/export `before`, then import the explicit replacement and refresh/export `after`; verify v1 history remains relational while only v2 enters the current graph. For unresolved conflict, verify both opposing assertions and source references remain current, no automatic `contradicts` edge exists, and issue date does not deactivate either source.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_behavior_graphs.py -v`

Expected: failure because fixture configurations and orchestration do not exist.

- [ ] **Step 4: Add explicit fixture configurations**

Mark the two existing isolation configs synthetic. Add correction-before, correction-after, and unresolved-conflict configs under `fixtures/configs/`, using only the source inputs and replacement metadata declared by `fixtures/behavior-cases.json`. Keep questions and expected behavior outside importable evidence paths.

- [ ] **Step 5: Implement fixture KG orchestration**

`run_kg_fixtures` executes the normal modules for each checked-in config, never clears another scope, and writes versioned metadata, node exports, edge exports, and validation reports under `results/kg/fixtures/<scenario>/<stage>/`. It performs no answer generation.

- [ ] **Step 6: Run fixture tests**

Run: `bin/runner pytest tests/integration/test_behavior_graphs.py -v`

Expected: all fixture graph tests pass using deterministic model clients where semantic extraction is required.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/fixture_runs.py fixtures/configs tests/integration/test_behavior_graphs.py
git commit -m "feat: exercise KG isolation and history fixtures"
```

### Task 11: Add Thin Notebooks, CLI Commands, and KG Documentation

**Files:**
- Create: `lib/potency_kg/cli.py`
- Create: `notebooks/01_build_knowledge_graph.ipynb`
- Create: `notebooks/02_inspect_knowledge_graph.ipynb`
- Create: `docs/kg-generation.md`
- Create: `tests/integration/test_cli.py`

**Interfaces:**
- Produces CLI commands `preflight`, `import`, `embed`, `classify`, `extract`, `materialize`, `refresh-graph`, `export-kg`, `inspect-kg`, `run-kg`, and `run-kg-fixtures` through `python -m potency_kg.cli`.
- Notebook 01 calls the same stage functions individually and ends with `refresh_graph(...)` plus export.
- Notebook 02 is read-only and calls graph counts, validation, sample Cypher, and retrieval smoke functions.

- [ ] **Step 1: Write failing CLI orchestration tests**

In `tests/integration/test_cli.py`, assert stage order, model-access gating, nonzero exit on validation failure, derived-version output, rejection of arbitrary new version labels, idempotent second execution, semantic refresh refusal after extraction failure, structural refresh availability without models, `--force` refresh behavior, fixture command output paths, absence of credentials in logs/artifacts, and valid notebook JSON containing imports from `potency_kg` rather than function implementations.

- [ ] **Step 2: Run the complete test suite and verify the expected failure**

Run: `bin/runner pytest -v`

Expected: CLI/notebook-specific tests fail because orchestration is missing; earlier tests remain green.

- [ ] **Step 3: Implement CLI dependency injection and stage commands**

Each command accepts `--config`; model-backed commands derive versions and may select an existing version for reads but may not label newly generated work. Refresh accepts `--mode structural|semantic`, `--allow-partial`, and `--force`; export accepts `--output`; fixture execution accepts `--config-dir` containing the checked-in fixture configs and a fixture output root. `run-kg` stops after the first failed stage and prints every created or reused run ID and derived processing/graph version.

- [ ] **Step 4: Create the build notebook**

Add ordered cells for configuration/preflight, engine creation, import, structural graph counts, embedding, relevance, extraction, materialization, `refresh_graph(engine, config.scope, mode="semantic", allow_partial=False, force=False)`, export, and concise summary tables. Read `KG_RUN_MODEL_STAGES`; when it is false or model access is not ready, skip embedding/relevance/extraction/materialization and call structural refresh instead. Keep credentials and full raw model responses out of cell output.

- [ ] **Step 5: Create the inspection notebook**

Add read-only cells for relational counts by type/source, extraction outcomes, unresolved failures, AGE counts, validation report, a parameterized sample Cypher traversal, and vector-seeded retrieval smoke output showing exact source references.

- [ ] **Step 6: Document exact container commands**

In `docs/kg-generation.md`, document image build/start/migration, notebook URLs, `bin/runner` commands, disposable test databases, model-access prerequisites, structural-only operation, full KG stages, deterministic versions, refresh reuse/force behavior, fixture commands and artifact paths, KG-phase contract validation, rerun/idempotency expectations, project isolation, and the trusted root-mount caveat. Link `docs/kg-glossary.md` and state the KG-only exclusions explicitly.

- [ ] **Step 7: Run offline verification**

Run: `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env build`

Run: `bin/start`

Run: `bin/runner alembic upgrade head`

Run: `bin/runner pytest -v`

Expected: images build, migrations apply, and all offline tests pass with no external model calls.

- [ ] **Step 8: Run structural full-corpus verification**

Run:

```bash
bin/runner python -m potency_kg.cli preflight --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli import --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli refresh-graph --config configs/casgevy-potency.json --mode structural
bin/runner python -m potency_kg.cli export-kg --config configs/casgevy-potency.json --output results/kg
```

Expected: 24 sources and 791 passage outcomes reconcile, every valid passage has structural representation, an unchanged re-import reports no duplicates, and graph validation reports no missing endpoints.

- [ ] **Step 9: Run real semantic verification only after access readiness**

Run:

```bash
bin/runner python -m potency_kg.cli run-kg --config configs/casgevy-potency.json --output results/kg
bin/runner env KG_RUN_MODEL_STAGES=true \
  jupyter nbconvert --to notebook --execute --inplace notebooks/01_build_knowledge_graph.ipynb
bin/runner jupyter nbconvert --to notebook --execute --inplace notebooks/02_inspect_knowledge_graph.ipynb
bin/runner python -m potency_kg.cli run-kg-fixtures --config-dir fixtures/configs --output results/kg/fixtures
```

Record actual provider/model IDs, usage, failures, selected/extracted counts, graph counts, and artifact paths in the generated reports and concise notebook summaries. If access is not ready, mark semantic verification `not_run` and execute `bin/runner env KG_RUN_MODEL_STAGES=false jupyter nbconvert --to notebook --execute --inplace notebooks/01_build_knowledge_graph.ipynb` for structural verification; never substitute fake outputs.

- [ ] **Step 10: Commit**

```bash
git add lib/potency_kg/cli.py notebooks/01_build_knowledge_graph.ipynb notebooks/02_inspect_knowledge_graph.ipynb docs/kg-generation.md tests/integration/test_cli.py
git commit -m "feat: add notebook-driven KG workflow"
```

## End-to-End Data Flow

```text
corpus/manifest.json + corpus/raw/*
              │ verify catalog, counts, and hashes
              ▼
      source_document versions
              │
corpus/records.jsonl ── validate exact text/hash ──► source_passage
                                                     │
                                                     └─ structural kg_node/kg_edge for every valid version
                                                               │
                                      complete source only      ▼
                                              source_head ──► scoped passage embedding
              │                                           ▼
              │                         deterministic skip + batched relevance
              │                                           │ selected
              │                                           ▼
              │                        redaction substitution + one-pass extraction
              │                                           │
              │                              mentions/assertions/relations/aliases
              │                                           │
              └────────────────────────► configured alias resolution
                                                          │
                                                          ▼
                                          canonical relational kg_node/kg_edge
                                                          │
                                             refresh_graph(current scope)
                                                          │
                                                          ▼
                                               disposable AGE projection
                                                  │               │
                                                  ├─ export       └─ traversal smoke test
                                                  ▼
                                                results/kg/
```

## Completion Criteria

- Compose exposes the complete repository at `/workspace` and supplies one headless runner for migrations, tests, CLI, and notebook execution.
- All documented executable commands run inside Compose through `bin/runner` or a Compose lifecycle script; host Python is not required.
- Integration tests create, identify, migrate, and drop a disposable database without modifying persistent notebook data.
- Migrations apply from an empty database and enforce composite project isolation for graph endpoints.
- Manifest, records file, raw-source hashes, scope, snapshot, source counts, and passage counts reconcile before import.
- All 24 sources have document representations and all 791 records have truthful import outcomes.
- Incomplete source versions never become active heads.
- Every valid imported source version and passage has relational structural graph representation; complete active passages have scoped embeddings, and retained historical embeddings are not deleted.
- Every passage has a persisted semantic-processing outcome and reason.
- Real model stages cannot run without completed access configuration and recorded embedding/generation verification.
- Selected passages produce validated atomic assertions and assertion-linked relations, or explicit failed outcomes; sanitized invalid responses and validation errors remain inspectable, while partial invalid output is never materialized.
- Every relational assertion has exactly one Assertion or qualifying ReviewConclusion graph node; unsupported conclusion classifications are downgraded with warnings, while invalid evidence quotes fail extraction.
- Source authority/status, assertion regulatory state, and extraction review state remain independent in relational storage, AGE properties, and exports.
- `subject_scope` never creates an evidence relationship by itself; assay roles and `applies_to` relationships require exact passage evidence.
- Supporting quotes resolve to exact original versioned passage text, including redaction-aware span restoration.
- Configured CASGEVY aliases resolve to one product; pending aliases never merge entities.
- Reprocessing unchanged inputs reuses completed embeddings/model runs and does not duplicate graph records.
- Historical source and extraction rows remain immutable while atomically updated active pointers determine the current graph.
- Processing versions are deterministic digests of their actual model, prompt/schema, normalization, and active inputs.
- `refresh_graph()` derives graph versions, reuses unchanged validated graphs, supports explicit forced rebuild, distinguishes partial versions by exact omitted/failed passage set, and validates logical IDs and endpoints before advancing graph state.
- AGE contains no full passage text; retrieval reloads original text from relational PostgreSQL.
- Project-isolation fixtures cannot leak through SQL, cache lookup, vector search, AGE traversal, export, or the retrieval smoke trace.
- Source-correction fixtures keep prior history while the current graph uses only the explicit replacement; conflict fixtures preserve both opposing claims without an invented `contradicts` edge.
- Checked-in configs and `run-kg-fixtures` reproduce isolation, correction-before, correction-after, and unresolved-conflict graphs under versioned `results/kg/fixtures/` paths without generating answers.
- `results/kg/` contains deterministic import, extraction-coverage, graph-validation, node, and edge artifacts validated against the documented KG-phase subset of the supplied output contract.
- `docs/kg-glossary.md` defines the canonical domain terms used by contracts, prompts, notebooks, and exports.
- The two notebooks contain orchestration and concise inspection only; all behavior is exercised through `lib/potency_kg/` interfaces also available to the CLI and tests.
- The root `README.md` and `docs/scratchpad.md` remain byte-for-byte unchanged.
