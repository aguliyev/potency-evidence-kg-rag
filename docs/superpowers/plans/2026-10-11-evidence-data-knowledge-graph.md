# Evidence Data and Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the KG-creation phase: import the supplied evidence without altering it, build a project-isolated and provenance-exact relational knowledge graph, publish a last-known-good Apache AGE projection, and emit the versioned `results/kg/` contract consumed by the later hybrid-RAG phase.

**Architecture:** Application-owned modules under `lib/potency_kg/` translate read-only supplied descriptors into immutable run inputs, keep source/run history in PostgreSQL, and expose deep stage interfaces that own transactions. Every active complete passage is embedded, seven TypeSafe/Jev questions gate an Instructor schema assembled from only the selected extraction bundles, and validated extractions are atomically materialized before a scope-and-version-derived shadow AGE graph is activated. `results/kg/` is an intermediate contract, not the final assignment submission.

**Tech Stack:** Python 3.12, Pydantic 2, Instructor 1.17, OpenAI-compatible generation and embedding endpoints, TypeSafe/Jev 0.7, SQLAlchemy 2, Psycopg 3, Alembic, PostgreSQL 17, pgvector, Apache AGE 1.7, pytest, JupyterLab, and nbconvert.

**Spec:** `ASSIGNMENT.md` sections 1–2 and the graph-retrieval seam of section 3; `MODEL_ACCESS.md`; `corpus/README.md`; `corpus/manifest.json`; `schema/graph-core.json`; `schema/output-contract.json`; `fixtures/README.md`; and `fixtures/BEHAVIOR_CASES.md`.

## Global Constraints

- This is the KG-creation phase. Hybrid retrieval, answer generation, evaluation, architecture/costing, and final submission assembly belong to later plans.
- Treat `results/kg/` as the versioned intermediate contract consumed by the hybrid-RAG phase. Do not emit final top-level answer, evaluation, architecture, or handoff artifacts here.
- Do not optimize this plan around the assignment's eight-hour work window. Prioritize exact provenance, current-version correctness, project isolation, and recoverable state transitions.
- Treat `configs/`, `corpus/`, `fixtures/`, and `schema/` as supplied, immutable, read-only inputs. Do not create, edit, move, reformat, or delete files beneath them.
- Hash every protected file before execution and verify the same path/hash set afterward. The check covers additions, removals, and content changes.
- Read evidence directly from `corpus/` and `fixtures/`; never copy evidence text into application modules, prompts, notebooks, tests, or generated configuration.
- Put all new documentation in `docs/`. Keep root `README.md` unchanged, and do not read or modify `docs/scratchpad.md`.
- Keep application behavior, prompts, Jev bundle definitions, and trusted project defaults beneath `lib/potency_kg/`; notebooks only orchestrate and present library results.
- Use only the existing environment variables `LLM_URI`, `LLM_MODEL`, `LLM_API_KEY`, `EMBEDDING_LLM_URI`, `EMBEDDING_LLM_MODEL`, `EMBEDDING_LLM_API_KEY`, and `TYPESAFE_API_KEY` for model stages.
- For the official KG run, set `EMBEDDING_LLM_URI=https://api.openai.com/v1` and `EMBEDDING_LLM_MODEL=text-embedding-3-large`. Request the model's full `3072` dimensions with `encoding_format="float"`; do not shorten vectors in this phase.
- Treat generation and embedding URIs as OpenAI-compatible `/v1` base endpoints. Internally map generation model `M` to Instructor selector `openai/M`.
- Use Instructor only for structured generation. Use a dedicated OpenAI-compatible embedding adapter for passage and query embeddings. Use TypeSafe/Jev for bundle classification.
- Load each model setting only when its stage runs. Import, structural projection, schema inspection, and structural export must work without model credentials.
- Store model IDs, sanitized endpoint origins, and credential-variable names. Never persist or log API keys, Authorization headers, endpoint userinfo, or endpoint query/fragment credentials.
- Treat `(tenant_id, project_id, source_id, source_version, passage_id)` as source-passage identity. Require tenant and project in every project-owned key, operation, run, cache lookup, graph identifier, trace, and export record.
- Import and account for all 791 normalized official records and all 24 manifest sources. Every valid imported source version gets relational structural representation, including incomplete and superseded versions.
- Only passages in complete active source heads receive semantic processing. Every such passage receives an embedding before deterministic or Jev selection; incomplete and superseded passages remain `not_run` with explicit reasons.
- Source replacement must be explicitly declared by a supplied descriptor. Issue date alone never supersedes a source.
- Preserve immutable embedding runs, Jev responses/selections, LLM extraction runs, validation failures, materialization runs, source versions, and exports.
- Keep exactly one mutable `passage_processing` row per full scoped passage identity. It points to immutable history and always has one state and one machine-readable reason.
- Processing states are `not_run`, `structural_only`, `selected`, `extracted`, and `failed`. Initial reasons include `inactive_source_version`, `incomplete_source_version`, `empty_passage`, `page_number_only`, `jev_no_bundle_above_threshold`, `jev_failed`, `extraction_failed`, and `extracted_successfully`.
- Apply passage-scoped redaction substitution before both Jev and Instructor. Restore only validated spans to untouched original text before persistence or export; never infer concealed `(b)(4)` content.
- Every evidence relationship references one atomic assertion. Exact supporting quotes must resolve uniquely in model text; require an occurrence index when repeated.
- Keep full passage text in relational PostgreSQL. AGE stores only identifiers, locators, compact attributes, and provenance needed for traversal.
- Use parameterized SQL for values. Derive AGE graph names and labels/types only through strict internal validators and allowlists.
- High-level `import_source_version()`, `materialize_extraction()`, and `refresh_graph()` own transactions. Their internal helpers accept a SQLAlchemy `Connection` and never open independent transactions.
- Semantic AGE refresh fails when any selected passage has an unresolved failure. `allow_partial=True` records the exact omissions and produces a distinct graph version.
- Do not accept caller-invented version labels. Derive every version from canonicalized inputs defined below.
- Keep deterministic adapters in the normal test suite. Real Jev, embedding, and generation checks run only when explicitly enabled.
- Do not add LangChain, LlamaIndex, NetworkX, a generic ontology framework, PDF/OCR processing, graph visualization, a review UI, or incremental AGE synchronization.

---

## Accepted Design Decisions

### Runtime settings and project profiles

`settings.py` exposes immutable stage-specific settings. URI normalization accepts a base ending in `/v1` or appends `/v1`, rejects userinfo/query/fragment fields, and records only the sanitized `scheme://host[:port]/v1` origin. Credential values are resolved immediately before constructing a real adapter and are excluded from `repr`, serialization, run metadata, errors, and logs. Instructor's documented custom-provider path accepts the application-mapped key and OpenAI-compatible `base_url`: [Instructor `from_provider`](https://python.useinstructor.com/concepts/from_provider/).

```python
class GenerationSettings(BaseModel, frozen=True):
    base_url: AnyHttpUrl
    model: str
    instructor_model: str              # f"openai/{model}"
    credential_env: Literal["LLM_API_KEY"]
    sanitized_origin: str
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    max_concurrency: int = 4

class EmbeddingSettings(BaseModel, frozen=True):
    base_url: AnyHttpUrl
    model: Literal["text-embedding-3-large"]
    credential_env: Literal["EMBEDDING_LLM_API_KEY"]
    sanitized_origin: str
    dimensions: Literal[3072] = 3072
    encoding_format: Literal["float"] = "float"
    normalization: Literal["provider_unit_length_v1"] = "provider_unit_length_v1"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    batch_size: int = 64
    max_concurrency: int = 4

class JevSettings(BaseModel, frozen=True):
    credential_env: Literal["TYPESAFE_API_KEY"]
    model: str = "jev-latest"
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    max_concurrency: int = 8

def load_generation_settings(env: Mapping[str, str]) -> GenerationSettings: ...
def load_embedding_settings(env: Mapping[str, str]) -> EmbeddingSettings: ...
def load_jev_settings(env: Mapping[str, str]) -> JevSettings: ...
```

`project_profiles.py` owns trusted aliases and other application defaults that cannot be added to protected config files. A `ProjectProfile` has a stable profile name, the supplied descriptor path/hash, resolved aliases, and a canonical `profile_version`. Unknown scopes use an explicit empty profile; fixture behavior never branches on scenario names, question IDs, or expected answers.

### Jev gating and extraction bundles

Every eligible passage receives one Jev request containing exactly seven SDK `NoulQuestion` values. Each `NoulAnswer.noul` response is a probability in `[0, 1]`; all probabilities and request metadata are persisted, matching the [TypeSafe SDK schema](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py).

| Bundle | Allowed extraction |
|---|---|
| `product_context` | Product, Application, Indication, and explicit applicability |
| `alias_identity` | Names, abbreviations, former names, and alias candidates |
| `potency_strategy` | PotencyStrategy and supported context |
| `assay_evidence` | Assay, MeasuredAttribute, assay role, `measures`, and `used_as` |
| `regulatory_history` | RegulatoryInteraction and proposed/requested/revised/accepted/rejected/unresolved chronology |
| `review_conclusion` | Explicit reviewer or agency conclusions and recommendations |
| `redaction_uncertainty` | Passage-scoped redactions, unknowns, and unresolved information |

Default thresholds are `0.50`, except `review_conclusion=0.35` and `redaction_uncertainty=0.35`. Selection is inclusive: `probability >= threshold`. Optional per-bundle overrides replace only their named defaults. Higher thresholds reduce LLM calls and noise but increase false-negative risk; lower thresholds increase recall and LLM calls and may increase cost and noise.

If no bundle is selected, the LLM is not called and the passage becomes `structural_only` with `jev_no_bundle_above_threshold`. A Jev timeout, invalid response, or exhausted retry becomes `failed` with `jev_failed`; there is no unrestricted LLM fallback.

For a selected passage, construct the Instructor response model from exactly the selected bundles. Merge overlapping types by the fixed bundle order shown above, then append universal assertion/provenance fields. The model uses passage-local IDs for entity, assertion, and relationship references. An unselected type triggers one retry with the allowed set emphasized; a second invalid response fails the complete passage. Partial output is never materialized.

### Processing and version identities

Canonical JSON uses sorted object keys, UTF-8, no insignificant whitespace, ordered lists where order is semantic, and SHA-256 with a version-kind prefix. The following identities are separate:

- `embedding_version`: embedding model ID (`text-embedding-3-large` for the official run), dimensions (`3072`), encoding format (`float`), and normalization (`provider_unit_length_v1`).
- `index_version`: `embedding_version` plus the sorted exact membership of active scoped passage identities and text hashes.
- `jev_response_version`: Jev model, seven question definitions, criteria, normalization, canonical redacted input text (hashed into the digest), prompt/bundle hashes, and project-profile name/resolved aliases/source hash/version.
- `jev_selection_version`: `jev_response_version` plus resolved default and per-bundle thresholds.
- `extraction_version`: generation model, Instructor mode, selected bundles, extraction prompt/schema hashes, normalization, and project-profile name/resolved aliases/source hash/version.
- `graph_version`: projection schema, sorted active source heads, sorted active extraction IDs, projection mode, partial policy, sorted omitted/failed passage identities, and project-profile version.

Threshold changes reuse cached Jev responses, create a new Jev selection, and create a new extraction only when the selected bundle set changes. Source correction reuses unchanged vectors by input hash but produces a new membership-aware `index_version`.

### Current state, provenance, and transactions

`passage_processing` is keyed by the complete source-passage identity and has one mutable current row. `active_extraction_run_id` is a nullable pointer, not a multi-row active flag; no contradictory partial unique index is used. State changes and pointer changes occur only inside stage-owner transactions.

Assertions, entity mentions, alias candidates, node source references, and edge source references have composite foreign keys to the complete `source_passage` identity. Deliberate reuse of source, version, and passage IDs across projects must be rejected when scope differs.

```python
def import_source_version(engine: Engine, run: ImportRequest) -> ImportReport: ...
def write_structural_version(
    connection: Connection,
    source: ScopedSourceVersion,
) -> GraphWriteReport: ...
def materialize_extraction(
    engine: Engine,
    passage: ScopedPassageIdentity,
    extraction_run_id: UUID,
) -> MaterializationReport: ...
def refresh_graph(
    engine: Engine,
    scope: Scope,
    mode: Literal["structural", "semantic"] = "semantic",
    allow_partial: bool = False,
    force: bool = False,
) -> GraphRefreshReport: ...
```

Passage import, structural materialization, and source-head activation commit together. Entity resolution, assertion/edge materialization, materialization-run completion, and extraction activation commit together. An injected failure rolls back the whole stage, preserves the previous active extraction/graph, and leaves no pointer to partial data.

### AGE last-known-good activation

Refresh never drops the active graph first. It derives and validates a shadow graph name from scope plus `graph_version`, builds it, validates logical IDs/endpoints/scope/counts and absence of passage text, then atomically updates `graph_scope` to the shadow graph. The former graph is removed only after successful activation. Concurrent readers resolve the active name from `graph_scope` in their transaction, so they see either the previous or new complete graph.

### Aliases and future correction

Trusted aliases come from `project_profiles.py` and carry configuration provenance: profile name, profile version, source path, and source hash. They do not require a passage quote. Extracted alias candidates remain inspectable and inactive, with exact source identity and quote. Approve/reject operations, entity redirects, a review UI, and automatic rematerialization after human review are deferred. A future phase may record an approval as a new profile/review event, resolve a replacement canonical ID, and rematerialize affected extractions without rewriting history.

### Exports and notebook behavior

Every completed graph version gets an immutable `history/<graph-version>/` directory. After validation, atomically replace a relative `current` symlink so `results/kg/current/` resolves to that history directory. Never overwrite a history directory.

```text
results/kg/
  current/
    manifest.json
    import-report.json
    extraction-coverage.json
    graph-validation.json
    graph/
      nodes.jsonl
      edges.jsonl
  history/
    <graph-version>/
      manifest.json
      import-report.json
      extraction-coverage.json
      graph-validation.json
      graph/
        nodes.jsonl
        edges.jsonl
  fixtures/
    <scenario>/
      <stage>/
```

The build notebook orchestrates stages through library interfaces. The inspection notebook is read-only unless an explicit flag enables one retrieval-smoke cell; that cell embeds query text through the passage embedding adapter and writes a sanitized trace. Notebook output never contains credentials or full raw model responses.

## Planned File Structure

```text
requirements.txt
docker/compose.yaml
bin/runner
migrations/versions/
  0002_create_evidence_store.py
  0003_create_knowledge_graph_store.py
lib/potency_kg/
  __init__.py
  settings.py
  project_profiles.py
  fixture_catalog.py
  processing_policy.py
  contracts.py
  ids.py
  redactions.py
  db.py
  ingest.py
  graph_store.py
  embedding_client.py
  embeddings.py
  jev_client.py
  jev_processing.py
  llm_client.py
  extraction.py
  resolution.py
  age_projection.py
  retrieval.py
  export.py
  fixture_runs.py
  cli.py
  prompt_resources/
    __init__.py
    jev_bundles.py
    extraction.py
notebooks/
  01_build_knowledge_graph.ipynb
  02_inspect_knowledge_graph.ipynb
tests/
  conftest.py
  test_runtime.py
  unit/
    test_settings.py
    test_project_profiles.py
    test_fixture_catalog.py
    test_processing_policy.py
    test_contracts.py
    test_ids.py
    test_redactions.py
    test_jev_client.py
    test_extraction.py
  integration/
    test_database_lifecycle.py
    test_migrations.py
    test_import.py
    test_embeddings.py
    test_jev_processing.py
    test_model_processing.py
    test_materialization.py
    test_age_refresh.py
    test_exports.py
    test_retrieval_smoke.py
    test_behavior_graphs.py
    test_cli.py
  real/
    test_real_model_stages.py
docs/
  kg-generation.md
  kg-glossary.md
  kg-extraction-review.md
results/kg/                  # generated, gitignored
```

## Relational Data Model

The two migrations create these project-scoped records and constraints:

- `import_run`, `import_error`, `source_document`, `source_head`, and `source_passage` preserve supplied catalog/input identity and reconciliation.
- `embedding_run` stores model, sanitized origin, credential-variable name, dimensions, normalization, input hash, vector, usage, status, and immutable `embedding_version`.
- `passage_embedding` binds a complete passage identity to an immutable embedding run. `index_snapshot` stores `index_version` and exact sorted membership; `index_member` holds the corresponding passage and embedding IDs.
- `jev_response` stores one passage request, all seven questions/criteria/probabilities, model, request ID, sanitized usage/error, attempts, prompt/profile hashes, input hash, and `jev_response_version`.
- `jev_selection` stores a response ID, resolved thresholds, selected bundles, and `jev_selection_version`.
- `llm_extraction_run` stores selected bundles, generation model, sanitized origin, credential-variable name, Instructor mode, prompt/schema/profile hashes, input hash, status, sanitized usage/error, and validated complete response when successful.
- `validation_failure` stores stage, immutable run ID, machine-readable code, JSON path, sanitized detail, and timestamp.
- `materialization_run` stores extraction run, previous active extraction, status, counts, sanitized error, and timestamps.
- `passage_processing` has the complete passage primary/foreign key, current state, reason, active Jev selection, active extraction run, and updated timestamp. Every state requires a reason; referenced runs must share the same complete passage identity.
- `redaction_mention`, `entity_mention`, `alias_candidate`, and `assertion` retain exact full source identity and original/model spans. `assertion` stores atomic claim, chronology state, graph representation, authority/status, and review state.
- `kg_node`, `kg_edge`, `kg_node_source_ref`, and `kg_edge_source_ref` are canonical relational graph records. Edge endpoints use `(tenant_id, project_id, node_id)` foreign keys; evidence edges require an assertion in the same scope.
- `graph_scope` stores one active AGE graph name/version per scope. `graph_refresh_run` stores shadow/previous names, graph version, mode, partial policy, exact omissions/failures, validation results, status, and timestamps.

Composite foreign keys from `assertion`, `entity_mention`, `alias_candidate`, `kg_node_source_ref`, and `kg_edge_source_ref` target all five `source_passage` identity columns. No table relies on application-only provenance checks.

---

## Execution Stages

Implement the plan as six gated stages. Do not begin a later stage until the current stage's exit gate passes. A stage may contain several tasks because each task remains a reviewable, testable commit; the stage is the larger delivery milestone.

| Stage | Focus | Tasks | Exit artifact or proof |
|---|---|---|---|
| 1 | Database schema and initial database load | 1–4 | A migrated relational store containing all 24 official sources and 791 reconciled passage outcomes, with structural graph records and an idempotent restart/re-import result |
| 2 | Embeddings and index membership | 5 | Immutable scoped embeddings plus a complete, membership-aware index snapshot for every active passage |
| 3 | Semantic classification, extraction, and materialization | 6–8 | Persisted Jev decisions, validated extraction history, and atomically activated relational semantic graph records |
| 4 | AGE projection, export, and graph retrieval seam | 9–10 | A validated last-known-good AGE graph, immutable `results/kg/history/` export, promoted `current`, and a bounded traversal trace |
| 5 | Isolation, correction, and conflict fixtures | 11 | The supplied fixtures run through the ordinary pipeline with preserved stage history and no cross-project leakage |
| 6 | Operator workflow and final KG verification | 12 | Tested CLI/notebooks/docs, a verified structural run, explicitly enabled real-model results when credentials are available, and unchanged protected inputs |

### Stage 1 gate: database schema and initial database load

Stage 1 is deliberately model-free. It includes only the minimum runtime and contract foundations needed to make the database enforce the system's invariants, followed by schema migration and the first full corpus import.

The stage is complete only when all of the following are true:

- Alembic upgrades a clean PostgreSQL database through revisions `0002` and `0003`.
- PostgreSQL, rather than application convention alone, rejects cross-project provenance and dangling graph endpoints.
- The importer creates source versions before passages, activates only complete explicitly current versions, and commits passage rows, structural graph records, processing state, and source-head changes atomically.
- The official load reconciles exactly 24 manifest sources and 791 normalized passage outcomes, with per-source counts and failures or omissions reported separately.
- Re-running the import after closing and recreating the engine produces no duplicate sources, passages, graph records, or changed source heads.
- No embedding, Jev, generation, AGE projection, or model credential is required or invoked.
- `configs/`, `corpus/`, `fixtures/`, and `schema/` retain the same path/hash set.

Run the Stage 1 gate:

```bash
bin/runner alembic upgrade head
bin/runner pytest \
  tests/test_runtime.py \
  tests/unit/test_settings.py \
  tests/unit/test_project_profiles.py \
  tests/unit/test_contracts.py \
  tests/unit/test_ids.py \
  tests/unit/test_redactions.py \
  tests/unit/test_processing_policy.py \
  tests/integration/test_database_lifecycle.py \
  tests/integration/test_migrations.py \
  tests/integration/test_import.py -v
```

Expected: every test passes; the import assertions reconcile `24` sources and `791` passage outcomes; restart/re-import is unchanged; injected failures leave no partial source, head, processing, or structural graph state.

### Stage 1 delivery slices

1. **Foundation:** Task 1 establishes the protected, model-free runtime; Task 2 defines the scoped identities and contracts used by constraints and import code.
2. **Schema:** Task 3 adds the evidence, immutable-run, current-state, and canonical graph tables with database-enforced composite provenance.
3. **Initial load:** Task 4 performs manifest-first import, creates structural nodes/edges in the same transaction, reconciles the official counts, and proves restart idempotency.

## Stage 1: Database Schema and Initial Database Load

### Task 1: Runtime, Settings, and Protected-Input Checks

**Files:**
- Modify: `requirements.txt`
- Modify: `docker/compose.yaml`
- Create: `bin/runner`
- Create: `lib/potency_kg/__init__.py`
- Create: `lib/potency_kg/settings.py`
- Create: `lib/potency_kg/project_profiles.py`
- Create: `tests/conftest.py`
- Create: `tests/test_runtime.py`
- Create: `tests/unit/test_settings.py`
- Create: `tests/unit/test_project_profiles.py`

**Interfaces:**
- Produces the three frozen settings classes and loader functions shown above.
- Produces `protected_input_hashes(root: Path) -> Mapping[str, str]`, `load_project_profile(scope: Scope, descriptor: Path) -> ProjectProfile`, and `ProjectProfile.profile_version`.
- `bin/runner <command...>` runs headless commands in the Compose runner; the script is executable (`100755`).

- [ ] **Step 1: Write failing runtime, settings, and protected-input tests**

Test that structural imports do not access model variables; each real loader names missing variables precisely; both OpenAI-compatible URIs normalize to `/v1`; userinfo/query/fragment credentials are rejected; generation maps to `openai/<model>`; the official embedding settings require `https://api.openai.com/v1`, `text-embedding-3-large`, `3072` dimensions, and `float` encoding; serialized settings omit values of all three keys; profile hashes change when aliases or descriptor bytes change; and protected hashes include every file beneath the four protected roots.

Also assert `stat.S_IMODE(Path("bin/runner").stat().st_mode) & 0o111 != 0` and that Compose mounts `/workspace/configs`, `/workspace/corpus`, `/workspace/fixtures`, and `/workspace/schema` read-only.

- [ ] **Step 2: Run tests and verify failure**

Run: `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env run --rm notebook pytest tests/test_runtime.py tests/unit/test_settings.py tests/unit/test_project_profiles.py -v`

Expected: import failures for `potency_kg.settings` and `potency_kg.project_profiles`, plus missing runner assertions.

- [ ] **Step 3: Implement the runtime and stage-local settings**

Add a direct pinned OpenAI Python dependency because application code imports its embedding client. Mount the repository at `/workspace`, then overlay the four protected roots with `:ro` bind mounts on notebook, migrate, and runner services. Set `PYTHONPATH=/workspace/lib`.

Implement URI validation and loaders with these exact mappings:

```python
GENERATION_ENV = ("LLM_URI", "LLM_MODEL", "LLM_API_KEY")
EMBEDDING_ENV = (
    "EMBEDDING_LLM_URI",
    "EMBEDDING_LLM_MODEL",
    "EMBEDDING_LLM_API_KEY",
)
JEV_ENV = ("TYPESAFE_API_KEY",)
```

For the official run, validate these non-secret values without adding new environment variables:

```dotenv
EMBEDDING_LLM_URI=https://api.openai.com/v1
EMBEDDING_LLM_MODEL=text-embedding-3-large
```

`EmbeddingSettings` supplies `dimensions=3072`, `encoding_format="float"`, and `normalization="provider_unit_length_v1"` as immutable application defaults. Resolve only `EMBEDDING_LLM_API_KEY` from `etc/secrets/.env` when constructing the real embedding adapter.

`load_project_profile` maps official scope `public-fda/casgevy-potency` to profile `casgevy-v1`, resolves `CASGEVY`, `exagamglogene autotemcel`, and `exa-cel` to product `casgevy`, and records the protected descriptor path/hash. All other scopes receive profile `empty-v1` with no aliases.

- [ ] **Step 4: Implement protected-input snapshots and the executable runner**

Hash sorted relative paths plus file bytes under all four roots. The pre/post assertion reports added, removed, and changed relative paths without reading file contents into logs. Create `bin/runner` as a strict Bash wrapper over the Compose runner and set executable mode.

- [ ] **Step 5: Verify runtime invariants**

Run: `bin/runner pytest tests/test_runtime.py tests/unit/test_settings.py tests/unit/test_project_profiles.py -v`

Expected: all tests pass; a write attempt beneath each protected mount fails with a read-only-filesystem error; settings repr/output contain variable names but no credential values.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt docker/compose.yaml bin/runner lib/potency_kg/__init__.py lib/potency_kg/settings.py lib/potency_kg/project_profiles.py tests/conftest.py tests/test_runtime.py tests/unit/test_settings.py tests/unit/test_project_profiles.py
git commit -m "chore: protect KG inputs and define runtime settings"
```

### Task 2: Contracts, IDs, Redaction Mapping, and Processing Policy

**Files:**
- Create: `lib/potency_kg/contracts.py`
- Create: `lib/potency_kg/ids.py`
- Create: `lib/potency_kg/redactions.py`
- Create: `lib/potency_kg/processing_policy.py`
- Create: `lib/potency_kg/prompt_resources/__init__.py`
- Create: `lib/potency_kg/prompt_resources/jev_bundles.py`
- Create: `lib/potency_kg/prompt_resources/extraction.py`
- Create: `docs/kg-glossary.md`
- Create: `tests/unit/test_contracts.py`
- Create: `tests/unit/test_ids.py`
- Create: `tests/unit/test_redactions.py`
- Create: `tests/unit/test_processing_policy.py`

**Interfaces:**
- Produces `Scope`, `ScopedPassageIdentity`, `ScopedSourceVersion`, `SourceRef`, draft/result contracts, graph records, and `ALLOWED_RELATION_ENDPOINTS`.
- Produces scoped identity functions and six version derivation functions matching the accepted version definitions.
- Produces `substitute_redactions(source_ref, text) -> RedactionSubstitution` and `restore_model_span(substitution, start, end) -> OriginalSpan`.
- Produces `JevSelectionPolicy.resolve(overrides) -> ResolvedJevThresholds` and `select_bundles(probabilities, thresholds) -> tuple[BundleName, ...]`.

- [ ] **Step 1: Write failing contract, identity, and version tests**

Cover the full five-part passage identity, deliberate same-local-ID/different-project collisions, deterministic canonical JSON, every version input listed above, graph names matching `^kg_[a-f0-9]{16}_[a-f0-9]{16}$`, exact relation endpoint allowlisting, atomic assertions, passage-local reference integrity, and an assertion reference on every evidence relationship.

- [ ] **Step 2: Write failing redaction and policy tests**

Cover `(b)(4)`, `(b) (4)`, spacing variants, repeated markers, passage-separated tokens, exact offset restoration, ambiguous quote occurrences, and rejection of reconstructed content. Verify thresholds `0.50/0.35/0.35`, inclusive equality, one per-bundle override without mutating defaults, and the fixed seven-bundle order.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_contracts.py tests/unit/test_ids.py tests/unit/test_redactions.py tests/unit/test_processing_policy.py -v`

Expected: missing contract, ID, redaction, and policy modules.

- [ ] **Step 4: Implement contracts, IDs, prompt resources, and versions**

Package the exact seven question definitions, criteria, and bundle-to-type mappings in `jev_bundles.py`. Package universal assertion/provenance instructions, type schemas, retry instruction, and deterministic prompt assembly in `extraction.py`. Hash resource bytes and resolved profile provenance as inputs to the applicable derived versions.

- [ ] **Step 5: Implement reversible redaction and threshold selection**

Store ordered original/model spans for every replacement. Reject zero-match quotes, require `occurrence_index` for repeated exact quotes, and fail any requested span crossing an unmappable token. Return selected bundles in fixed order using `>=`.

- [ ] **Step 6: Define canonical terminology and tradeoffs**

Document Source Version, Active Source Head, Assertion, ReviewConclusion, Processing State, Jev Response, Jev Selection, Extraction Run, Current Graph, and all six Version identities. Put the threshold recall/cost tradeoff beside `JevSelectionPolicy` in both its docstring and the glossary.

- [ ] **Step 7: Verify and commit**

Run: `bin/runner pytest tests/unit/test_contracts.py tests/unit/test_ids.py tests/unit/test_redactions.py tests/unit/test_processing_policy.py -v`

Expected: all focused tests pass.

```bash
git add lib/potency_kg/contracts.py lib/potency_kg/ids.py lib/potency_kg/redactions.py lib/potency_kg/processing_policy.py lib/potency_kg/prompt_resources docs/kg-glossary.md tests/unit
git commit -m "feat: define KG contracts and processing policy"
```

### Task 3: Evidence Schema and Composite Provenance Constraints

**Files:**
- Create: `migrations/versions/0002_create_evidence_store.py`
- Create: `migrations/versions/0003_create_knowledge_graph_store.py`
- Create: `lib/potency_kg/db.py`
- Create: `tests/integration/test_database_lifecycle.py`
- Create: `tests/integration/test_migrations.py`

**Interfaces:**
- Produces `create_engine_from_env() -> Engine`, `connection_scope(engine) -> Iterator[Connection]`, and the relational model above.
- Internal write helpers accept `Connection`; only named high-level stage functions receive `Engine` and own transactions.

- [ ] **Step 1: Write failing database-factory lifecycle and migration tests**

Create one database per test factory, migrate it, close all test connections, and assert deletion from the factory after fixture teardown. Do not assert database deletion inside the yielding session fixture.

Assert every table/enum/check exists; every processing state requires a nonempty reason; `active_extraction_run_id` is a single nullable pointer; and no active-pointer partial unique index exists.

- [ ] **Step 2: Write failing provenance-isolation tests**

Insert same `source_id/source_version/passage_id` values in two projects. Verify the database rejects cross-project assertions, entity mentions, alias candidates, node source refs, edge source refs, edge endpoints, active selections, and active extractions. Also reject nonexistent passage provenance.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_database_lifecycle.py tests/integration/test_migrations.py -v`

Expected: migrations `0002` and `0003` are absent and schema assertions fail.

- [ ] **Step 4: Implement evidence and immutable-run tables**

Use PostgreSQL `JSONB`, timezone-aware timestamps, explicit status checks, complete composite keys, and named composite foreign keys. Persist sanitized origins and credential-variable names, never credential values. Give immutable history tables insert-only repository functions.

- [ ] **Step 5: Implement canonical graph and current-pointer tables**

Add scoped node/edge endpoint constraints, assertion linkage, normalized source-reference tables, `graph_scope`, and refresh history. Make `passage_processing` the sole current semantic-state row for a passage.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner alembic upgrade head`

Run: `bin/runner pytest tests/integration/test_database_lifecycle.py tests/integration/test_migrations.py -v`

Expected: clean migration and all database-enforced isolation tests pass.

```bash
git add migrations/versions/0002_create_evidence_store.py migrations/versions/0003_create_knowledge_graph_store.py lib/potency_kg/db.py tests/integration/test_database_lifecycle.py tests/integration/test_migrations.py
git commit -m "feat: enforce scoped evidence provenance"
```

### Task 4: Manifest-First Import and Structural Graph

**Files:**
- Create: `lib/potency_kg/ingest.py`
- Create: `lib/potency_kg/graph_store.py`
- Create: `tests/integration/test_import.py`

**Interfaces:**

```python
def preflight_official_inputs(descriptor: Path) -> CorpusPreflightReport: ...
def import_source_version(engine: Engine, run: ImportRequest) -> ImportReport: ...
def write_structural_version(
    connection: Connection,
    source: ScopedSourceVersion,
) -> GraphWriteReport: ...
def load_current_projection(
    connection: Connection,
    scope: Scope,
    mode: ProjectionMode,
    allow_partial: bool,
) -> ProjectionSnapshot: ...
```

- [ ] **Step 1: Write failing preflight/import tests**

For the official corpus, verify protected descriptor/manifest/records/raw hashes, 24 manifest documents, 791 passage outcomes, per-source counts, passage text hashes, and path containment. Assert a document row exists even if one passage is invalid.

- [ ] **Step 2: Write failing activation, transaction, and restart tests**

Import the same data twice, close the engine, create a new engine, import again, and assert zero duplicate rows and zero changed heads. Inject failure after passage insert, after structural writes, and before head activation; each failure must roll back the whole source version. Verify incomplete versions never activate and replacements activate only when explicitly declared.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_import.py -v`

Expected: missing ingestion and graph-store interfaces.

- [ ] **Step 4: Implement manifest-first import without protected writes**

Read `configs/casgevy-potency.json`, `corpus/manifest.json`, raw files, and `corpus/records.jsonl` in place. Create document versions before streaming passages. For every valid version, call `write_structural_version(connection, source)` inside the import transaction. Activate a head only when expected counts/hashes reconcile; explicit replacement metadata, never dates, determines supersession.

- [ ] **Step 5: Persist truthful non-active processing state**

Create/update one `passage_processing` row for every valid passage. Use `not_run/incomplete_source_version` for incomplete versions and `not_run/inactive_source_version` for complete non-head versions. When a replacement activates, preserve old immutable runs and change its current processing reason to `inactive_source_version`.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/integration/test_import.py -v`

Expected: all official counts reconcile, reconnect re-import is unchanged, injected failures leave no partial source/head/structural state.

```bash
git add lib/potency_kg/ingest.py lib/potency_kg/graph_store.py tests/integration/test_import.py
git commit -m "feat: import protected evidence atomically"
```

## Stage 2: Embeddings and Membership-Aware Index

**Entry condition:** Stage 1 passed against a clean migrated database and the official import is reconciled.

**Exit gate:** Task 5 tests pass; every complete active passage has exactly one usable embedding binding for the selected embedding version; unchanged reruns make zero adapter calls; and explicit source replacement produces the correct new membership-aware index without losing old history.

### Task 5: Embedding Cache and Membership-Aware Index

**Files:**
- Create: `lib/potency_kg/embedding_client.py`
- Create: `lib/potency_kg/embeddings.py`
- Create: `tests/integration/test_embeddings.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class EmbeddingClientIdentity:
    provider: str
    model: str
    dimensions: int
    encoding_format: Literal["float"]
    normalization: str
    sanitized_origin: str
    credential_env: str

@dataclass(frozen=True)
class EmbeddingBatch:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    encoding_format: Literal["float"]
    normalization: str
    prompt_tokens: int
    total_tokens: int

class EmbeddingInputError(ValueError): ...
class EmbeddingProtocolError(RuntimeError): ...

class EmbeddingClient(Protocol):
    identity: EmbeddingClientIdentity
    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...

class OpenAIEmbeddingClient:
    def __init__(
        self,
        settings: EmbeddingSettings,
        api_key: SecretStr,
        client: OpenAI | None = None,
    ) -> None: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...

def embed_active_passages(
    engine: Engine,
    scope: Scope,
    client: EmbeddingClient,
) -> EmbeddingReport: ...

def embed_query(client: EmbeddingClient, text: str) -> QueryEmbedding: ...
def vector_search(
    connection: Connection,
    scope: Scope,
    index_version: str,
    query_vector: Sequence[float],
    limit: int,
) -> list[PassageHit]: ...
```

- [ ] **Step 1: Write failing adapter/cache/index tests**

Verify the real adapter constructs `openai.OpenAI(base_url="https://api.openai.com/v1", api_key=<resolved value>, timeout=60.0)` and calls `.embeddings.create` with exactly:

```python
{
    "model": "text-embedding-3-large",
    "input": list(texts),
    "encoding_format": "float",
    "dimensions": 3072,
}
```

Assert Instructor is never imported or called by the embedding adapter. Reject an empty batch, empty/whitespace input, a response count different from the input count, duplicate/missing/out-of-range response indexes, non-finite values, and vectors whose length is not exactly `3072`. Return vectors in input order by sorting and validating each response item's `index`, even if the test double returns response items out of order. Capture `response.model`, `usage.prompt_tokens`, and `usage.total_tokens` without retaining the SDK response object.

Verify batches stop at `64` inputs, query embedding calls the same adapter with one input, adapter `repr` and exceptions omit the key, metadata contains only `text-embedding-3-large`, `3072`, `float`, `provider_unit_length_v1`, the sanitized origin, and `EMBEDDING_LLM_API_KEY`, and structural workflows never load embedding settings.

Verify every active complete passage is embedded before skip/gating; unchanged inputs produce zero repeated calls; same text in another project does not share project cache entries; superseded vectors remain; correction reuses unchanged vectors; and exact active membership changes `index_version`.

- [ ] **Step 2: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_embeddings.py -v`

Expected: missing embedding adapter and index interfaces.

- [ ] **Step 3: Implement the dedicated OpenAI-compatible adapter**

Resolve `EMBEDDING_LLM_API_KEY` only during adapter construction. Implement the adapter body as the following algorithm; helper names may be private, but its validation and returned fields are part of the interface:

```python
class OpenAIEmbeddingClient:
    def __init__(
        self,
        settings: EmbeddingSettings,
        api_key: SecretStr,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            base_url=str(settings.base_url),
            api_key=api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
        )
        self.identity = EmbeddingClientIdentity(
            provider="openai-compatible",
            model=settings.model,
            dimensions=settings.dimensions,
            encoding_format=settings.encoding_format,
            normalization=settings.normalization,
            sanitized_origin=settings.sanitized_origin,
            credential_env=settings.credential_env,
        )

    def __repr__(self) -> str:
        return f"OpenAIEmbeddingClient(identity={self.identity!r})"

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        inputs = list(texts)
        if not inputs or any(not text.strip() for text in inputs):
            raise EmbeddingInputError("embedding inputs must be non-empty text")
        if len(inputs) > self._settings.batch_size:
            raise EmbeddingInputError("embedding batch exceeds configured size 64")

        response = self._client.embeddings.create(
            model=self._settings.model,
            input=inputs,
            encoding_format=self._settings.encoding_format,
            dimensions=self._settings.dimensions,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        indexes = [item.index for item in ordered]
        if indexes != list(range(len(inputs))):
            raise EmbeddingProtocolError("embedding response indexes do not match inputs")

        vectors = tuple(tuple(float(value) for value in item.embedding) for item in ordered)
        if any(len(vector) != self._settings.dimensions for vector in vectors):
            raise EmbeddingProtocolError("embedding response dimension is not 3072")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise EmbeddingProtocolError("embedding response contains non-finite values")

        return EmbeddingBatch(
            model=response.model,
            vectors=vectors,
            dimensions=self._settings.dimensions,
            encoding_format=self._settings.encoding_format,
            normalization=self._settings.normalization,
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )
```

The official API accepts batched string inputs, `encoding_format="float"`, and an explicit `dimensions` value for `text-embedding-3` models: [OpenAI embeddings API reference](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create). `text-embedding-3-large` uses `3072` dimensions when not shortened: [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings). Store model ID, dimensions, encoding, normalization, sanitized origin, and credential-variable name on each run. Deterministic test adapters use identity `test-double` and never set real-run status.

- [ ] **Step 4: Implement scoped immutable caching and exact search**

Key vector reuse by scope, `embedding_version`, and normalized input hash. Build `index_snapshot/index_member` from the sorted complete active membership after all passage embeddings succeed. For this corpus size, filter by exact scope/index and use cosine ordering without an ANN index.

- [ ] **Step 5: Verify and commit**

Run: `bin/runner pytest tests/integration/test_embeddings.py -v`

Expected: all tests pass, including zero adapter calls on unchanged rerun and a new index version after explicit replacement.

```bash
git add lib/potency_kg/embedding_client.py lib/potency_kg/embeddings.py tests/integration/test_embeddings.py
git commit -m "feat: cache scoped embeddings and index membership"
```

## Stage 3: Semantic Classification, Extraction, and Materialization

**Entry condition:** Stage 2 produced a complete index snapshot for the active scoped passages.

**Exit gate:** Tasks 6–8 tests pass; every eligible passage has a truthful current processing state; Jev and extraction history is immutable; invalid or partial responses never materialize; and a successful materialization atomically switches the passage's active extraction pointer together with its assertions, entities, edges, and source references.

### Task 6: Jev Bundle Classification and Selection

**Files:**
- Create: `lib/potency_kg/jev_client.py`
- Create: `lib/potency_kg/jev_processing.py`
- Create: `tests/unit/test_jev_client.py`
- Create: `tests/integration/test_jev_processing.py`

**Interfaces:**

```python
class JevClient(Protocol):
    identity: JevClientIdentity
    def classify(
        self,
        text: str,
        questions: Sequence[NoulQuestion],
    ) -> JevResponse: ...

def classify_active_passages(
    engine: Engine,
    scope: Scope,
    client: JevClient,
    policy: JevSelectionPolicy,
) -> JevProcessingReport: ...
```

- [ ] **Step 1: Write failing one-request/seven-question tests**

Assert exactly one Jev call per eligible passage and exactly seven uniquely named `Noul` questions in the fixed bundle order. Validate every probability is numeric and in `[0, 1]`. Verify bounded concurrency across passages, never batching multiple passages into one request.

- [ ] **Step 2: Write failing persistence, retry, and cache tests**

Assert persistence of all probabilities, model, request ID, usage, question text, criteria, thresholds, selected bundles, input hash, and both Jev versions. Test inclusive threshold boundaries, defaults, per-bundle overrides, threshold-only reuse of response probabilities, changed selection versions, and new extraction eligibility only when selected bundles change.

Verify no LLM call is scheduled when none is selected. A timeout, invalid response, or exhausted retry must set `failed/jev_failed`; it must not create a selection or invoke Instructor. Empty and page-number-only passages become `structural_only` after embedding and before Jev.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_jev_client.py tests/integration/test_jev_processing.py -v`

Expected: missing Jev client and processing modules.

- [ ] **Step 4: Implement the TypeSafe/Jev adapter and processing owner**

Resolve `TYPESAFE_API_KEY` only during real adapter construction. Apply redaction substitution, call all seven questions in one request, retry only declared transient failures up to `max_attempts`, and sanitize errors. `classify_active_passages` owns persistence transactions per completed passage and a bounded worker pool.

- [ ] **Step 5: Implement immutable response reuse and mutable current state**

Reuse by `jev_response_version`; calculate selection independently by `jev_selection_version`. Persist all response data before selection. Set current state to `selected` with reason `jev_bundle_selected`, `structural_only/jev_no_bundle_above_threshold`, or `failed/jev_failed`.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/unit/test_jev_client.py tests/integration/test_jev_processing.py -v`

Expected: all Jev policy, reuse, failure, and no-fallback tests pass.

```bash
git add lib/potency_kg/jev_client.py lib/potency_kg/jev_processing.py tests/unit/test_jev_client.py tests/integration/test_jev_processing.py
git commit -m "feat: gate KG extraction with Jev bundles"
```

### Task 7: Instructor Extraction and Complete-Response Validation

**Files:**
- Create: `lib/potency_kg/llm_client.py`
- Create: `lib/potency_kg/extraction.py`
- Create: `tests/unit/test_extraction.py`
- Create: `tests/integration/test_model_processing.py`

**Interfaces:**

```python
class StructuredExtractionClient(Protocol):
    identity: GenerationClientIdentity
    def extract(
        self,
        model_text: str,
        response_model: type[BaseModel],
        prompt: str,
    ) -> BaseModel: ...

def extract_selected_passages(
    engine: Engine,
    scope: Scope,
    client: StructuredExtractionClient,
    max_concurrency: int,
) -> ExtractionReport: ...
```

- [ ] **Step 1: Write failing dynamic-schema tests**

For every bundle and overlap combination, assert the response model contains exactly selected bundle types plus universal assertion/provenance fields. Verify deterministic type merging, passage-local IDs, relation-to-assertion linkage, and rejection of unselected types.

- [ ] **Step 2: Write failing grounding and atomicity tests**

Cover atomic assertions, allowed endpoint matrix, exact unique quotes, occurrence-index disambiguation, redaction restoration to untouched original text, no concealed-value inference, and complete-response atomicity. An unselected type triggers exactly one emphasized retry; a second invalid response stores one failed run and no mentions/assertions/relations.

- [ ] **Step 3: Write failing client/settings/version tests**

Assert Instructor is built with `instructor.from_provider("openai/<model>", base_url=<normalized /v1>, api_key=<resolved key>)`; custom environment variables are explicitly mapped rather than assumed by Instructor. Persist model, sanitized origin, credential-variable name, Instructor mode, selected bundles, prompt/schema/profile hashes, usage, and `extraction_version`, with no secrets or endpoint query credentials.

- [ ] **Step 4: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_extraction.py tests/integration/test_model_processing.py -v`

Expected: missing structured generation adapter and extraction owner.

- [ ] **Step 5: Implement schema construction, extraction, and validation**

Build the dynamic Pydantic response class from `prompt_resources`. Call one selected passage per Instructor request with bounded concurrency. Validate the whole response, resolve every local reference, map quotes from redacted model text to original offsets/text, and persist a successful immutable extraction run only after all checks pass.

- [ ] **Step 6: Implement retry/failure semantics and verify**

Retry once only for an otherwise parseable response containing unselected types; emphasize the exact allowed type set. Other transient transport retries follow client policy. Validation failure stores sanitized response metadata plus `validation_failure` rows and sets `failed/extraction_failed` without partial materialization.

Run: `bin/runner pytest tests/unit/test_extraction.py tests/integration/test_model_processing.py -v`

Expected: all focused tests pass with deterministic adapters.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/llm_client.py lib/potency_kg/extraction.py tests/unit/test_extraction.py tests/integration/test_model_processing.py
git commit -m "feat: extract bundle-scoped grounded claims"
```

### Task 8: Resolution and Atomic Materialization

**Files:**
- Create: `lib/potency_kg/resolution.py`
- Create: `tests/integration/test_materialization.py`

**Interfaces:**

```python
def bootstrap_profile_aliases(
    connection: Connection,
    profile: ProjectProfile,
) -> AliasBootstrapReport: ...

def resolve_mention(
    connection: Connection,
    scope: Scope,
    mention: EntityMentionDraft,
    profile: ProjectProfile,
) -> ResolvedEntity: ...

def materialize_extraction(
    engine: Engine,
    passage: ScopedPassageIdentity,
    extraction_run_id: UUID,
) -> MaterializationReport: ...
```

- [ ] **Step 1: Write failing alias and resolution tests**

Verify the three official trusted aliases resolve to product `casgevy` with profile provenance and no passage quote. Extracted aliases remain inactive candidates with complete source provenance and exact quotes. Similarity alone never merges entities. Confirm no approve/reject functions or entity redirect implementation exists.

- [ ] **Step 2: Write failing transaction and provenance tests**

Verify entity resolution, assertions, claim nodes, evidence edges, source refs, materialization completion, `active_extraction_run_id`, and `extracted/extracted_successfully` commit together. Inject failures after each write group; every case preserves the previous active extraction and leaves the failed run inactive. Reject cross-project IDs even when all local IDs collide.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_materialization.py -v`

Expected: missing resolution/materialization interfaces.

- [ ] **Step 4: Implement trusted-profile and extracted-candidate handling**

Bootstrap Product/Application anchor nodes and trusted aliases from the immutable `ProjectProfile`. Persist extracted candidates separately with proposer, extraction version, exact full source identity, original quote, and pending status. Never use candidates during current resolution.

- [ ] **Step 5: Implement one atomic materialization transaction**

Resolve mentions first, then create every atomic assertion, exactly one Assertion or qualifying ReviewConclusion node, assertion-linked evidence relationships, and normalized source refs. Update the materialization run and current pointer last, in the same transaction.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/integration/test_materialization.py -v`

Expected: all rollback, pointer, source-reference, and collision tests pass.

```bash
git add lib/potency_kg/resolution.py tests/integration/test_materialization.py
git commit -m "feat: materialize semantic graph atomically"
```

## Stage 4: AGE Projection, Export, and Retrieval Seam

**Entry condition:** Stage 3 has a complete relational semantic snapshot, or an explicitly recorded partial policy is selected.

**Exit gate:** Tasks 9–10 tests pass; refresh failure preserves the previous active AGE graph; successful refresh validates before activation; immutable export history is complete before `current` promotion; and graph retrieval reloads exact source text from the scoped relational store while recording seeds, traversed edges, limits, and versions.

### Task 9: Shadow AGE Projection and Activation

**Files:**
- Create: `lib/potency_kg/age_projection.py`
- Create: `tests/integration/test_age_refresh.py`

**Interfaces:**
- Produces `refresh_graph(...)`, `validate_graph(connection, graph_name, snapshot) -> GraphValidationReport`, `active_graph_name(connection, scope) -> str`, and `graph_counts(connection, scope) -> GraphCounts`.
- Consumes only `load_current_projection(connection, ...)`, the projection type/label allowlist, strict name validators, and derived `graph_version`.

- [ ] **Step 1: Write failing shadow-build tests**

Assert strict graph names, unique scope/version shadow names, allowlisted labels/types, logical IDs as properties, no full passage text, no dangling endpoints, exact scope/count validation, deterministic graph versions, and unchanged validated reuse.

- [ ] **Step 2: Write failing last-known-good tests**

Inject failures during vertex creation, edge creation, validation, and activation. Verify the previous `graph_scope` pointer and former AGE graph remain active. Verify a successful activation switches atomically and only then removes the prior graph. Run a concurrent reader loop during refresh and assert every observed snapshot is wholly old or wholly new.

Test semantic refusal for unresolved selected/failed passages, exact omission recording with `allow_partial=True`, and a distinct partial graph version.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_age_refresh.py -v`

Expected: missing AGE projection interface.

- [ ] **Step 4: Implement shadow construction and validation**

Within `refresh_graph`'s transaction, create a new scope/version-derived graph, project current structural and semantic records, validate logical ID sets, endpoints, scope, counts, and forbidden text properties, then update `graph_scope`. Record every attempt in immutable refresh history.

- [ ] **Step 5: Implement safe cleanup and reader resolution**

Resolve the active graph through `graph_scope` at reader transaction start. Drop only the exact previous validated graph after activation; retain a failed shadow name in the failure record and clean that shadow through strict validation. `force=True` rebuilds the same derived version under a unique attempt suffix before swapping.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/integration/test_age_refresh.py -v`

Expected: all failure injection, concurrent reader, isolation, and partial-policy tests pass.

```bash
git add lib/potency_kg/age_projection.py tests/integration/test_age_refresh.py
git commit -m "feat: activate last-known-good AGE projections"
```

### Task 10: Export and Vector-Seeded Traversal

**Files:**
- Create: `lib/potency_kg/export.py`
- Create: `lib/potency_kg/retrieval.py`
- Create: `tests/integration/test_exports.py`
- Create: `tests/integration/test_retrieval_smoke.py`

**Interfaces:**

```python
def export_graph_version(
    engine: Engine,
    scope: Scope,
    output_root: Path,
    graph_version: str,
) -> ExportReport: ...

def validate_kg_artifacts(version_dir: Path) -> ExportValidationReport: ...
def promote_current(output_root: Path, graph_version: str) -> Path: ...

def retrieve_graph_smoke(
    engine: Engine,
    scope: Scope,
    query_text: str,
    embedding_client: EmbeddingClient,
    index_version: str,
    limits: RetrievalLimits,
) -> RetrievalTrace: ...
```

- [ ] **Step 1: Write failing current/history export tests**

Assert exact layout, immutable version directories, deterministic ordering, source-versioned exact quotes, no dangling endpoints, relational/AGE count agreement, and a `manifest.json` containing scope, graph/index/embedding/Jev/extraction versions, active heads, profile provenance, hashes, statuses, and artifact hashes. Structural-only manifests use explicit `not_run` statuses and nullable semantic/index versions.

Verify `current` promotion is atomic and points to a complete history version. An interrupted export or failed validation must leave the prior current target unchanged.

- [ ] **Step 2: Write failing query-embedding and trace tests**

Assert query text uses `embed_query` on the same adapter as passages. Persist query hash, embedding/index versions, ordered vector hits, seed node IDs, traversed edge IDs, hop/node/edge limits, graph version, and exact selected source references. Ensure selected original text is reloaded relationally and no other project appears.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/integration/test_exports.py tests/integration/test_retrieval_smoke.py -v`

Expected: missing export and retrieval interfaces.

- [ ] **Step 4: Implement immutable export and atomic current promotion**

Write a staging directory, validate it, fsync files/directories, rename it to `history/<graph-version>`, then atomically replace a relative `current` symlink. Refuse an existing history destination even if contents appear equal. Export only the KG-phase subset of the supplied output contract and document later-phase exclusions in the manifest.

- [ ] **Step 5: Implement bounded vector-seeded traversal**

Use top eight scoped vector passages, up to eight adjacent semantic seeds, two recorded hops, 30 nodes, and 50 edges. Return trace data and exact source refs only; do not generate an answer.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/integration/test_exports.py tests/integration/test_retrieval_smoke.py -v`

Expected: all history, promotion, sanitization, embedding, traversal, and isolation tests pass.

```bash
git add lib/potency_kg/export.py lib/potency_kg/retrieval.py tests/integration/test_exports.py tests/integration/test_retrieval_smoke.py
git commit -m "feat: publish versioned KG artifacts"
```

## Stage 5: Isolation, Correction, and Conflict Fixtures

**Entry condition:** The ordinary official-data pipeline can import, embed, classify, extract, materialize, project, retrieve, and export without fixture-specific branches.

**Exit gate:** Task 11 tests pass; same local identifiers remain isolated across projects; explicit correction replaces only the current source head while preserving v1 history; equal-authority conflicting notes remain simultaneously current; and all generated fixture artifacts stay beneath `results/kg/fixtures/`.

### Task 11: Read-Only Fixture Orchestration

**Files:**
- Create: `lib/potency_kg/fixture_catalog.py`
- Create: `lib/potency_kg/fixture_runs.py`
- Create: `tests/unit/test_fixture_catalog.py`
- Create: `tests/integration/test_behavior_graphs.py`

**Interfaces:**

```python
def load_fixture_catalog(package_root: Path) -> FixtureCatalog: ...
def resolve_fixture_run(
    catalog: FixtureCatalog,
    scenario_id: str,
    stage: str,
) -> FixtureRunConfiguration: ...
def run_kg_fixtures(
    engine: Engine,
    catalog: FixtureCatalog,
    output_root: Path,
    clients: ModelClients,
) -> FixtureGraphReport: ...
```

- [ ] **Step 1: Write failing descriptor-translation tests**

Parse existing `fixtures/configs/*.json`, `fixtures/isolation-cases.json`, and `fixtures/behavior-cases.json` without modifying them. Resolve config-relative paths relative to the descriptor file and behavior-case `input_records` relative to package root through one conversion function. Reject escapes, evidence paths outside named JSONL inputs, and questions/checks/expected answers as evidence.

- [ ] **Step 2: Write failing fixture behavior tests**

Load both isolation projects into one database, reconnect/restart, and verify collision containment. Run correction before/after using the supplied explicit replacement; preserve v1 history and use only v2 in current membership. Run conflict with both current sources; preserve both assertions and never invent `contradicts`. Confirm dates alone do not supersede.

Assert all generated metadata/results are beneath `results/kg/fixtures/<scenario>/<stage>/`, and protected hashes are unchanged after the suite.

- [ ] **Step 3: Run tests and verify failure**

Run: `bin/runner pytest tests/unit/test_fixture_catalog.py tests/integration/test_behavior_graphs.py -v`

Expected: missing fixture catalog/orchestration modules.

- [ ] **Step 4: Implement one fixture conversion layer**

Translate supplied descriptors into validated in-memory `FixtureRunConfiguration` values containing scope, snapshot, input paths, explicit replacements, and output stage. Source replacement remains a generic import operation; no code branches on fixture name, scenario name, marker, or question ID.

- [ ] **Step 5: Implement ordinary-pipeline fixture execution**

Call the same import, embedding, Jev, extraction, materialization, refresh, retrieval-smoke, and export interfaces as the official run. Keep deterministic adapters in offline tests and clearly label them `test-double`.

- [ ] **Step 6: Verify and commit**

Run: `bin/runner pytest tests/unit/test_fixture_catalog.py tests/integration/test_behavior_graphs.py -v`

Expected: isolation, correction, conflict, restart, path-resolution, and protected-hash checks pass.

```bash
git add lib/potency_kg/fixture_catalog.py lib/potency_kg/fixture_runs.py tests/unit/test_fixture_catalog.py tests/integration/test_behavior_graphs.py
git commit -m "feat: orchestrate supplied fixtures read-only"
```

## Stage 6: Operator Workflow and Final KG Verification

**Entry condition:** Stages 1–5 pass through their library interfaces.

**Exit gate:** Task 12 passes the offline suite, structural full-corpus run, protected-input and secret scans, and artifact validation. Real embedding/Jev/generation runs are performed only when explicitly enabled and valid credentials are available; otherwise their status is reported as `not_run`, never as a successful production result.

### Task 12: CLI, Notebooks, Documentation, and Real-Run Verification

**Files:**
- Create: `lib/potency_kg/cli.py`
- Create: `notebooks/01_build_knowledge_graph.ipynb`
- Create: `notebooks/02_inspect_knowledge_graph.ipynb`
- Create: `docs/kg-generation.md`
- Create: `docs/kg-extraction-review.md`
- Create: `tests/integration/test_cli.py`
- Create: `tests/real/test_real_model_stages.py`

**Interfaces:**
- Produces CLI commands `hash-inputs`, `preflight`, `import`, `embed`, `classify-jev`, `extract`, `materialize`, `refresh-graph`, `export-kg`, `inspect-kg`, `run-kg`, and `run-kg-fixtures` through `python -m potency_kg.cli`.
- Notebook 01 calls the same stage interfaces and explicitly refreshes/exports. Notebook 02 performs read-only inspection unless `KG_ENABLE_RETRIEVAL_SMOKE=true` enables its query-embedding/trace cell.

- [ ] **Step 1: Write failing CLI and notebook tests**

Assert stage order, lazy credential loading, nonzero exit on stage failure, derived-version output, rejection of version-label arguments, idempotent rerun, semantic refresh refusal on unresolved failure, structural operation without model variables, exact fixture output roots, protected pre/post hashes, executable `bin/runner`, and sanitized output.

Parse both notebooks as JSON. Require imports from `potency_kg`, no function/class implementations, no credentials, no full raw model responses, read-only inspection by default, and the explicit smoke flag gate.

- [ ] **Step 2: Run the suite and verify the expected failure**

Run: `bin/runner pytest -v -m "not real_models"`

Expected: CLI/notebook/documentation assertions fail while Tasks 1–11 remain green.

- [ ] **Step 3: Implement CLI orchestration and notebook behavior**

`run-kg` takes protected descriptor paths and an output root, snapshots protected hashes, executes stages, exports/promotes only after validation, verifies protected hashes in `finally`, and prints run IDs/derived versions. Structural commands never construct real model adapters. The inspection notebook writes a retrieval trace only when the flag is true.

- [ ] **Step 4: Document exact operation and future alias correction**

In `docs/kg-generation.md`, document Compose build/start/migrate, protected mounts/hashes, settings variables, structural-only operation, all semantic stages, thresholds and tradeoffs, transaction guarantees, version reuse, last-known-good graph behavior, current/history exports, fixture commands, sanitization, rerun expectations, and the handoff from `results/kg/` to the later hybrid-RAG plan.

In `docs/kg-extraction-review.md`, define a question-independent review form sampling multiple source types. Record source identity, selected bundles/probabilities, extracted claims/relations, quote accuracy, omissions, unsupported additions, and reviewer notes. State the deferred alias approval/redirect/rematerialization path without implementing it.

- [ ] **Step 5: Run offline verification**

Run:

```bash
docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env build
bin/start
bin/runner alembic upgrade head
bin/runner pytest -v -m "not real_models"
```

Expected: images build, migrations apply, all deterministic tests pass, no external model call occurs, and protected hashes match.

- [ ] **Step 6: Run structural full-corpus verification**

Run:

```bash
bin/runner python -m potency_kg.cli hash-inputs --output /tmp/kg-protected-before.json
bin/runner python -m potency_kg.cli preflight --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli import --config configs/casgevy-potency.json
bin/runner python -m potency_kg.cli refresh-graph --config configs/casgevy-potency.json --mode structural
bin/runner python -m potency_kg.cli export-kg --config configs/casgevy-potency.json --output results/kg
bin/runner python -m potency_kg.cli hash-inputs --verify /tmp/kg-protected-before.json
```

Expected: 24 sources and 791 passage outcomes reconcile; all valid passages have structural representation; semantic/index fields are null with `not_run`; reconnect re-import is unchanged; graph validation passes; protected inputs are identical.

- [ ] **Step 7: Run explicitly enabled real adapters**

Run:

```bash
bin/runner env KG_RUN_REAL_MODELS=true pytest tests/real/test_real_model_stages.py -v -m real_models
bin/runner env KG_RUN_REAL_MODELS=true python -m potency_kg.cli run-kg --config configs/casgevy-potency.json --output results/kg
bin/runner env KG_RUN_REAL_MODELS=true KG_ENABLE_RETRIEVAL_SMOKE=false jupyter nbconvert --to notebook --execute --inplace notebooks/01_build_knowledge_graph.ipynb
bin/runner env KG_ENABLE_RETRIEVAL_SMOKE=false jupyter nbconvert --to notebook --execute --inplace notebooks/02_inspect_knowledge_graph.ipynb
```

Expected: the embedding report records requested and returned model `text-embedding-3-large`, dimension `3072`, float encoding, token usage, sanitized OpenAI origin, and credential-variable name; the remaining reports record generation/Jev model IDs, request IDs/usage when supplied, all seven Jev probabilities, selected/extracted/failure counts, derived versions, graph counts, and artifact paths without credentials or full raw responses.

- [ ] **Step 8: Perform the small human extraction review**

Select a deterministic sample stratified across guidance, approval letter, multidisciplinary review, discipline review, and synthetic fixture source types. Complete `docs/kg-extraction-review.md` fields against current results, record observed omissions/unsupported claims without changing thresholds during the review, and mark the real review `not_run` if real extraction artifacts do not exist.

- [ ] **Step 9: Run final protected-input, secret, and artifact checks**

Run: `bin/runner pytest tests/test_runtime.py tests/integration/test_exports.py tests/integration/test_cli.py -v`

Run: `bin/runner python -m potency_kg.cli hash-inputs --verify /tmp/kg-protected-before.json`

Expected: protected path/hash set is identical; `results/kg/current` resolves to one immutable history version; scans find no configured secret values, Authorization headers, endpoint query strings, or full raw model responses.

- [ ] **Step 10: Commit**

```bash
git add lib/potency_kg/cli.py notebooks/01_build_knowledge_graph.ipynb notebooks/02_inspect_knowledge_graph.ipynb docs/kg-generation.md docs/kg-extraction-review.md tests/integration/test_cli.py tests/real/test_real_model_stages.py
git commit -m "feat: document and verify KG creation workflow"
```

## End-to-End Data Flow

```text
read-only configs/ + corpus/ + fixtures/ + schema/
                         │ hash before
                         ▼
              descriptor/profile translation
                         │
      manifest-first source import transaction
                         │
        relational structural representation
                         │ complete active passages only
                         ▼
        embedding cache ──► membership-aware index
                         │
          deterministic empty/page skip
                         │ eligible passage
                         ▼
 redaction substitution ──► one Jev request / seven Noul questions
                         │ selected bundles
                         ▼
 dynamic Instructor schema ──► complete validated extraction
                         │
       resolution + atomic relational materialization
                         │
     shadow AGE build ── validate ── atomic activation
                         │
     immutable history export ── atomic current promotion
                         │
                  results/kg/current/
                         │
              later hybrid-RAG phase
                         │
                   hash after/verify
```

## Completion Criteria

- The plan remains limited to KG creation and emits the intermediate `results/kg/` contract for a later hybrid-RAG plan.
- `configs/`, `corpus/`, `fixtures/`, and `schema/` are read-only at runtime and byte-for-byte/path-for-path unchanged after all workflows.
- Runtime settings use only the seven named variables, load lazily by stage, map generation to Instructor `openai/<model>`, and sanitize every stored endpoint/credential reference.
- The official embedding adapter uses `https://api.openai.com/v1`, `text-embedding-3-large`, float encoding, full `3072`-dimension vectors, response-index ordering, finite/dimension validation, and the same function for passage and query embeddings.
- Application-owned profiles, fixture translation, policy, adapters, prompts, and bundle definitions live under `lib/potency_kg/`.
- All 24 official sources and 791 normalized records reconcile; unchanged import still holds after reconnect/restart.
- Every valid version has relational structural graph records. Only active complete versions receive semantic work, and every active passage receives an embedding before skip or Jev decisions.
- Embedding, index, Jev response, Jev selection, extraction, and graph versions are derived independently from their exact declared inputs.
- One Jev request carries all seven `Noul` questions per eligible passage; every probability is persisted; threshold changes reuse response probabilities; no selected bundle means no LLM call; Jev failure never falls back to unrestricted extraction.
- Instructor accepts only the selected bundle types plus universal assertion/provenance fields; all local references and exact quotes validate; invalid responses never partially materialize.
- One mutable processing row reports a machine-readable current state/reason while all model, failure, materialization, and projection history remains immutable.
- Database constraints reject nonexistent and cross-project provenance for assertions, mentions, alias candidates, and node/edge source references.
- Stage-owner transactions prevent partial import, partial extraction activation, and current pointers to partial data.
- Source correction reuses unchanged vectors, changes membership-aware index identity, requires explicit replacement, and preserves prior source/run/export history.
- AGE refresh builds and validates a shadow graph, preserves the prior active graph on every injected failure, atomically switches readers, and cleans the former graph only after successful activation.
- Ordinary exports represent the current active graph; every graph version has immutable history; structural exports use explicit `not_run` and nullable semantic/index versions.
- Query embedding uses the passage adapter and retrieval traces contain query hash, versions, hits, seed/edge IDs, limits, and exact selected source references.
- Supplied fixture descriptors are translated in memory through one documented path resolver; all generated fixture data stays under `results/kg/fixtures/`.
- Trusted aliases have profile/configuration provenance; extracted candidates have exact evidence provenance; approval, redirects, UI, and review-triggered rematerialization remain deferred and documented.
- Deterministic tests cover threshold boundaries, Jev reuse/failure, extraction atomicity, composite provenance, cross-project collisions, injected rollbacks, source replacement, last-known-good AGE behavior, current/history export, protected hashes, reconnect idempotency, zero repeated embeddings, and secret/endpoint sanitization.
- Real Jev/embedding/generation checks run only when explicitly enabled, and a small question-independent human review spans multiple source types.
- `bin/runner` is executable and tested. Root `README.md` and `docs/scratchpad.md` remain byte-for-byte unchanged.
