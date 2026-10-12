# Scoped Alias Management and Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard-coded and extraction-only aliases with an idempotent, tenant/project-scoped alias registry populated by authoritative JSON seeds, deterministic corpus discovery, optional LLM discovery, and explicit human review.

**Architecture:** PostgreSQL stores one reviewable alias mapping plus immutable supporting evidence and discovery-run history. Static seeds and discovery feed the same registry, approved-catalog snapshots participate in Jev/extraction identity, and materialization resolves only exact approved Product and Assay aliases. Public stage wrappers serve both CLI commands and a fully decomposed operator notebook.

**Tech Stack:** Python 3.12, PostgreSQL 16, SQLAlchemy 2, Alembic, Pydantic 2, Instructor/OpenAI-compatible structured generation, pytest, Jupyter notebooks.

**Spec:** `docs/superpowers/specs/2026-10-12-scoped-alias-management-design.md`

## Global Constraints

- Every alias operation is scoped by exact `(tenant_id, project_id)`; no global inheritance or cross-scope fallback.
- Only `Product` and `Assay` are supported alias target types in this version.
- Static JSON aliases are authoritative, approved, source-controlled, idempotently reconciled, and not mutable through the review API.
- Deterministic, LLM, and extraction discoveries always enter as `pending`; confidence never approves a mapping.
- `DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY = False`; no environment variable silently enables LLM discovery.
- LLM discovery failure is recorded and does not block embeddings, Jev, extraction, or structural graph work.
- Approval changes future semantic versions but does not automatically rematerialize or republish the active graph.
- Supplied `configs/`, `corpus/`, `fixtures/`, and `schema/` remain byte-for-byte unchanged.
- Keep the root `README.md` and `docs/scratchpad.md` unchanged; do not read or process `docs/scratchpad.md`.
- Do not emit credentials, authorization headers, raw provider responses, or unredacted provider errors.
- Use `bin/runner` for migration and test commands so PostgreSQL/AGE behavior is verified in the supported container runtime.

---

### Task 1: Add the scoped alias registry migration

**Files:**
- Create: `migrations/versions/0005_create_scoped_alias_registry.py`
- Modify: `tests/conftest.py`
- Modify: `tests/integration/test_migrations.py`

**Interfaces:**
- Consumes: revision `0004_current_semantic_state`; legacy `alias_candidate`; existing fully scoped passage and extraction keys.
- Produces: `aliases`, `alias_evidence`, `alias_discovery_run`; `alias_catalog_version` columns on `jev_response` and `llm_extraction_run`; revision `0005_scoped_alias_registry`.

- [ ] **Step 1: Write migration tests that describe the final schema and legacy upgrade**

Add assertions for the three new tables, removal of `alias_candidate`, partial approved-alias uniqueness, approved/static/decision checks, composite scoped foreign keys, and non-null semantic catalog-version columns. Add a dedicated fixture that creates a disposable database at revision `0004_current_semantic_state`, inserts two legacy candidates, upgrades to head, and verifies two pending alias rows plus two extraction evidence rows with null canonical targets.

```python
def test_legacy_alias_candidates_migrate_without_invented_targets(migrated_0004_engine) -> None:
    engine, upgrade_to_head = migrated_0004_engine
    upgrade_to_head()
    with engine.connect() as connection:
        aliases = connection.execute(text(
            "SELECT alias_value,status,entity_type,canonical_key,canonical_name "
            "FROM aliases ORDER BY alias_value"
        )).all()
        evidence = connection.execute(text(
            "SELECT source_type,extraction_run_id FROM alias_evidence ORDER BY evidence_id"
        )).all()
    assert aliases == [
        ("CGV-X", "pending", None, None, None),
        ("exa-cel", "pending", None, None, None),
    ]
    assert all(row.source_type == "extraction" and row.extraction_run_id for row in evidence)
```

- [ ] **Step 2: Run the migration tests and verify they fail**

Run:

```bash
bin/runner pytest -q tests/integration/test_migrations.py
```

Expected: FAIL because revision `0005_scoped_alias_registry` and its tables/columns do not exist.

- [ ] **Step 3: Implement revision `0005_scoped_alias_registry`**

Create the three tables with the exact columns and checks from the spec. Use composite primary and foreign keys everywhere. Add these essential constraints/indexes:

```sql
CHECK (entity_type IS NULL OR entity_type IN ('Product','Assay')),
CHECK (status <> 'approved' OR
       (entity_type IS NOT NULL AND length(trim(canonical_key)) > 0
        AND length(trim(canonical_name)) > 0)),
CHECK (managed_by <> 'static_config' OR status IN ('approved','rejected')),
CHECK (status <> 'rejected' OR (decided_at IS NOT NULL AND decided_by IS NOT NULL))
```

```sql
CREATE UNIQUE INDEX uq_aliases_one_approved_target
ON aliases (tenant_id, project_id, entity_type, normalized_alias)
WHERE status = 'approved';
```

Add `alias_catalog_version TEXT NOT NULL` to both semantic-run tables using a temporary server default of `'legacy-unversioned'`, then remove the default. Preserve each legacy `alias_id`, copy its candidate text and provenance without deriving a target, validate migrated counts, and only then drop `alias_candidate`. The downgrade must recreate the legacy table and copy one representable extraction evidence row per alias before dropping the new structures.

- [ ] **Step 4: Run migration and constraint tests**

Run:

```bash
bin/runner pytest -q tests/integration/test_migrations.py
bin/runner alembic upgrade head
```

Expected: PASS; Alembic reaches `0005_scoped_alias_registry`.

- [ ] **Step 5: Commit the migration**

```bash
git add migrations/versions/0005_create_scoped_alias_registry.py tests/conftest.py tests/integration/test_migrations.py
git commit -m "feat: add scoped alias registry schema"
```

---

### Task 2: Implement alias normalization, authoritative seeds, catalog snapshots, and review decisions

**Files:**
- Create: `lib/potency_kg/aliases.py`
- Create: `lib/potency_kg/config_data/scoped_aliases.json`
- Create: `tests/unit/test_aliases.py`
- Create: `tests/integration/test_alias_registry.py`
- Modify: `lib/potency_kg/project_profiles.py`
- Modify: `tests/unit/test_project_profiles.py`

**Interfaces:**
- Consumes: Task 1 alias tables; `Scope`; `canonical_json()` and `derive_scoped_id()`.
- Produces:
  - `normalize_alias(value: str) -> str`
  - `load_alias_seeds(engine: Engine, scope: Scope, seed_path: Path = DEFAULT_ALIAS_SEED_PATH) -> AliasSeedReport`
  - `approved_alias_catalog(connection: Connection, scope: Scope) -> AliasCatalog`
  - `list_aliases(connection: Connection, scope: Scope, *, status: str | None = None, entity_type: str | None = None) -> Sequence[AliasRecord]`
  - `decide_alias(engine: Engine, scope: Scope, decision: AliasDecision) -> AliasRecord`
  - `alias_catalog_status(connection: Connection, scope: Scope) -> AliasCatalogStatus`

- [ ] **Step 1: Write unit tests for normalization, seed parsing, and catalog identity**

Define the public immutable contracts in the tests before implementation:

```python
@dataclass(frozen=True)
class AliasRecord:
    alias_id: str
    alias_value: str
    normalized_alias: str
    entity_type: Literal["Product", "Assay"] | None
    canonical_key: str | None
    canonical_name: str | None
    status: Literal["pending", "approved", "rejected"]
    managed_by: Literal["static_config", "review"]
    confidence: float | None

@dataclass(frozen=True, order=True)
class ApprovedAlias:
    alias_id: str
    alias_value: str
    normalized_alias: str
    entity_type: Literal["Product", "Assay"]
    canonical_key: str
    canonical_name: str
    normalization_version: str

@dataclass(frozen=True)
class AliasCatalog:
    scope: Scope
    version: str
    entries: Sequence[ApprovedAlias]

@dataclass(frozen=True)
class AliasDecision:
    alias_id: str
    status: Literal["approved", "rejected"]
    decided_by: str
    entity_type: Literal["Product", "Assay"] | None = None
    canonical_key: str | None = None
    canonical_name: str | None = None
    note: str | None = None

@dataclass(frozen=True)
class AliasSeedReport:
    scope: Scope
    inserted: int
    unchanged: int
    retired: int
    replaced: int
    evidence_inserted: int
    catalog_version: str

@dataclass(frozen=True)
class AliasCatalogStatus:
    scope: Scope
    catalog_version: str
    latest_jev_catalog_versions: Sequence[str]
    latest_extraction_catalog_versions: Sequence[str]
    semantic_stale: bool
```

Assert NFKC + casefold + whitespace normalization, deterministic ordering/versioning, invalid seed rejection before writes, and a version change for every approved mapping field.

- [ ] **Step 2: Write integration tests for seed reconciliation and review semantics**

Cover initial load, unchanged reload with zero inserted/updated rows, a removed seed becoming loader-rejected, atomic target replacement, static review refusal, pending approval/rejection, idempotent repeated decisions, contradictory repeat failure, approval uniqueness, cross-scope isolation, and review decisions surviving later evidence insertion.

The authoritative JSON must contain the existing scoped Product aliases:

```json
{
  "schema_version": "scoped-alias-seeds-v1",
  "entries": [
    {
      "tenant_id": "public-fda",
      "project_id": "casgevy-potency",
      "entity_type": "Product",
      "canonical_key": "casgevy",
      "canonical_name": "CASGEVY",
      "aliases": ["CASGEVY", "exagamglogene autotemcel", "exa-cel"]
    }
  ]
}
```

- [ ] **Step 3: Run the registry tests and verify they fail**

Run:

```bash
bin/runner pytest -q tests/unit/test_aliases.py tests/unit/test_project_profiles.py tests/integration/test_alias_registry.py
```

Expected: FAIL because `potency_kg.aliases` and the seed file do not exist.

- [ ] **Step 4: Implement `aliases.py` and the seed file**

Use these version constants:

```python
ALIAS_NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
ALIAS_CATALOG_VERSION = "approved-alias-catalog-v1"
STATIC_ALIAS_DETECTOR_VERSION = "scoped-alias-seeds-v1"
```

Parse and validate the whole JSON document before `engine.begin()`. Under one scoped transaction, lock existing static rows, upsert deterministic alias/evidence IDs, retire removed rows, and perform target replacements as retire-then-insert. `approved_alias_catalog()` must select only `status='approved'`, return canonical ordered immutable entries, and hash `canonical_json()` of all catalog fields plus scope and `ALIAS_CATALOG_VERSION`.

`decide_alias()` must lock the selected row with PostgreSQL `FOR UPDATE`, reject static rows, require complete target fields for approval, preserve decisions during later discovery, and return without writing when an identical decision is repeated.

- [ ] **Step 5: Remove alias literals from project profiles**

Change `ProjectProfile` to receive an `AliasCatalog` rather than construct aliases:

```python
@dataclass(frozen=True)
class ProjectProfile:
    name: str
    descriptor_path: str
    descriptor_sha256: str
    profile_version: str
    alias_catalog: AliasCatalog
    scope: Scope | None = None
    applications: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
```

Implement the exact signature `load_project_profile(scope: Scope, descriptor: Path, alias_catalog: AliasCatalog) -> ProjectProfile`. Derive `profile_version` from profile name, descriptor digest, applications, and `alias_catalog.version`. Keep application constants and descriptor-path mapping in Python; remove only alias literals.

- [ ] **Step 6: Run focused tests and commit**

```bash
bin/runner pytest -q tests/unit/test_aliases.py tests/unit/test_project_profiles.py tests/integration/test_alias_registry.py
git add lib/potency_kg/aliases.py lib/potency_kg/config_data/scoped_aliases.json lib/potency_kg/project_profiles.py tests/unit/test_aliases.py tests/unit/test_project_profiles.py tests/integration/test_alias_registry.py
git commit -m "feat: load and review scoped aliases"
```

Expected: PASS.

---

### Task 3: Add idempotent deterministic document-level alias discovery

**Files:**
- Create: `lib/potency_kg/alias_discovery.py`
- Create: `tests/unit/test_alias_discovery.py`
- Create: `tests/integration/test_alias_discovery.py`
- Modify: `lib/potency_kg/aliases.py`

**Interfaces:**
- Consumes: active complete source versions; Task 2 normalization and registry persistence.
- Produces:
  - `AliasProposal`
  - `AliasDiscoveryReport`
  - `discover_aliases_deterministic(engine: Engine, scope: Scope) -> AliasDiscoveryReport`
  - `record_alias_proposal(connection: Connection, scope: Scope, proposal: AliasProposal) -> ProposalPersistence`

- [ ] **Step 1: Write detector tests for accepted and rejected forms**

Use table-driven tests for both parenthetical directions, `also known as`, `abbreviated as`, and `formerly called`. Assert exact passage ID, quote, offsets, and canonical/alias direction. Include false positives for page numbers, numeric-only forms, unmatched parentheses, equal normalized strings, single-character abbreviations, extreme length ratios, and acronyms whose letters do not align with the long form.

```python
def test_parenthetical_long_form_and_acronym_is_grounded() -> None:
    proposals = discover_alias_patterns(
        ScopedPassageIdentity("t", "p", "s", "v", "p1"),
        "Exagamglogene autotemcel (exa-cel) was evaluated.",
    )
    assert [(p.alias_value, p.canonical_name, p.exact_quote)] == [
        ("exa-cel", "Exagamglogene autotemcel", "Exagamglogene autotemcel (exa-cel)")
    ]
```

- [ ] **Step 2: Write persistence and rerun tests**

Import two active complete documents with repeated evidence for the same proposal. Assert one alias row, multiple evidence rows, one successful run per document, exact source provenance, and a second call reporting reused runs with unchanged row counts. Replace one source version and assert only that document gets a new run/evidence identity. Verify a prior rejection remains rejected.

- [ ] **Step 3: Run the discovery tests and verify they fail**

```bash
bin/runner pytest -q tests/unit/test_alias_discovery.py tests/integration/test_alias_discovery.py
```

Expected: FAIL because deterministic discovery is not implemented.

- [ ] **Step 4: Implement conservative rules and document transactions**

Define:

```python
DETERMINISTIC_ALIAS_DETECTOR = "document-alias-patterns"
DETERMINISTIC_ALIAS_DETECTOR_VERSION = "document-alias-patterns-v1"

@dataclass(frozen=True)
class AliasProposal:
    alias_value: str
    canonical_name: str | None
    entity_type: Literal["Product", "Assay"] | None
    canonical_key: str | None
    confidence: float
    source: ScopedPassageIdentity
    exact_quote: str
    original_start: int
    original_end: int
    source_type: Literal["deterministic_rule", "llm_mined", "extraction"]
    detector_name: str
    detector_version: str

@dataclass(frozen=True)
class ProposalPersistence:
    alias_id: str
    alias_inserted: bool
    evidence_inserted: bool

@dataclass(frozen=True)
class AliasDiscoveryReport:
    method: Literal["deterministic", "llm"]
    documents: int
    succeeded: int
    failed: int
    reused: int
    proposals: int
    evidence_inserted: int
    discovery_run_ids: Sequence[UUID]
```

Fetch active complete documents in stable source/passage order. Derive the document input hash from ordered passage IDs, text hashes, and detector version. Reuse only matching successful runs. Persist one document's run, proposals, and evidence atomically; record a sanitized failed run if validation unexpectedly fails, then continue with later documents.

- [ ] **Step 5: Run focused and importer idempotency tests, then commit**

```bash
bin/runner pytest -q tests/unit/test_alias_discovery.py tests/integration/test_alias_discovery.py tests/integration/test_import.py
git add lib/potency_kg/alias_discovery.py lib/potency_kg/aliases.py tests/unit/test_alias_discovery.py tests/integration/test_alias_discovery.py
git commit -m "feat: discover aliases deterministically"
```

Expected: PASS, including unchanged structural-import counts after discovery reruns.

---

### Task 4: Add separately gated LLM alias discovery

**Files:**
- Create: `lib/potency_kg/prompt_resources/alias_discovery.py`
- Create: `tests/unit/test_alias_discovery_llm.py`
- Modify: `lib/potency_kg/alias_discovery.py`
- Modify: `tests/integration/test_alias_discovery.py`

**Interfaces:**
- Consumes: `StructuredExtractionClient`, generation settings, document windows, Task 3 persistence.
- Produces: `discover_aliases_llm(engine: Engine, scope: Scope, client: StructuredExtractionClient, *, enabled: bool = DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY) -> AliasDiscoveryReport`.

- [ ] **Step 1: Write tests for the disabled boundary and typed grounding**

Assert `DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY is False`, a disabled call makes no database or client call, valid responses persist pending proposals, repeated calls reuse successful runs, oversized documents split into deterministic overlapping passage windows, and every accepted quote maps exactly to its declared original passage and offsets.

Define the strict response contract in tests:

```python
class AliasDiscoveryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias_value: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: Literal["Product", "Assay"] | None = None
    passage_id: str = Field(min_length=1)
    supporting_quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)

class AliasDiscoveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aliases: list[AliasDiscoveryItem]
```

- [ ] **Step 2: Add failure-path tests**

Use test doubles for invalid entity type, unknown passage, quote mismatch, ambiguous repeated quote without occurrence index, transient exhaustion, and provider exception containing a fake secret. Assert no partial alias/evidence rows, a failed run with sanitized detail, preserved earlier successful documents, and no secret in reports or captured output.

- [ ] **Step 3: Run the LLM discovery tests and verify they fail**

```bash
bin/runner pytest -q tests/unit/test_alias_discovery_llm.py tests/integration/test_alias_discovery.py
```

Expected: FAIL because the LLM discovery contract and prompt do not exist.

- [ ] **Step 4: Implement the prompt, windowing, validation, and failure persistence**

Set:

```python
DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY = False
LLM_ALIAS_DETECTOR = "structured-document-alias-discovery"
LLM_ALIAS_DETECTOR_VERSION = "structured-document-alias-discovery-v1"
MAX_ALIAS_DISCOVERY_CHARACTERS = 48_000
ALIAS_DISCOVERY_OVERLAP_PASSAGES = 1
```

Hash the prompt and Pydantic JSON schema into run identity. Reuse `StructuredExtractionClient` and its thread-local usage/attempt accessors without changing the provider adapter. Validate and restore grounding before opening the proposal transaction. Convert provider and protocol failures to the fixed persisted detail `"alias discovery failed"`.

- [ ] **Step 5: Run tests and commit**

```bash
bin/runner pytest -q tests/unit/test_alias_discovery_llm.py tests/integration/test_alias_discovery.py tests/unit/test_extraction.py
git add lib/potency_kg/alias_discovery.py lib/potency_kg/prompt_resources/alias_discovery.py tests/unit/test_alias_discovery_llm.py tests/integration/test_alias_discovery.py
git commit -m "feat: add optional LLM alias discovery"
```

Expected: PASS with no external provider calls.

---

### Task 5: Bind Jev, extraction, and materialization to the approved alias catalog

**Files:**
- Modify: `lib/potency_kg/ids.py`
- Modify: `lib/potency_kg/jev_processing.py`
- Modify: `lib/potency_kg/extraction.py`
- Modify: `lib/potency_kg/resolution.py`
- Modify: `lib/potency_kg/cli.py`
- Modify: `tests/unit/test_ids.py`
- Modify: `tests/unit/test_extraction.py`
- Modify: `tests/integration/test_jev_processing.py`
- Modify: `tests/integration/test_model_processing.py`
- Modify: `tests/integration/test_materialization.py`
- Modify: `tests/integration/test_exports.py`
- Modify: `tests/integration/test_retrieval_smoke.py`

**Interfaces:**
- Consumes: `AliasCatalog` and `alias_catalog_version` from Task 2; typed proposals from Tasks 3–4.
- Produces: exact approved Product/Assay resolution; semantic reuse rejection after catalog changes; extraction-origin alias evidence in the unified registry.

- [ ] **Step 1: Write versioning and stale-state regression tests**

Update ID tests so Jev/extraction profile payloads contain ordered alias entries and `alias_catalog_version`. Assert any approved mapping change alters Jev response, extraction, profile, and eventual graph identity. In integration tests: classify/extract with catalog A, approve another alias to create catalog B, then prove old response reuse and old materialization are rejected until semantic stages rerun.

- [ ] **Step 2: Write Product/Assay resolution and extraction-proposal tests**

Replace the profile-only resolution test with database-backed approved/pending/rejected mappings. Assert exact normalization resolves approved Product and Assay mentions to stable scoped nodes, similar text remains a new entity, and pending/rejected aliases never merge. Change `AliasCandidate` extraction records from Product-only `product_local_id` to typed `target_local_id`; validation must accept only local Product or Assay targets.

- [ ] **Step 3: Run focused semantic tests and verify they fail**

```bash
bin/runner pytest -q tests/unit/test_ids.py tests/unit/test_extraction.py tests/integration/test_jev_processing.py tests/integration/test_model_processing.py tests/integration/test_materialization.py
```

Expected: FAIL because semantic runs do not yet carry or validate catalog versions.

- [ ] **Step 4: Thread the catalog through profile, Jev, and extraction identity**

Change `_profile_for_scope` to accept a database connection/engine, load `approved_alias_catalog`, and call `load_project_profile(scope, descriptor, catalog)`. Serialize the exact ordered catalog payload into existing `profile_aliases` JSONB and write its version to `alias_catalog_version`. Add that version to all cache/reuse predicates and to `_pending_materializations()` current-identity validation.

Do not coerce legacy object-shaped `profile_aliases` into the new list shape. Rows carrying `legacy-unversioned` must miss reuse and be reprocessed.

- [ ] **Step 5: Replace profile bootstrap resolution with live approved-catalog resolution**

Remove `bootstrap_profile_aliases()`. At the start of the materialization transaction, call `approved_alias_catalog(connection, scope)`, require its version to equal both the Jev response and validated extraction run, and pass the immutable catalog into `resolve_mention(connection: Connection, scope: Scope, mention: EntityMentionDraft, catalog: AliasCatalog) -> ResolvedEntity`.

Match on `(entity_type, normalize_alias(mention.text))`. Upsert canonical Product or Assay anchors only for approved matches, recording alias/catalog provenance in node attributes. Keep Application bootstrap behavior separate from aliases.

- [ ] **Step 6: Persist extraction alias proposals in the unified registry**

For an extracted `AliasCandidate`, validate `target_local_id` points to a local Product or Assay, derive proposed target fields from that local entity, and call `record_alias_proposal()` with `source_type='extraction'`, exact quote/offsets, extraction run ID, prompt/schema hashes, and extraction input hash. Rename `MaterializationReport.alias_candidates` to `alias_proposals` and update count assertions.

- [ ] **Step 7: Run the affected cross-stage suite and commit**

```bash
bin/runner pytest -q tests/unit/test_ids.py tests/unit/test_extraction.py tests/integration/test_jev_processing.py tests/integration/test_model_processing.py tests/integration/test_materialization.py tests/integration/test_exports.py tests/integration/test_retrieval_smoke.py
git add lib/potency_kg/ids.py lib/potency_kg/jev_processing.py lib/potency_kg/extraction.py lib/potency_kg/resolution.py lib/potency_kg/cli.py tests/unit/test_ids.py tests/unit/test_extraction.py tests/integration/test_jev_processing.py tests/integration/test_model_processing.py tests/integration/test_materialization.py tests/integration/test_exports.py tests/integration/test_retrieval_smoke.py
git commit -m "feat: resolve entities through approved aliases"
```

Expected: PASS; no test uses hard-coded Python aliases.

---

### Task 6: Expose independent stages through the CLI and build notebook

**Files:**
- Create: `lib/potency_kg/workflow_stages.py`
- Modify: `lib/potency_kg/cli.py`
- Modify: `notebooks/01_build_knowledge_graph.ipynb`
- Modify: `tests/integration/test_cli.py`
- Create: `tests/integration/test_workflow_stages.py`

**Interfaces:**
- Consumes: all stage functions from Tasks 2–5 and existing import/embed/Jev/extract/materialize/refresh/export functions.
- Produces public protected wrappers used identically by the CLI and notebook:
  - `preflight_stage(config: Path) -> object`
  - `import_stage(config: Path) -> object`
  - `load_alias_seeds_stage(config: Path) -> AliasSeedReport`
  - `discover_aliases_deterministic_stage(config: Path) -> AliasDiscoveryReport`
  - `discover_aliases_llm_stage(config: Path, *, enabled: bool, env: Mapping[str, str] | None = None) -> AliasDiscoveryReport`
  - `inspect_aliases_stage(config: Path, *, status: str | None = None) -> object`
  - existing semantic/graph stages through equivalent named wrappers.

- [ ] **Step 1: Write wrapper and parser tests**

Extend the exact parser command set with:

```text
load-alias-seeds
discover-aliases-deterministic
discover-aliases-llm
list-aliases
review-alias
```

Assert wrappers verify protected inputs in `finally`, construct no model client for deterministic stages, require the real-model gate and generation configuration only for the explicit LLM command, and emit JSON-serializable sanitized reports. Assert the composite structural workflow orders `import`, `load-alias-seeds`, and `discover-aliases-deterministic` before structural refresh.

- [ ] **Step 2: Write notebook structure tests**

Replace the old assertion requiring `run_kg_workflow` with ordered cell assertions. Require one explanatory Markdown cell and one call cell for each of:

```text
preflight, structural import, static aliases, deterministic discovery,
optional LLM discovery, alias inspection/review, embeddings, Jev,
extraction, materialization, refresh, export, inspection
```

Assert every write-stage explanation mentions inputs, outputs, database effect, rerun behavior, and invalidation; each write cell is followed by a read-only inspection cell; LLM discovery passes `DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY`; code cells define no functions/classes, contain no credential names/values, and have empty outputs.

- [ ] **Step 3: Run CLI/notebook tests and verify they fail**

```bash
bin/runner pytest -q tests/integration/test_cli.py tests/integration/test_workflow_stages.py
```

Expected: FAIL because stage wrappers, commands, and decomposed notebook cells do not exist.

- [ ] **Step 4: Implement protected public stage wrappers and CLI commands**

Move common config/scope/engine/protected-input ownership from private CLI closures into `workflow_stages.py`. Each wrapper opens and disposes its own engine, calls exactly one stage, and preserves the existing protected-input `finally` guarantee. `run_kg_workflow()` delegates to these underlying stage owners without duplicating business logic.

`review-alias` requires `--alias-id`, `--decision`, and `--reviewer`; approval additionally requires `--entity-type`, `--canonical-key`, and `--canonical-name`. `list-aliases` accepts optional status/entity filters. The explicit LLM discovery command passes `enabled=True`; the composite workflow passes the module default.

- [ ] **Step 5: Rewrite the build notebook as explained independent stages**

Use imports and calls only. The setup cell computes `package_root`, `config_path`, `output_root`, and `enable_llm_alias_discovery = DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY`. Every stage cell assigns a report, displays a small sanitized summary, and can be rerun independently. The alias review section lists pending rows and shows commented CLI examples in Markdown; it must not auto-approve.

- [ ] **Step 6: Run CLI/notebook tests and commit**

```bash
bin/runner pytest -q tests/integration/test_cli.py tests/integration/test_workflow_stages.py
git add lib/potency_kg/workflow_stages.py lib/potency_kg/cli.py notebooks/01_build_knowledge_graph.ipynb tests/integration/test_cli.py tests/integration/test_workflow_stages.py
git commit -m "feat: expose inspectable alias workflow stages"
```

Expected: PASS with no external calls.

---

### Task 7: Update operator documentation and run end-to-end verification

**Files:**
- Modify: `docs/kg-generation.md`
- Modify: `docs/kg-extraction-review.md`
- Modify: `tests/integration/test_behavior_graphs.py`
- Modify: `tests/real/test_real_model_stages.py`

**Interfaces:**
- Consumes: completed scoped alias workflow and notebook stage API.
- Produces: operator guidance, fixture coverage, explicitly gated real discovery check, and final repository evidence.

- [ ] **Step 1: Write end-to-end behavior tests**

Extend fixture orchestration coverage to prove structural import, seed loading, and deterministic discovery are idempotent; pending aliases do not affect KG identity; an approved alias changes the catalog/semantic version on the next run; and a subsequent graph export remains immutable and scoped. Add a `real_models` test that calls LLM alias discovery only when the existing generation gate/configuration is valid and otherwise reports `not_run` rather than passing silently.

- [ ] **Step 2: Run the new end-to-end tests and verify the documentation expectations fail**

```bash
bin/runner pytest -q tests/integration/test_behavior_graphs.py tests/real/test_real_model_stages.py -m "not real_models"
```

Expected before documentation/test-fixture updates: at least one new alias-workflow assertion fails.

- [ ] **Step 3: Update operator documentation**

In `docs/kg-generation.md`, replace the deferred-alias section with the exact stage order, idempotency contract, static-vs-review ownership, catalog invalidation behavior, CLI examples, and notebook usage. In `docs/kg-extraction-review.md`, document pending/approved/rejected review criteria, required approval target fields, evidence inspection, static-row immutability, and the fact that approval requires an explicit later semantic rebuild.

- [ ] **Step 4: Run the full offline verification gate**

```bash
bin/runner alembic upgrade head
bin/runner pytest -q -m "not real_models"
git diff --check
git diff --quiet -- README.md configs corpus fixtures schema
```

Expected: migration succeeds; all offline tests pass; four protected input roots and root README have no diff; only the pre-existing Alembic `path_separator` warning may remain.

- [ ] **Step 5: Run a fresh-database structural workflow twice**

Create a disposable database through the existing test/runtime mechanism, run preflight, structural import, static seed load, deterministic discovery, and structural refresh twice, then assert:

```text
source count: 24
passage count: 791
approved static Product aliases: 3
second import inserted sources/passages: 0/0
second static seed load inserted/updated: 0/0
second deterministic discovery new runs/evidence: 0/0
```

Also assert protected-input hashes are unchanged and remove only the disposable database through the existing fixture/runner cleanup path.

- [ ] **Step 6: Refresh the codebase graph and commit documentation/tests**

```bash
graphify update .
git add docs/kg-generation.md docs/kg-extraction-review.md tests/integration/test_behavior_graphs.py tests/real/test_real_model_stages.py graphify-out
git commit -m "docs: document scoped alias workflow"
```

- [ ] **Step 7: Perform final review**

Run an independent whole-plan review against the spec, fix any Critical or Important findings, rerun the full offline gate, and commit the reviewed fixes separately. Report real-provider stages as passed, failed, or `not_run` from actual evidence; never infer success from deterministic test doubles.
