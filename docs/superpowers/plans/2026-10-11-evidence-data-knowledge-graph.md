# Evidence Data and Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the complete supplied corpus into an idempotent relational evidence store, extract a provenance-grounded semantic model, project it into project-isolated Apache AGE graphs, and support vector-seeded graph retrieval back to exact source passages.

**Architecture:** PostgreSQL relational tables are the durable system of record for source versions, passages, embeddings, model runs, mentions, aliases, assertions, graph nodes, and graph edges. Apache AGE is a reproducible projection of the canonical `kg_node` and `kg_edge` tables, with one AGE graph per tenant/project scope; pgvector stores passage embeddings in the same database. All 791 passages receive deterministic structural treatment, while a corpus-wide relevance policy selects passages for one structured generative extraction pass that also identifies explicit alias claims.

**Tech Stack:** Python 3.12, Pydantic 2, Instructor, TypeSafe/Jev, SQLAlchemy 2, Psycopg 3, Alembic, PostgreSQL 17, pgvector, Apache AGE, pytest.

**Spec:** `ASSIGNMENT.md`, `MODEL_ACCESS.md`, `schema/graph-core.json`, `schema/output-contract.json`, and `fixtures/BEHAVIOR_CASES.md`.

## Global Constraints

- Keep the root `README.md` unchanged unless the user explicitly requests an edit.
- Do not read or modify `docs/scratchpad.md`.
- Treat `(tenant_id, project_id, source_id, source_version, passage_id)` as passage identity; `passage_id` alone is never globally unique.
- Import and account for all 791 normalized records and create an inspectable `SourceDocument` representation for every supplied source.
- Treat raw files as source of truth and `corpus/records.jsonl` as a supplied mechanical extraction.
- Do not ingest assignment text, questions, schemas, fixture contracts, or generated outputs as FDA evidence.
- Preserve every source version and prior run artifact; a correction changes the active source head without deleting history.
- Keep tenant and project in every project-owned relational key, node ID, edge ID, cache key, trace, and graph-selection operation.
- Never traverse from one project graph into another project graph.
- Automatically extracted assertions, relationships, and aliases remain `unreviewed` or `pending_review`; model agreement is not human review.
- Do not merge redacted markers across passages. Redaction identity includes full passage identity and occurrence index.
- Do not infer concealed `(b)(4)` content.
- Apply a corpus-wide semantic selection policy rather than rules keyed to P1-P6, fixture names, or expected answers.
- Make unchanged re-import, re-embedding, re-extraction, and graph synchronization idempotent.
- Every evidence-bearing assertion and edge retains exact source references plus extraction actor, version, timestamp, and review state.
- Use original source text for citations and final evidence. Redaction-substituted text is only an extraction input and is stored separately.
- Use parameterized SQL for values. AGE graph names and labels are generated only by internal allowlisted functions, never from raw user text.
- Keep the implementation small: no LangChain, LlamaIndex, NetworkX, generic ontology framework, PDF parser, OCR pipeline, or graph visualization.

---

## Accepted Design Decisions

### Canonical storage and projection

The relational schema is authoritative. `kg_node`, `kg_edge`, and their normalized evidence-reference tables contain the exact graph that must be exported and projected. AGE is rebuilt or incrementally synchronized from those tables and is used for bounded graph traversal. AGE internal vertex and edge IDs are never exposed as stable application identifiers; each vertex and edge carries the logical `node_id` or `edge_id` property.

### Scope and graph boundaries

`scope = (tenant_id, project_id)`. One AGE graph is created per scope, using a database-controlled name such as `kg_5c4b15a82e9d`. A `graph_scope` relational table maps the full scope to that generated name. This permits connections across every document in one project and physically prevents graph traversal into another project. Production reuse of shared public evidence is a separately authorized shared scope; it does not justify removing `project_id` from project-owned identities.

### Structural versus semantic coverage

Every valid source and passage creates structural nodes and `contains_passage` edges. Each passage records one semantic-processing outcome: `structural_only`, `selected`, `extracted`, or `failed`. Empty/page-number-only passages may be deterministically classified `structural_only`; all other passages receive the same relevance policy. Only selected passages receive generative semantic extraction.

### Extraction and alias resolution

There is one generative extraction pass per selected, unchanged passage. The response contains passage-local entity mentions, assertions, relationships, and only explicitly supported alias claims. Alias resolution is a later deterministic phase over the aggregate extraction output; a second LLM pass is not required. Configuration-supplied identity mappings are trusted as `configured`, LLM-extracted alias claims enter `pending_review`, and only `configured` or human-`approved` aliases participate in automatic merging.

### Entity identity

Identity strategies are registered by entity type:

- `SourceDocument`: scope + source ID + source version.
- `SourcePassage`: full five-part record identity.
- `Application`: normalized authoritative identifier such as `STN 125785`.
- `Product`: configured product ID such as `casgevy`.
- Redacted entity: full passage identity + redaction occurrence index.
- Other semantic entities: scope + entity type + compatible product/application context + normalized name, after configured/approved alias lookup.

Node identity and display labels are separate. A display label may improve without changing the node ID. If two previously materialized name-based entities are later approved as equivalent, retain one canonical node ID and store the other in `entity_redirect`; never silently rewrite historical extraction records.

### Edge identity

Structural edge identity is a digest of scope, edge type, source node ID, and target node ID. Evidence-bearing edge identity additionally includes the assertion ID. Reprocessing the same extraction upserts the same edge; a finding from a different passage has a different assertion and therefore remains separately inspectable.

### Redactions

The exact `source_text` is immutable. Extraction receives `model_text`, in which variants such as `(b)(4)` and `(b) (4)` become opaque tokens such as `[[REDACTED:r_7fd81a:01]]`. The token maps to tenant, project, source, source version, passage, and occurrence index in `redaction_mention`. The model may classify the role of a redacted mention from surrounding text but may not supply or infer its concealed value.

### Graph seed selection

Graph retrieval embeds the question, retrieves scoped source passages through pgvector, maps those passages to `SourcePassage` graph nodes, and uses adjacent semantic nodes as seeds. It then traverses a declared one- or two-hop neighborhood with node and edge caps, collects evidence references, and reloads original passages from the relational store. This avoids an additional query-entity-linking model while ensuring the graph participates in evidence selection.

## Planned File Structure

```text
requirements.txt
migrations/versions/
  0002_create_evidence_store.py
  0003_create_knowledge_graph_store.py
lib/potency_kg/
  __init__.py
  config.py                 # Validated project/model configuration
  db.py                     # SQLAlchemy engine and transaction helpers
  contracts.py              # Pydantic import/extraction/graph/retrieval contracts
  ids.py                    # Scoped stable IDs and AGE graph names
  redactions.py             # Redaction detection, substitution, and mapping
  ingest.py                 # JSONL import, source heads, counts, structural records
  embeddings.py             # Remote embedding adapter, cache, and vector retrieval
  relevance.py              # Corpus-wide Jev relevance decisions
  extraction.py             # Instructor extraction and deterministic alias patterns
  aliases.py                # Alias candidates, review state, and lookup
  resolution.py             # Mention-to-entity resolution and assertion materialization
  graph_store.py            # Canonical kg_node/kg_edge writes
  age_projection.py         # Scoped AGE graph synchronization and validation
  retrieval.py              # Vector and vector-seeded graph retrieval
  export.py                 # Required JSON/JSONL graph and import artifacts
  cli.py                    # Reproducible command entry points
prompts/
  relevance-v1.md
  extraction-v1.md
tests/
  conftest.py
  unit/
    test_ids.py
    test_redactions.py
    test_extraction_contract.py
    test_aliases.py
    test_resolution.py
  integration/
    test_import.py
    test_embeddings.py
    test_age_projection.py
    test_retrieval.py
    test_behavior_cases.py
    test_exports.py
```

## Relational Data Model

The migrations create these tables and constraints:

- `import_run`: run ID, scope, corpus snapshot, status, timestamps, and reconciled counts.
- `source_document`: scoped source/version identity, metadata, raw path, content metadata, and import run.
- `source_head`: one active source version per `(tenant, project, source_id)`; corrections update this pointer only.
- `source_passage`: full record identity, exact text, hash, locator, authority, status, subject scope, and metadata.
- `embedding_cache`: provider/model/input hash, vector dimension, vector value, usage, and timestamps.
- `passage_embedding`: scoped passage, embedding-cache row, index version, and active state.
- `model_run`: provider, model, task, prompt/extractor version, input hash, sanitized usage, elapsed time, status, and error.
- `passage_processing`: scoped passage, deterministic/relevance decisions, outcome, reasons, and model-run references.
- `redaction_mention`: scoped passage, occurrence, raw marker, opaque token, character offsets, and contextual role when extracted.
- `entity_mention`: passage-local mention ID, exact span/text, entity type, context IDs, extraction run, and resolved node ID.
- `entity_alias`: alias text, normalized alias, relation type, target node, status, provenance, proposer, and review fields.
- `entity_redirect`: deprecated node ID to canonical node ID with approved merge provenance.
- `assertion`: stable assertion ID, passage, text, state, authority/status, event date, extraction provenance, and review state.
- `kg_node`: output-contract node fields plus extraction fields required for evidence-bearing assertions.
- `kg_node_source_ref`: normalized source references for graph nodes.
- `kg_edge`: stable edge ID, type, endpoints, status, actor, version, creation time, review state, and assertion ID.
- `kg_edge_source_ref`: normalized source references for graph edges.
- `graph_scope`: scope-to-AGE-graph mapping and last synchronized graph version.
- `graph_sync_run`: desired graph version, counts, validation result, status, and timestamps.

All project-owned primary or unique keys include `tenant_id` and `project_id`, directly or through a stable ID whose digest input includes them. `kg_edge.source_node_id` and `kg_edge.target_node_id` reference `kg_node.node_id`, and database triggers or application validation reject endpoint scope mismatches.

---

### Task 1: Add Direct Dependencies and Core Contracts

**Files:**
- Modify: `requirements.txt`
- Create: `lib/potency_kg/__init__.py`
- Create: `lib/potency_kg/contracts.py`
- Create: `tests/unit/test_extraction_contract.py`

**Interfaces:**
- Produces: `Scope`, `SourceRef`, `EntityMentionDraft`, `AliasClaimDraft`, `AssertionDraft`, `RelationDraft`, `PassageExtraction`, `GraphNodeRecord`, and `GraphEdgeRecord` Pydantic models.
- Consumes: enums and required fields from `schema/graph-core.json` and `schema/output-contract.json`.

- [ ] **Step 1: Add pinned direct dependencies**

Add these lines while preserving existing pins:

```text
instructor==1.17.0
pgvector==0.5.1
pydantic==2.14.0
pytest==9.1.1
tenacity==9.2.1
typesafe-sdk==0.7.2
```

- [ ] **Step 2: Write failing contract tests**

Test that unknown entity/edge types fail, `review_state` defaults to `unreviewed`, local relation endpoints must exist in the same extraction, and evidence-bearing relations require a non-empty supporting quote.

```python
def test_extraction_rejects_unknown_relation_endpoint():
    with pytest.raises(ValueError, match="unknown local entity"):
        PassageExtraction.model_validate({
            "entities": [{"local_id": "e1", "type": "Assay", "text": "TIDE"}],
            "assertions": [],
            "relations": [{
                "source_local_id": "e1",
                "type": "measures",
                "target_local_id": "e2",
                "supporting_quote": "TIDE measures editing frequency",
            }],
            "alias_claims": [],
        })
```

- [ ] **Step 3: Run the contract test and verify failure**

Run: `pytest tests/unit/test_extraction_contract.py -v`

Expected: import failure because `potency_kg.contracts` does not exist.

- [ ] **Step 4: Implement the contracts and validators**

Use `Literal` values for the agreed node, edge, assertion-state, assay-role, alias-relation, and review-state vocabularies. Extend the supplied edge vocabulary with `mentions`, `contradicts`, `supersedes`, and `supplements`; retain `supersedes_or_supplements` only when importing supplied schema data.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/unit/test_extraction_contract.py -v`

Expected: all contract tests pass.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt lib/potency_kg/__init__.py lib/potency_kg/contracts.py tests/unit/test_extraction_contract.py
git commit -m "feat: define evidence extraction contracts"
```

### Task 2: Implement Stable Identities and Redaction Mapping

**Files:**
- Create: `lib/potency_kg/ids.py`
- Create: `lib/potency_kg/redactions.py`
- Create: `tests/unit/test_ids.py`
- Create: `tests/unit/test_redactions.py`

**Interfaces:**
- Produces: `document_id(scope, source_id, source_version) -> str`, `passage_id(scope, source_id, source_version, local_passage_id) -> str`, `application_id(scope, raw_identifier) -> str`, `configured_product_id(scope, product_id) -> str`, `name_entity_id(scope, entity_type, context_id, normalized_name) -> str`, `edge_id(scope, edge_type, source_id, target_id, assertion_id=None) -> str`, and `graph_name_for_scope(scope) -> str`.
- Produces: `substitute_redactions(source_ref, text) -> RedactionSubstitution` containing immutable source text, model text, and ordered redaction mappings.

- [ ] **Step 1: Write identity tests**

Cover deterministic reruns, different project IDs producing different values, STN/BLA spelling normalization, graph names matching `^kg_[a-f0-9]{16}$`, and evidence edge IDs differing by assertion.

```python
def test_passage_identity_is_project_scoped():
    left = passage_id(Scope(tenant_id="t", project_id="p1"), "s", "v", "x")
    right = passage_id(Scope(tenant_id="t", project_id="p2"), "s", "v", "x")
    assert left != right
```

- [ ] **Step 2: Write redaction tests**

Verify that `(b)(4)`, `(b) (4)`, and whitespace variants are replaced in `model_text`; source text remains byte-for-byte unchanged; multiple occurrences get distinct tokens; and identical local passage IDs in different projects generate distinct opaque prefixes.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/unit/test_ids.py tests/unit/test_redactions.py -v`

Expected: import failures for the missing modules.

- [ ] **Step 4: Implement scoped UUIDv5 or SHA-256 identifiers**

Serialize digest inputs as canonical JSON arrays rather than delimiter-joined strings. Prefix IDs with their logical kind, for example `passage:`, `application:`, `product:`, `assertion:`, and `edge:`.

- [ ] **Step 5: Implement redaction substitution**

Use a compiled regex for supported `(b)(4)` spacing variants. Derive the opaque passage token from the full scoped passage identity and append a one-based occurrence index. Store character offsets into original text.

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_ids.py tests/unit/test_redactions.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/ids.py lib/potency_kg/redactions.py tests/unit/test_ids.py tests/unit/test_redactions.py
git commit -m "feat: add scoped identities and redaction tokens"
```

### Task 3: Create the Relational Evidence Schema

**Files:**
- Create: `migrations/versions/0002_create_evidence_store.py`
- Create: `migrations/versions/0003_create_knowledge_graph_store.py`
- Create: `lib/potency_kg/db.py`
- Create: `tests/integration/test_import.py`

**Interfaces:**
- Produces: SQLAlchemy engine helpers `create_engine_from_env()`, `transaction(engine)`, and the relational tables listed in this plan.
- Consumes: the existing `0001_enable_extensions.py` migration.

- [ ] **Step 1: Write a migration smoke test**

The test upgrades an empty test database to `head`, queries `information_schema.tables`, and asserts every planned table exists. It also attempts a duplicate full passage identity and expects the database uniqueness constraint to reject it.

- [ ] **Step 2: Run the migration test and verify failure**

Run: `pytest tests/integration/test_import.py::test_migrations_create_evidence_schema -v`

Expected: failure because the tables do not exist.

- [ ] **Step 3: Implement `0002_create_evidence_store.py`**

Create import, source, source-head, embedding, model-run, processing, redaction, mention, alias, redirect, and assertion tables. Use PostgreSQL `JSONB`, timezone-aware timestamps, explicit check constraints for statuses, and `vector` without a fixed dimension so the not-yet-completed access sheet does not have to be guessed.

- [ ] **Step 4: Implement `0003_create_knowledge_graph_store.py`**

Create canonical node/edge tables, normalized node/edge evidence-reference tables, graph scope, and graph sync runs. Add endpoint foreign keys and indexes for `(tenant_id, project_id, type)`, logical IDs, active source heads, input hashes, normalized aliases, and passage-vector joins.

- [ ] **Step 5: Implement connection helpers**

Set `search_path` explicitly per connection and register pgvector with Psycopg 3. Do not add `apache-age-python`; later AGE operations use Psycopg with parameterized SQL and controlled identifiers.

- [ ] **Step 6: Run migrations and tests**

Run: `bin/migrate`

Run: `pytest tests/integration/test_import.py::test_migrations_create_evidence_schema -v`

Expected: migration succeeds and the schema test passes.

- [ ] **Step 7: Commit**

```bash
git add migrations/versions/0002_create_evidence_store.py migrations/versions/0003_create_knowledge_graph_store.py lib/potency_kg/db.py tests/integration/test_import.py
git commit -m "feat: add relational evidence and graph schema"
```

### Task 4: Import Sources, Passages, and Structural Graph Records

**Files:**
- Create: `lib/potency_kg/config.py`
- Create: `lib/potency_kg/ingest.py`
- Create: `lib/potency_kg/graph_store.py`
- Modify: `tests/integration/test_import.py`

**Interfaces:**
- Produces: `load_project_config(path) -> ProjectConfig`, `import_records(engine, config) -> ImportReport`, and `upsert_structural_graph(engine, scope, source_ids) -> GraphWriteReport`.
- Produces canonical `SourceDocument` and `SourcePassage` rows in `kg_node` plus `contains_passage` rows in `kg_edge`.

- [ ] **Step 1: Write import tests using small fixture JSONL files**

Assert that the first import reports imported rows, the second unchanged import reports zero new rows, source versions remain distinct, project-one records never appear under project-two scope, and the totals reconcile as `imported + unchanged + failed + omitted = input records`.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/integration/test_import.py -v`

Expected: failures for missing import functions.

- [ ] **Step 3: Implement configuration validation**

Resolve relative paths against the config file directory. Validate package edition, scope, corpus snapshot, source records, graph schema, and output contract. Model credentials remain environment-only and never enter the validated config dump.

- [ ] **Step 4: Implement streaming JSONL import**

Validate each line independently, group document metadata by scoped source/version, preserve exact passage text and supplied hashes, verify calculated text hashes, write errors without converting them to success, and update `source_head` only after a complete source version imports successfully.

- [ ] **Step 5: Create structural nodes and edges**

Upsert document and passage nodes from relational source rows. Give structural edges actor `system`, version `structural-v1`, review state `unreviewed`, and empty evidence references because they describe storage structure rather than scientific claims.

- [ ] **Step 6: Run fixture and full-corpus preflight tests**

Run: `pytest tests/integration/test_import.py -v`

Expected: all fixture tests pass. A read-only preflight over `corpus/records.jsonl` reports 791 valid input lines without changing the database.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/config.py lib/potency_kg/ingest.py lib/potency_kg/graph_store.py tests/integration/test_import.py
git commit -m "feat: import scoped evidence and structural graph"
```

### Task 5: Add Idempotent Embedding Storage and Vector Retrieval

**Files:**
- Create: `lib/potency_kg/embeddings.py`
- Create: `tests/integration/test_embeddings.py`

**Interfaces:**
- Produces: `EmbeddingClient.embed(texts: list[str]) -> EmbeddingBatch`, `embed_active_passages(engine, scope, client, index_version) -> EmbeddingReport`, and `vector_search(engine, scope, query_vector, limit) -> list[PassageHit]`.
- Consumes: completed model-access configuration at runtime; tests use a deterministic fake client.

- [ ] **Step 1: Write embedding-cache tests**

Verify finite vectors, consistent dimensions, no repeated client call for the same model/input hash, a new call after passage text or model changes, strict project filtering, and deterministic ranking from the fake client.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/integration/test_embeddings.py -v`

Expected: missing embedding module.

- [ ] **Step 3: Implement the provider-neutral adapter and cache**

The adapter records actual provider/model IDs, batch size, usage when available, elapsed time, retry count, and sanitized errors. Tenacity retries only declared transient failures with bounded exponential backoff. It never retries authentication, invalid input, or exhausted-budget errors.

- [ ] **Step 4: Implement scoped vector retrieval**

Join through `source_head` so current retrieval cannot mix superseded source versions. Filter tenant and project before ordering by cosine distance. Use exact search for the 791-record POC; do not create a dimension-specific ANN index until model access supplies and verifies the dimension.

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/test_embeddings.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/embeddings.py tests/integration/test_embeddings.py
git commit -m "feat: cache passage embeddings and vector search"
```

### Task 6: Classify Relevance and Extract Structured Semantics Once

**Files:**
- Create: `prompts/relevance-v1.md`
- Create: `prompts/extraction-v1.md`
- Create: `lib/potency_kg/relevance.py`
- Create: `lib/potency_kg/extraction.py`
- Modify: `tests/unit/test_extraction_contract.py`

**Interfaces:**
- Produces: `RelevanceClassifier.classify(passage) -> RelevanceDecision`, `extract_passage(passage, model_text, client) -> PassageExtraction`, and `extract_explicit_alias_patterns(text) -> list[AliasClaimDraft]`.
- The extraction result uses passage-local IDs and never contains global database IDs.

- [ ] **Step 1: Write tests for corpus-wide selection and explicit alias patterns**

Cover page-number-only deterministic skips, multi-label decisions, no question ID in classifier inputs, parenthetical abbreviation extraction, `formerly` relationships, exact supporting spans, and rejection of an alias claim unsupported by the passage.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/unit/test_extraction_contract.py -v`

Expected: missing relevance and extraction functions.

- [ ] **Step 3: Write the versioned relevance prompt**

Ask independent typed questions for semantic extraction, potency strategy, assay, measured attribute, regulatory interaction, proposal/request/revision/conclusion, conflict, and redaction. Jev receives fixed choices or boolean decisions; it does not construct graph records.

- [ ] **Step 4: Write the versioned extraction prompt**

Require exact text spans, local entity IDs, allowed enums, assertion state separate from review state, source-grounded relation quotes, explicit alias relation types (`abbreviation`, `former_name`, `synonym`, `trade_name`, `proper_name`), and no reconstruction of redacted text.

- [ ] **Step 5: Implement one-pass structured extraction**

Use Instructor and Pydantic validation. Compute an input hash from full passage identity, model text, prompt version, extractor version, and model ID. Reuse a completed matching run. Persist invalid responses and errors as failed model runs; do not write partial semantic graph records from invalid output.

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_extraction_contract.py -v`

Expected: all tests pass using fake Jev and generation clients.

- [ ] **Step 7: Commit**

```bash
git add prompts/relevance-v1.md prompts/extraction-v1.md lib/potency_kg/relevance.py lib/potency_kg/extraction.py tests/unit/test_extraction_contract.py
git commit -m "feat: classify and extract passage semantics"
```

### Task 7: Build the Alias Registry and Entity Resolver

**Files:**
- Create: `lib/potency_kg/aliases.py`
- Create: `lib/potency_kg/resolution.py`
- Create: `tests/unit/test_aliases.py`
- Create: `tests/unit/test_resolution.py`

**Interfaces:**
- Produces: `bootstrap_configured_aliases(engine, config)`, `record_alias_candidates(engine, passage, extraction)`, `resolve_mention(engine, scope, mention, context) -> ResolvedEntity`, and `materialize_extraction(engine, passage, extraction) -> MaterializationReport`.
- Alias lookup uses only `configured` and `approved` rows. `pending_review` rows are visible but inactive.

- [ ] **Step 1: Write alias tests**

Verify configuration maps `CASGEVY` and `exagamglogene autotemcel` to `product:casgevy`; an extracted `exa-cel` candidate remains inactive while pending; approving it makes future mentions resolve to the product; rejected candidates remain inspectable; and no automatic merge occurs from embedding similarity.

- [ ] **Step 2: Write resolver tests**

Verify resolution priority: authoritative identifier, configured/approved alias, exact normalized name in compatible context, then new entity. Confirm `STN 125785`, `BLA 125785`, and `application 125785` resolve together, while 125785 and 125787 remain distinct. Confirm generic assay names in different product/application contexts do not merge.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/unit/test_aliases.py tests/unit/test_resolution.py -v`

Expected: missing alias and resolver modules.

- [ ] **Step 4: Implement alias bootstrapping and candidate recording**

Create configured mappings from the case object without claiming human review. Store pattern- and LLM-derived claims as `pending_review` with exact source references, relation type, proposer, extractor version, and quoted evidence. Provide explicit approve/reject functions requiring reviewer identity and timestamp.

- [ ] **Step 5: Implement deterministic normalization and identity strategies**

Normalize with Unicode NFKC, trim and collapse whitespace, lowercase comparison keys, normalize harmless hyphen/space differences, and preserve original observed text. Do not remove scientifically meaningful numbers, units, qualifiers, or application identifiers.

- [ ] **Step 6: Implement two-stage materialization**

First resolve/upsert all mentions and persist passage-local-to-global mappings. Then create assertions and relations using the resolved endpoints. A transaction either commits the complete passage extraction or none of its semantic graph records.

- [ ] **Step 7: Run tests**

Run: `pytest tests/unit/test_aliases.py tests/unit/test_resolution.py -v`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add lib/potency_kg/aliases.py lib/potency_kg/resolution.py tests/unit/test_aliases.py tests/unit/test_resolution.py
git commit -m "feat: resolve aliases and semantic entities"
```

### Task 8: Project the Canonical Graph into Apache AGE

**Files:**
- Create: `lib/potency_kg/age_projection.py`
- Create: `tests/integration/test_age_projection.py`

**Interfaces:**
- Produces: `ensure_scope_graph(connection, scope) -> str`, `sync_scope_graph(engine, scope, graph_version) -> GraphSyncReport`, and `validate_scope_graph(engine, scope) -> GraphValidationReport`.
- Consumes canonical current `kg_node` and `kg_edge` rows only.

- [ ] **Step 1: Write AGE projection tests**

Assert project scopes receive distinct graph names, logical node/edge IDs appear as AGE properties, all projected edges have existing endpoints, rerunning sync does not duplicate records, and project-one traversal cannot return project-two fixture values.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/integration/test_age_projection.py -v`

Expected: missing AGE projection module.

- [ ] **Step 3: Implement safe graph creation**

Generate graph names only through `graph_name_for_scope`. Verify the name against the strict graph-name regex before interpolating it into SQL. Parameterize all property values through AGE's parameter map rather than string-concatenating text into Cypher.

- [ ] **Step 4: Implement idempotent projection**

Load `age`, set `search_path`, create or update vertices by logical `node_id`, then edges by logical `edge_id`. Project source refs and attributes as JSON-compatible properties, keeping large passage text in relational PostgreSQL and storing only lookup/provenance fields in AGE.

- [ ] **Step 5: Implement validation and failure behavior**

Compare expected versus projected node/edge ID sets, detect missing endpoints and cross-scope properties, and mark the sync failed without advancing `graph_scope.last_graph_version` when validation fails.

- [ ] **Step 6: Run tests**

Run: `pytest tests/integration/test_age_projection.py -v`

Expected: all tests pass against the Compose database.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/age_projection.py tests/integration/test_age_projection.py
git commit -m "feat: project scoped evidence graphs into AGE"
```

### Task 9: Implement Vector and Vector-Seeded Graph Retrieval

**Files:**
- Create: `lib/potency_kg/retrieval.py`
- Create: `tests/integration/test_retrieval.py`

**Interfaces:**
- Produces: `retrieve_vector(request: RetrievalRequest) -> RetrievalTrace` and `retrieve_graph(request: RetrievalRequest) -> RetrievalTrace`.
- Graph defaults: top 8 vector passage hits, up to 8 adjacent semantic seed nodes, two directed-or-undirected hops as explicitly recorded, maximum 30 nodes, maximum 50 edges, and final evidence packed under the caller's character budget.

- [ ] **Step 1: Write retrieval tests**

Verify vector mode returns only current scoped passages; graph mode records vector passage hits, semantic seed node IDs, traversed edge IDs, traversal limits, and final passage IDs; graph-selected evidence comes only from traversed source references; and identical fixture identifiers in another project never appear.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/integration/test_retrieval.py -v`

Expected: missing retrieval module.

- [ ] **Step 3: Implement passage-to-seed mapping**

Map vector-hit passage logical IDs to AGE `SourcePassage` vertices. Traverse to supported assertions and adjacent domain entities, rank seed candidates first by best originating vector rank and then by stable node ID, and record the selected seeds before neighborhood expansion.

- [ ] **Step 4: Implement bounded traversal and evidence recovery**

Traverse the declared direction policy and hop count inside the scope graph. Collect evidence references from traversed assertion nodes and evidence-bearing edges. Reload exact current source passages relationally, deduplicate by full passage identity, retain vector scores and graph paths, and pack evidence deterministically.

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/test_retrieval.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/retrieval.py tests/integration/test_retrieval.py
git commit -m "feat: add vector-seeded graph retrieval"
```

### Task 10: Preserve Corrections and Unresolved Conflicts

**Files:**
- Modify: `lib/potency_kg/ingest.py`
- Modify: `lib/potency_kg/resolution.py`
- Create: `tests/integration/test_behavior_cases.py`

**Interfaces:**
- Extends import with explicit replacement metadata that advances `source_head` while preserving the prior version.
- Produces `supersedes` edges only from supplied replacement evidence and `contradicts` edges only when independently supported opposing assertions are detected or reviewed.

- [ ] **Step 1: Write source-correction tests**

Import fixture revision 1, capture its graph IDs, import revision 2, and assert: old source/assertion rows remain; current vector and graph retrieval use revision 2 only; the source head points to revision 2; and history exports can still reproduce revision 1 artifacts.

- [ ] **Step 2: Write unresolved-conflict tests**

Import both independent notes and assert both current assertions and evidence references remain, neither source supersedes the other, the conflict is represented without choosing a winner, and issue date alone does not resolve it.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/integration/test_behavior_cases.py -v`

Expected: correction or conflict assertions fail.

- [ ] **Step 4: Implement source-head changes and semantic state handling**

Advance heads only for explicit replacement relationships. Keep described historical state separate from source-version state. Do not deactivate one independent source merely because another was issued later.

- [ ] **Step 5: Run behavior tests**

Run: `pytest tests/integration/test_behavior_cases.py -v`

Expected: both supplied scenarios pass without special cases keyed to scenario or question IDs.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/ingest.py lib/potency_kg/resolution.py tests/integration/test_behavior_cases.py
git commit -m "feat: preserve source corrections and conflicts"
```

### Task 11: Export Inspectable Artifacts and Reconcile Counts

**Files:**
- Create: `lib/potency_kg/export.py`
- Create: `tests/integration/test_exports.py`

**Interfaces:**
- Produces: `export_import_report(engine, run_id, output_path)`, `export_graph(engine, scope, graph_version, output_dir)`, and `validate_export(output_dir) -> ExportValidationReport`.
- Writes `results/import-report.json`, `results/graph/nodes.jsonl`, and `results/graph/edges.jsonl` without making model calls.

- [ ] **Step 1: Write export tests**

Validate all required node/edge fields, normalized source-reference fields, reconciled import totals, 24 document representations, all valid passages structurally represented, no dangling endpoints, and exact scoped passage IDs in citations.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/integration/test_exports.py -v`

Expected: missing export module.

- [ ] **Step 3: Implement deterministic exports**

Sort nodes by `(scope, type, id)` and edges by `(scope, type, source, target, id)`. Serialize timestamps in UTC ISO 8601, preserve source-reference arrays, and write through temporary files followed by atomic rename so interrupted exports do not look complete.

- [ ] **Step 4: Implement validation**

Validate required fields from the supplied output contract, graph endpoints, extraction coverage per source, omitted/failed reasons, and relational-versus-AGE logical ID counts. Return explicit failures rather than silently dropping invalid rows.

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/test_exports.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lib/potency_kg/export.py tests/integration/test_exports.py
git commit -m "feat: export and validate evidence graph artifacts"
```

### Task 12: Add Reproducible Pipeline Commands and Final Verification

**Files:**
- Create: `lib/potency_kg/cli.py`
- Create: `tests/conftest.py`
- Modify: `docs/readme.md`

**Interfaces:**
- Produces commands `import`, `embed`, `classify`, `extract`, `resolve`, `sync-graph`, `retrieve`, `export`, and `run-pipeline` through `python -m potency_kg.cli`.
- `run-pipeline` stops on failed stages, prints run IDs and counts, and reuses unchanged completed work.

- [ ] **Step 1: Write CLI tests with fake model clients**

Assert stage ordering, nonzero exit status on failed validation, run-ID output, idempotent second execution, and no credential values in logs or artifacts.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `pytest tests -v`

Expected: CLI-specific tests fail because the CLI does not exist; completed earlier tests remain green.

- [ ] **Step 3: Implement commands and dependency injection**

Each command accepts `--config`, explicit run/version options where applicable, and output paths. Model-backed commands check completed model access before doing paid work. Tests inject deterministic clients rather than contacting external services.

- [ ] **Step 4: Document exact commands**

Add setup, migration, staged pipeline, full-pipeline, fixture, export, and test commands to `docs/readme.md`. Explain that the first real model run begins timed assignment work, while smoke access checks follow `MODEL_ACCESS.md`.

- [ ] **Step 5: Run local verification**

Run: `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env build notebook`

Run: `bin/start`

Run: `bin/migrate`

Run: `pytest -v`

Expected: image build, migrations, and all offline tests pass. Tests requiring actual configured model access are marked separately and do not masquerade as completed real runs.

- [ ] **Step 6: Verify full-corpus structural import**

Run:

```bash
python -m potency_kg.cli import --config configs/casgevy-potency.json
python -m potency_kg.cli sync-graph --config configs/casgevy-potency.json --structural-only
python -m potency_kg.cli export --config configs/casgevy-potency.json --output results
```

Expected: 791 passage outcomes reconcile, every supplied source has a document node, a repeated import reports unchanged records, and graph validation reports no missing endpoints.

- [ ] **Step 7: Commit**

```bash
git add lib/potency_kg/cli.py tests/conftest.py docs/readme.md
git commit -m "feat: add reproducible evidence graph pipeline"
```

## End-to-End Data Flow

```text
configs/casgevy-potency.json
        │
        ▼
validate scope and package paths
        │
        ▼
stream corpus/records.jsonl ───────────────┐
        │                                   │
        ├─ source_document/source_head      │ preserve exact source versions
        ├─ source_passage                   │ and immutable source text
        ├─ structural kg_node/kg_edge       │
        └─ redaction mappings               │
        │                                   │
        ▼                                   │
embedding cache + passage vectors           │
        │                                   │
        ▼                                   │
deterministic checks + Jev relevance        │
        │                                   │
        ▼ selected passages                 │
Instructor/Pydantic extraction              │ one generative pass
        │                                   │
        ├─ mentions                         │
        ├─ assertions                       │
        ├─ relations                        │
        └─ explicit alias candidates        │
        │                                   │
        ▼                                   │
configured/approved alias resolution        │
        │                                   │
        ▼                                   │
canonical kg_node/kg_edge + provenance ◄────┘
        │
        ▼
project-isolated Apache AGE projection
        │
        ├─ inspect/export nodes and edges
        └─ vector passage hits → graph seeds → bounded traversal → exact passages
```

## Completion Criteria

- Database migrations apply cleanly from an empty PostgreSQL instance.
- First import and unchanged re-import produce truthful reconciled counts.
- All 791 records are stored or explicitly failed/omitted with reasons.
- Every supplied source and valid passage has structural graph representation.
- Embeddings and extraction calls are skipped when their complete versioned inputs are unchanged.
- Alias extraction and entity resolution require no second generative pass.
- Pending aliases do not merge entities; configured and human-approved aliases do.
- Redacted markers have distinct scoped identities and never become inferred values.
- Semantic assertions from different passages remain separately inspectable.
- AGE projection matches canonical relational logical IDs and has no dangling endpoints.
- Project isolation fixtures cannot leak through SQL, vector search, AGE traversal, exports, or traces.
- Correction fixtures preserve old artifacts while current retrieval uses the replacement version.
- Conflict fixtures preserve and retrieve both opposing claims without inventing resolution.
- Graph retrieval records vector passage hits, graph seeds, traversed edge IDs, limits, and final exact source passages.
- Required graph and import artifacts validate against the supplied output expectations.

