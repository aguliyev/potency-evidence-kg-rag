# Normalized Entity Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make semantic materialization reuse one entity node for equal normalized names and for approved alias targets, then leave the database ready for replay through `materialize_stage`.

**Architecture:** `resolve_mention()` will first resolve an exact approved typed alias, then an unambiguous approved canonical name, and finally use a deterministic scoped identity based on `(entity_type, normalize_alias(mention.text))`. Provenance remains mention- and extraction-specific through the existing reference tables. Materialization resolves against the current approved alias catalog, while extraction and Jev catalog versions remain immutable provenance. Each materialization records the resolution catalog version it used, so alias approvals can be replayed without rerunning extraction. After verification, the materialization-owned relational rows will be cleared and the active AGE graph will be refreshed from the cleaned relational snapshot.

**Tech Stack:** Python, SQLAlchemy, PostgreSQL, Apache AGE, pytest, repository graphify tooling.

**Spec:** `docs/superpowers/plans/2026-10-11-evidence-data-knowledge-graph.md`

## Global Constraints

- Approved aliases are the only alias mappings used for semantic identity.
- Alias resolution remains typed and scope-bound.
- Ambiguous approved canonical-name matches must not be guessed.
- Structural document and passage data must survive materialization cleanup.
- Historical extraction and Jev catalog snapshots must remain valid provenance and must not block replay with a newer approved alias catalog.
- Existing user changes outside this feature must be preserved.

---

### Task 1: Add failing identity-resolution regression tests

**Files:**
- Modify: `tests/integration/test_materialization.py`
- Test: `tests/integration/test_materialization.py`

**Interfaces:**
- Consumes: `resolve_mention()`, `AliasCatalog`, `EntityMentionDraft`.
- Produces: executable cases for normalized-name reuse and approved canonical-name reuse.

- [x] **Step 1: Write the failing tests**

Add tests that resolve two same-typed mentions such as `"New Product"` and `" new   product "` and assert identical `node_id` and normalized `canonical_key`. Add an approved alias whose `canonical_name` is not itself an alias value, resolve a mention using that canonical name, and assert it uses the approved canonical target node.

- [x] **Step 2: Run the focused tests**

Run:

```bash
bin/runner pytest -q tests/integration/test_materialization.py -k "normalized_name or canonical_name"
```

Expected: FAIL because unmatched mentions currently include passage, extraction, and local IDs in their node identity, and canonical names are not searched in the alias catalog.

### Task 2: Implement deterministic materialization identity

**Files:**
- Modify: `lib/potency_kg/resolution.py:164-254`
- Test: `tests/integration/test_materialization.py`

**Interfaces:**
- Consumes: approved `AliasCatalog` entries and `normalize_alias()`.
- Produces: stable `ResolvedEntity` IDs for approved aliases, approved canonical names, and normalized unmatched names.

- [x] **Step 1: Add the smallest resolver change**

In `resolve_mention()`:

1. Keep exact typed `normalized_alias` lookup first.
2. If there is no alias match, collect typed catalog entries whose normalized `canonical_name` equals the mention’s normalized text; use the target only when exactly one entry exists.
3. For an unresolved mention, derive the node ID from `("entity", scope, mention.entity_type, normalized_name)` instead of source/extraction/local identity.
4. Store `normalized_name` and the deterministic normalized-name key in node attributes while retaining extraction provenance attributes.

- [x] **Step 2: Run focused tests**

Run:

```bash
bin/runner pytest -q tests/integration/test_materialization.py -k "normalized_name or canonical_name or resolution_uses_only_exact"
```

Expected: PASS.

- [x] **Step 3: Run the full materialization suite**

Run:

```bash
bin/runner pytest -q tests/integration/test_materialization.py
```

Expected: PASS with no changed pointer/provenance behavior.

### Task 3: Refresh repository metadata and verify the change

**Files:**
- Modify: `graphify-out/` generated metadata as produced by graphify.

- [x] **Step 1: Update the code graph**

Run:

```bash
graphify update .
```

- [x] **Step 2: Run regression checks**

Run:

```bash
git diff --check
bin/runner pytest -q tests/integration/test_materialization.py
```

Expected: clean diff check and passing materialization tests.

### Task 4: Clear materialization-owned data and rebuild the active graph snapshot

**Files:**
- Runtime database scope: `public-fda/casgevy-potency` from `configs/casgevy-potency.json`.
- Existing cleanup path: `lib/potency_kg/cli.py:404-448`.

- [x] **Step 1: Record pre-cleanup counts**

Read counts for materialization-owned relational rows and the active AGE graph.

- [x] **Step 2: Clear materialization outputs**

Run the existing guarded `clear_materialization_stage` path for the configured scope. This preserves documents, passages, extraction responses, validation state, and alias-discovery evidence.

- [x] **Step 3: Refresh the active graph**

Run a forced graph refresh from the cleaned relational snapshot so the old active AGE graph is replaced atomically and no stale materialized vertices remain.

- [x] **Step 4: Verify readiness for replay**

Confirm materialization-owned tables are empty, passages with validated extraction are back in `selected` state, and the active graph contains only the remaining structural/system nodes.

### Task 5: Make alias catalog changes replayable without extraction

**Files:**
- Modify: `lib/potency_kg/resolution.py`
- Modify: `lib/potency_kg/cli.py`
- Modify: `tests/integration/test_materialization.py`
- Add: `migrations/versions/0007_resolution_catalog.py`

- [x] **Step 1: Record the catalog used for resolution**

Add `materialization_run.resolution_alias_catalog_version` and populate it from the current approved catalog, including failed materialization records.

- [x] **Step 2: Remove current-catalog gating from pending selection**

Keep extraction/Jev self-consistency checks, but allow validated historical extraction rows to be selected for materialization regardless of the catalog version stored on those upstream rows.

- [x] **Step 3: Add replay regression coverage**

Verify that approving aliases after extraction makes the existing extraction rows pending again, that direct materialization uses the new aliases, and that malformed or internally inconsistent extraction/Jev identity is still rejected.

- [x] **Step 4: Materialize and refresh the live scope**

Run `materialize_stage` and the semantic graph refresh without invoking `extract_stage`. Verify that all 746 existing extraction rows materialize and that the semantic graph validates against the resulting relational snapshot.
