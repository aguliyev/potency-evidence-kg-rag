regarding # Scoped Alias Management and Discovery Design

## Purpose

Replace the split between hard-coded trusted aliases and extraction-owned
`alias_candidate` rows with one tenant/project-scoped alias registry. Populate the
registry from authoritative JSON seeds and auditable corpus discovery, expose human
review as an explicit stage, and allow only approved aliases to influence entity
resolution.

This design adds alias preparation between structural import and the model-backed KG
stages. It does not add a review UI, automatic approval, fuzzy entity merging, or
automatic retroactive rematerialization.

## Goals

- Store alias decisions in PostgreSQL under an exact `(tenant_id, project_id)` scope.
- Support typed aliases, initially for `Product` and `Assay`.
- Load known aliases from a version-controlled JSON file instead of Python literals.
- Preserve immutable evidence from static configuration, deterministic rules,
  extraction, and optional LLM discovery.
- Aggregate repeated evidence under one proposed alias mapping.
- Make static loading, structural import, and discovery safe to rerun.
- Keep deterministic discovery available without credentials.
- Keep LLM discovery optional, separately callable, and non-blocking.
- Make approved-alias changes invalidate semantic reuse without silently rebuilding
  the graph.
- Present every workflow stage separately, with explanations and inspection cells, in
  the build notebook.

## Non-goals

- No global or cross-project aliases.
- No automatic approval based on confidence.
- No fuzzy or embedding-based identity merge.
- No browser review application.
- No automatic correction of an already materialized graph after an approval.
- No deletion or rewriting of immutable extraction, graph, or export history.

## Architectural decisions

### Scope

Every alias and every evidence row belongs to exactly one tenant/project. A single
JSON file may contain entries for many scopes, but no row is inherited by another
scope. All primary keys, foreign keys, queries, unique indexes, and mutation APIs
include `tenant_id` and `project_id`.

### Registry plus evidence

The design uses two conceptual layers:

- `aliases` is the current reviewable mapping and decision.
- `alias_evidence` contains immutable observations supporting that mapping.

A repeated observation adds evidence instead of another review item. Conflicting
targets may coexist while pending, but only one approved target may exist for the
same normalized alias and entity type in a scope.

### Static seeds are authoritative

Static aliases are application-owned configuration. They load as approved, are
managed through source control, and cannot be approved or rejected through the
database review API. Reloading unchanged seed data is a no-op. A reviewed change or
removal in JSON is reconciled explicitly without deleting prior evidence. On removal,
the loader marks the former static row rejected with
`decided_by='static_config'` and a configuration-removal note. On a target change,
the loader retires the old mapping and creates the new approved mapping in one
transaction.

### Discovery never approves

Deterministic and LLM discovery only produce pending proposals and evidence. A
confidence value helps review prioritization but cannot change status.

### Approval affects subsequent semantic runs

Approving or rejecting an alias changes the catalog immediately. It does not modify
the currently active materialized graph. The next explicitly requested semantic run
uses the new catalog, produces new derived versions, materializes a new relational
state, refreshes a shadow AGE graph, and publishes a new immutable export.

## Relational model

Alembic migration `0005` replaces `alias_candidate` with the following tables.

### `aliases`

Required columns:

- `tenant_id TEXT NOT NULL`
- `project_id TEXT NOT NULL`
- `alias_id TEXT NOT NULL`
- `alias_value TEXT NOT NULL`
- `normalized_alias TEXT NOT NULL`
- `entity_type TEXT NULL`
- `canonical_key TEXT NULL`
- `canonical_name TEXT NULL`
- `status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected'))`
- `managed_by TEXT NOT NULL CHECK (managed_by IN ('static_config','review'))`
- `normalization_version TEXT NOT NULL`
- `confidence DOUBLE PRECISION NULL CHECK (confidence BETWEEN 0 AND 1)`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`
- `decided_at TIMESTAMPTZ NULL`
- `decided_by TEXT NULL`
- `decision_note TEXT NULL`

The primary key is `(tenant_id, project_id, alias_id)`. `alias_id` is an immutable
identifier derived when the proposal is first created from the full scope,
normalized alias, and the proposal's initial target fields, using explicit null
sentinels. Supplying or correcting target fields during review never changes the
primary key. Discovery first searches for an equivalent scoped mapping; when no
target was proposed, it aggregates under the existing unresolved row for the same
normalized alias instead of creating one row per occurrence.

Database constraints enforce:

- trimmed, nonempty alias values and normalized aliases;
- supported non-null entity types (`Product` and `Assay` in this version);
- approved rows have nonempty `entity_type`, `canonical_key`, and `canonical_name`;
- rejected rows have `decided_at` and `decided_by`;
- active static rows are approved and have complete targets;
- retired static rows are rejected, retain their complete historical target, and
  carry a loader-owned decision audit;
- only one approved row exists for
  `(tenant_id, project_id, entity_type, normalized_alias)` through a partial unique
  index.

Pending rows may have an incomplete target. Approval requires the reviewer to supply
or confirm all target fields. A rejected proposal is not automatically reopened by a
later discovery run; new evidence attaches to it and its decision remains intact.

### `alias_evidence`

Required columns:

- `tenant_id TEXT NOT NULL`
- `project_id TEXT NOT NULL`
- `evidence_id TEXT NOT NULL`
- `alias_id TEXT NOT NULL`
- `source_type TEXT NOT NULL CHECK (source_type IN
  ('static_config','deterministic_rule','llm_mined','extraction'))`
- nullable `source_id`, `source_version`, and `passage_id`
- nullable `extraction_run_id`
- nullable `discovery_run_id`
- nullable `exact_quote`, `original_start`, and `original_end`
- `detector_name TEXT NOT NULL`
- `detector_version TEXT NOT NULL`
- nullable `model`, `prompt_hash`, and `schema_hash`
- `input_sha256 TEXT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`

The primary key is `(tenant_id, project_id, evidence_id)`. The evidence identifier is
derived from the proposed mapping, source identity, detector identity, exact quote or
configuration entry, and input hash. Identical reruns therefore conflict harmlessly
and create no duplicate evidence.

Corpus-backed evidence has a composite foreign key to `source_passage`. Extraction
evidence additionally has the existing fully scoped extraction foreign key. Static
evidence has no passage identity and records the configuration file digest and entry
identity instead.

### `alias_discovery_run`

Required columns:

- scoped `discovery_run_id UUID` identity;
- `method TEXT CHECK (method IN ('deterministic','llm'))`;
- source document identity and complete document input hash;
- detector version;
- nullable provider model, sanitized origin, credential environment name, prompt
  hash, and schema hash;
- status (`running`, `succeeded`, or `failed`), usage, attempts, sanitized error
  detail, counts, and timestamps.

A successful run is reusable only when its complete method/input/detector/model/prompt
identity matches. A failed LLM run is auditable but does not block downstream stages.

### Legacy migration

Each existing `alias_candidate` becomes a pending `aliases` row with
`managed_by='review'`. Its extraction provenance becomes an `alias_evidence` row.
Because the old database row did not reliably preserve the candidate's canonical
target, migration must leave target fields unset rather than infer one. After counts
and foreign keys are validated, `alias_candidate` is dropped.

The downgrade recreates the legacy table and copies representable extraction
evidence back. Canonical-target and multi-evidence information that the legacy schema
cannot represent is deliberately not fabricated.

## Static alias configuration

Create `lib/potency_kg/config_data/scoped_aliases.json` with a top-level
`schema_version` and an `entries` array. Every entry explicitly contains:

```json
{
  "tenant_id": "public-fda",
  "project_id": "casgevy-potency",
  "entity_type": "Product",
  "canonical_key": "casgevy",
  "canonical_name": "CASGEVY",
  "aliases": ["CASGEVY", "exagamglogene autotemcel", "exa-cel"]
}
```

The loader validates the complete file before opening its write transaction. It uses
deterministic alias and evidence identifiers, inserts missing rows, confirms existing
static rows are byte-for-byte equivalent after canonical normalization, and refuses
silent target reassignment. A removed mapping is retired as described above; a
changed mapping is an atomic retire-and-add operation. Database review operations
reject every `managed_by='static_config'` row, including retired rows.

`project_profiles.py` retains descriptor and application metadata but contains no
alias literals. Profile construction incorporates the database-derived approved alias
catalog version and snapshot supplied by the stage owner.

## Deterministic discovery

Discovery runs over active, complete source-document versions after structural
import. Passages are ordered deterministically and retain their own provenance even
when rules examine neighboring passage text.

The first detector version implements conservative rules:

1. Long form followed by a short form in parentheses.
2. Short form followed by a long form in parentheses.
3. Explicit phrases: `also known as`, `abbreviated as`, and `formerly called`.
4. Acronym validation using ordered character alignment between the short form and
   candidate long form.

Rules reject empty strings, page furniture, numeric-only forms, equal normalized
values, excessive length ratios, and candidates without exact source offsets. They
do not use edit distance, embeddings, or nearest-neighbor matching.

The detector may leave `entity_type` and canonical target incomplete when text alone
does not establish them. Every result remains pending. Re-running structural import
does not duplicate source rows, and re-running deterministic discovery does not
duplicate discovery runs, alias mappings, or evidence.

## Optional LLM discovery

`alias_discovery.py` defines:

```python
DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY = False
```

The notebook invokes LLM discovery in its own cell and passes this constant
explicitly. The composite CLI workflow uses the same default. No environment value
silently enables the stage.

LLM discovery reuses the existing generation-provider settings and credential
boundary, but has a dedicated prompt, typed response model, schema hash, and discovery
version. It processes one source document at a time. Documents exceeding the request
budget are split into deterministic overlapping passage windows; evidence always
points back to the original passage and exact quote.

The typed response contains proposed alias value, proposed canonical name, optional
entity type, exact supporting quote, passage identifier, and confidence. Validation
requires exact quote grounding and rejects unsupported entity types, missing
passages, and offset mismatches before any proposal is persisted.

Provider failures, invalid responses, and retry exhaustion produce failed discovery
runs with sanitized diagnostics. They do not change alias decisions and do not stop
the user from proceeding to embeddings or Jev.

## Review operations

The library provides scoped operations to:

- list aliases by status, entity type, source type, and canonical target;
- inspect all evidence for one alias;
- approve a pending or rejected review-managed alias with a complete typed target;
- reject a pending or approved review-managed alias with an optional note;
- report the current approved catalog version and whether semantic state is stale.

Every mutation locks the scoped alias row, validates the requested state transition,
sets decision audit fields, and relies on the approved-mapping unique index to reject
conflicts. Static rows cannot be mutated through these operations. Repeating the same
decision with the same target and audit values is idempotent; a contradictory repeat
fails explicitly.

CLI commands expose these library seams without adding a UI. Secret values and raw
provider responses are never emitted.

## Catalog versioning and semantic consistency

The approved alias snapshot is the canonical ordered list of scoped rows containing
alias ID, normalized alias, entity type, canonical key, canonical name,
normalization version, and authoritative source identity. Its canonical JSON digest
is `alias_catalog_version`.

Jev response identity, extraction identity, and persisted profile snapshots include
this version and exact approved aliases. Reuse is permitted only when the current
catalog version matches. Approval after Jev or extraction makes that semantic state
stale.

Materialization opens one transaction, locks the passage processing row, loads only
approved aliases for the exact scope, recomputes the catalog version, and requires it
to match the validated extraction run. It then resolves exact normalized aliases for
`Product` and `Assay`. Pending and rejected aliases never merge identity.

An approval does not mutate existing KG nodes or the active graph. The stale-state
report directs the operator to rerun Jev, extraction, materialization, shadow refresh,
and export explicitly.

## Pipeline and notebook

The workflow stage order is:

1. Preflight.
2. Structural import.
3. Authoritative static-alias load/reconciliation.
4. Deterministic alias discovery.
5. Optional LLM alias discovery.
6. Alias inspection and explicit review.
7. Embeddings.
8. Jev classification.
9. Structured extraction.
10. Materialization.
11. Shadow graph refresh and activation.
12. Immutable export.
13. Read-only inspection and optional retrieval smoke.

`notebooks/01_build_knowledge_graph.ipynb` exposes each stage as a separate call. Each
stage has a preceding Markdown explanation covering inputs, outputs, credentials,
expected database effects, idempotent rerun behavior, and invalidation rules. Each
write stage is followed by a read-only inspection cell showing scoped counts,
versions, statuses, and sanitized failure summaries. The notebook never owns pipeline
logic; it imports the same functions used by the CLI.

The existing `run-kg` command remains a convenience orchestrator using these exact
stage functions. Static loading and deterministic discovery run normally. LLM alias
discovery runs only when the module constant is explicitly passed as enabled by the
caller.

## Error handling and safety

- Validate seed files and model responses completely before writes.
- Use one transaction per seed reconciliation, review decision, or document discovery
  result.
- Preserve earlier successful runs when later documents fail.
- Store sanitized provider metadata and errors only.
- Never log credentials, raw authorization headers, or unvalidated provider payloads.
- Keep supplied `configs/`, `corpus/`, `fixtures/`, and `schema/` read-only.
- Preserve tenant/project scope on every read, write, lock, and foreign key.

## Verification criteria

The implementation is complete when automated tests prove:

- migration preserves every legacy candidate as a pending proposal with extraction
  evidence and no invented target;
- scoped rows and foreign keys reject cross-project access;
- static seeds load approved and unchanged reruns produce no mutations or duplicates;
- static rows reject database review mutations;
- structural import remains idempotent before and after alias stages;
- deterministic discovery recognizes the specified patterns and rejects documented
  false positives;
- identical discovery reruns reuse successful runs and evidence;
- repeated observations aggregate evidence under one proposal;
- conflicting proposals may remain pending but cannot both become approved for the
  same typed normalized alias;
- human decisions survive discovery reruns;
- LLM discovery is disabled by default and its success, invalid response, retry, and
  sanitized failure paths are covered with test doubles;
- approved aliases resolve exact normalized Product and Assay mentions while pending
  and rejected aliases do not;
- an alias catalog change prevents reuse or materialization of stale semantic runs;
- the build notebook contains separate, ordered, explained, inspectable stage calls;
- the complete offline suite passes without external model calls;
- explicitly gated real-model tests remain separate and truthfully report `not_run`
  when provider access is unavailable.
