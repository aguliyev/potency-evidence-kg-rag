# DeffAI Potency KG / RAG assignment materials

Start with [ASSIGNMENT.md](ASSIGNMENT.md), then complete [MODEL_ACCESS.md](MODEL_ACCESS.md) before the clock starts. Package edition: `2026-09-21-potency-rag-v2`; the frozen source files are retained; corpus snapshot v3 includes one audited heading-order correction.

This folder contains the data and contracts for the exercise. It does not prescribe an implementation stack or include starter application code.

## Contents

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

## Evidence boundary

- Process all records in `corpus/records.jsonl`; do not pre-filter to documents known to answer the questions.
- Treat files under `corpus/raw/` as the source of truth. The JSONL is a mechanical extraction provided for convenience.
- Do not ingest the assignment, questions, schemas, fixture records, or generated outputs as factual FDA evidence.
- Use the raw files to inspect provenance and page context; PDF/XML parsing is not required.
- Use only the supplied corpus for factual answers. URLs are provenance references, not permission to change the frozen snapshot.
- `(b)(4)` denotes public redaction and must not be filled by inference.
- The CASGEVY CMC review memorandum, approval letter, and SBRA are separate documents.
- The configured application is TDT/STN 125785, but the shared exa-cel potency history spans TDT and SCD/STN 125787. Preserve application and indication scope.

The record identity is:

```text
(tenant_id, project_id, source_id, source_version, passage_id)
```

Do not treat `passage_id` alone as globally unique. The fixture inputs intentionally collide on local source and passage identifiers across project scopes.

## Starting the assignment

1. Confirm the archive opens and manifest-listed resources are available. Report any supplied-data mismatch before starting; you are not expected to repair the extraction.
2. Complete the team-provided model access sheet and verify both embedding and generation with tiny test input. A successful local file check is not a successful model-access check.
3. Confirm paid terms, any material clarification, readiness timestamp and submission deadline. The eight-hour active-work clock starts only then; personal model credits are not required.

Implement your own system and generated `results/` alongside these resources. The package contains no starter application. The access template/checklist is onboarding documentation, not a completed implementation. All source records are considered; semantic extraction can be selective under a documented corpus-wide policy. Public source IDs, fixture scope IDs and metadata are inputs, not prewritten answers.

Provide the seven required answers/traces, a compact executed regression evaluation with saved per-case results, and a concise architecture/cost/data-contract design. The evaluator checks decisions and actual mechanisms; graph size and diagram polish do not earn separate credit.

## Confirmed design scope

CASGEVY is the sole main case; P6 still compares both ZYNTEGLO and LYFGENIA. P3 runs vector and graph modes under the 6,000-character evidence cap; each other question needs one mode. The graph schema is adjustable under the documented behavioral contracts.

Run the [source-correction and conflict fixtures](fixtures/BEHAVIOR_CASES.md) in addition to project isolation, and preserve before/after source and graph artifacts with three separate fixture answers/traces. Historical re-query and working human-review operations are not required.
