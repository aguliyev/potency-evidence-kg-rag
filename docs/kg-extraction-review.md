# Question-independent extraction review

## Purpose and method

This review checks extraction fidelity independently of any downstream user
question. Use a deterministic, stratified sample spanning guidance, an approval
letter, a multidisciplinary summary, a discipline review, and synthetic fixture
evidence. Do not tune thresholds while reviewing: record omissions and unsupported
additions first, then evaluate policy changes separately against the fixed sample.

For every sampled passage, compare the persisted selected bundles and probabilities,
validated extraction, and materialized relations against the original protected
passage. Quotes must be exact substrings of the original, not paraphrases. A claim or
relation without exact local evidence is an unsupported addition even if it is
generally true.

## Review status

**Current real-extraction review: `not_run`.** No verified real extraction artifact
was available when this form was created. Structural graph output is not a substitute
for model extraction and must not be scored as if extraction occurred. Complete the
rows below only after an explicitly gated real run publishes matching immutable
artifacts; record the graph/extraction versions and artifact path first.

- Review date: not_run
- Reviewer: not_run
- Export target: not_run
- Graph version: not_run
- Extraction version(s): not_run
- Sampling rule: first eligible active passage by stable `(source_id,
  source_version, passage_id)` ordering within each stratum
- Threshold changes during review: none

## Per-passage form

For each row record:

1. Source identity: tenant, project, source ID, source version, passage ID, source
   type, authority, and locator.
2. Jev decision: all seven probabilities, thresholds, selected bundles, response
   version, selection version, model ID, and request ID/usage when supplied.
3. Extraction: extraction run/version, model ID, entity/claim/relation counts, and
   the claims and relationships themselves.
4. Quote accuracy: exact / inexact / absent, with the exact supporting quote and its
   original offsets.
5. Omissions: explicit source-supported content within a selected bundle that the
   extraction missed.
6. Unsupported additions: claims, entities, aliases, or relations not supported by
   the cited passage.
7. Reviewer notes and disposition: acceptable / needs follow-up / invalid.

| Stratum | Deterministic source candidate | Passage | Seven probabilities and selected bundles | Claims / relations and quote accuracy | Omissions | Unsupported additions | Notes / disposition |
|---|---|---|---|---|---|---|---|
| Final guidance | `guidance-cgt-potency-2011` | not_run | not_run | not_run | not_run | not_run | real extraction artifact absent |
| Approval letter | `casgevy-125785-approval-letter` | not_run | not_run | not_run | not_run | not_run | real extraction artifact absent |
| Multidisciplinary summary | `sbra-casgevy-tdt-2024-01` | not_run | not_run | not_run | not_run | not_run | real extraction artifact absent |
| Discipline review | `casgevy-125785-analytical-method-review` | not_run | not_run | not_run | not_run | not_run | real extraction artifact absent |
| Synthetic fixture | `assay-attribute-note` at `fixture-v2` | not_run | not_run | not_run | not_run | not_run | real extraction artifact absent |

## Review rules and follow-up

- Preserve disagreement. Do not choose a winner based only on recency or an invented
  authority hierarchy.
- Review all seven persisted probabilities, including unselected bundles. Inclusive
  threshold boundaries are part of the recorded decision.
- A no-selection result is valid only when no generation call exists. A Jev failure
  must not have a selection or unrestricted extraction.
- Failed or invalid model output must not partially materialize. Verify current
  pointers and provenance against the reviewed extraction run.
- Record observed omissions and unsupported additions without changing thresholds,
  prompts, aliases, or source data during this review.

## Alias review

The alias registry is reviewed independently from extraction quality. Every row is
scoped to the tenant/project being reviewed and has immutable evidence. Inspect the
queue and evidence before deciding:

```bash
bin/runner python -m potency_kg.cli list-aliases \
  --config configs/casgevy-potency.json --status pending
```

Use the scoped immutable `alias_evidence` records associated with the selected alias
ID to inspect its exact quote, passage, detector, input hash, and source/provenance
before making a decision.

Apply these criteria:

- **Pending:** approve only when the exact grounded quote, source passage, and
  proposed synonym support the mapping. The reviewer must provide/confirm all
  required target fields: `entity_type` (`Product` or `Assay`), `canonical_key`, and
  `canonical_name`, plus a reviewer identity. Reject when the evidence is not a
  synonym, is out of scope, or cannot support a complete target; record a concise
  decision note.
- **Approved:** verify that the mapping is exact after normalization and that no
  other approved mapping conflicts on the same scoped typed normalized alias.
  Approved aliases can resolve later exact mentions, but they do not rewrite prior
  extractions, materializations, graph nodes, or exports.
- **Rejected:** retain the decision and all evidence. Later discovery may add
  evidence but does not silently reopen, approve, or delete the row.

Static configuration rows are source-controlled and application-owned. They are
already approved when loaded, cannot be approved or rejected through the database
review command, and retain static evidence when configuration reconciliation retires
them. Review commands are for `review`-managed rows only.

An alias decision changes the scoped approved-catalog version immediately. Treat
that as invalidation, not as a rebuild: inspect the alias status, then explicitly
rerun Jev, extraction, materialization, semantic graph refresh, and export. The
new export is a later immutable version; existing history and `current` remain
unchanged until the validated replacement is promoted.

```bash
bin/runner python -m potency_kg.cli review-alias \
  --config configs/casgevy-potency.json --alias-id <alias-id> \
  --decision approved --reviewer <operator> --entity-type Product \
  --canonical-key casgevy --canonical-name CASGEVY
bin/runner python -m potency_kg.cli inspect-kg --config configs/casgevy-potency.json
```
