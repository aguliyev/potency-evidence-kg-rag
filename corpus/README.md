# Frozen official-source corpus

This corpus contains 24 original public FDA/eCFR documents and 791 mechanically extracted passages under snapshot `potency-assignment-2026-09-21-v3`.

## Source of truth

Files under `raw/` are authoritative for this exercise:

- `raw/regulatory/`: FDA guidance and policy PDFs plus dated official eCFR XML snapshots.
- `raw/precedent/`: official ZYNTEGLO and LYFGENIA SBRAs.
- `raw/casgevy-125785/`: the TDT approval letter, SBRA, and supplied public approval-history files.
- `raw/casgevy-125787/`: the SCD approval letter and SBRA needed to scope the shared exa-cel strategy.

`records.jsonl` is a convenience extraction so candidates do not need to build PDF or XML parsing. It was generated mechanically from the raw documents without regulatory summaries or added interpretation. PDFs were extracted by physical page; long pages were split at approximately 4,500 characters. The two relevant CFR sections were extracted from dated eCFR XML.

If raw and normalized content appear to disagree, use the raw source and report the discrepancy.

## Coverage expectation

The entire supplied corpus is in scope. Import outcomes must account for every record and every document must have a `SourceDocument` node. Semantic extraction depth may vary with relevance when the selection method and deliberate omissions are reported, but candidates should not pre-filter to a known answer set.

The configured application is TDT/STN 125785. The product-level potency history also includes SCD/STN 125787. Preserve which evidence applies to the product generally, one application, one indication, or both.

## Record contract

Each record contains scope, snapshot/layer, source and passage IDs, source version, authority/status, dates, URL, subject scope, physical page or CFR section, exact extracted text, and text SHA-256.

The record identity is:

```text
(tenant_id, project_id, source_id, source_version, passage_id)
```

Do not use `passage_id` alone as a global key. The isolation fixtures deliberately collide on local source and passage identifiers.

Redaction markers such as `(b)(4)` are evidence that content was withheld from the public document. Do not infer the hidden content.

## Normalization correction in this variant

Snapshot v3 corrects the reading order of the `FDA LOT RELEASE` heading on physical page 2 of the supplied SCD approval letter. The raw PDF was visually verified; no words or regulatory interpretation were added. The other 790 passage texts and all 24 raw files are unchanged. See [corrections.json](corrections.json) for IDs and hashes. Use the supplied current record text for verbatim citations and the raw page for layout context.
