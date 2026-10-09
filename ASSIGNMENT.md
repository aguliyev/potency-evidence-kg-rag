# DeffAI Paid Take-Home: Build a Potency Knowledge Graph and RAG Evidence System

**Role:** Founding Engineer  
**Edition:** Potency KG / RAG variant, September 21, 2026 (`2026-09-21-potency-rag-v2`)  
**Active work time:** 8 hours  
**Delivery window:** 14 calendar days after complete materials and working model access are provided.  
**Compensation:** Paid; amount and engagement terms are confirmed before the clock starts.  
**Submission:** Private repository or source archive containing your code, generated results, and design documents.

Stop after eight hours and submit the strongest coherent version you have. Incomplete work is acceptable when it is clearly identified.

AI coding and research tools are allowed. We care about the decisions you make, the system you produce, and how you verify AI-generated work. Record meaningful AI use in your README.

## Goal: build a working potency KG and RAG system

Build a working **Retrieval-Augmented Generation (RAG) system backed by a potency knowledge graph (KG)** for reviewing the supplied cell and gene therapy evidence. A reviewer should be able to follow a substantive answer through the retrieval trace and graph to an exact source passage. The graph must participate in evidence selection.

Use one reusable import/extraction/retrieval pipeline driven by scope and source metadata; do not implement a CASGEVY-specific answer engine. This variant focuses on potency evidence and its review history. CASGEVY is the sole main case. ZYNTEGLO and LYFGENIA remain comparison evidence for P6; no independent second-product run is required. A general client profile, personalized playbook, and client action roadmap are outside this variant.

Use the supplied FDA corpus to answer:

> What potency strategy was expected, proposed, reviewed, changed, and ultimately accepted; what evidence supports those conclusions; and what remains unknown or redacted?

Your submission should demonstrate three connected capabilities:

1. A useful evidence graph grounded in source passages.
2. Retrieval, answers, and evaluation over that graph and corpus.
3. A credible, low-cost production architecture for taking the prototype forward.

This is an engineering exercise, not a regulatory examination. Use only the supplied corpus for factual answers. Do not infer confidential content hidden by FDA redactions or treat one product approval as a universal requirement.

## Supplied materials

```text
candidate-materials/
  ASSIGNMENT.md
  README.md
  MODEL_ACCESS.md
  configs/model-access.template.json
  corpus/
    manifest.json
    records.jsonl
    raw/
      regulatory/                   # official FDA guidance/policy and dated eCFR XML
      precedent/                    # official ZYNTEGLO and LYFGENIA SBRAs
      casgevy-125785/               # supplied public TDT approval materials
      casgevy-125787/               # SCD approval letter and SBRA
  configs/casgevy-potency.json
  questions.json
  schema/
    graph-core.json
    output-contract.json
  fixtures/                         # synthetic isolation, correction and conflict cases
```

The package contains 24 original public documents and 791 mechanically normalized passages. `corpus/records.jsonl` is provided so the exercise is not a PDF-parsing test. Stable IDs, source versions, authority/status metadata, hashes, and page or section locators are supplied.

The files under `corpus/raw/` are the source of truth. The normalized records were produced directly from those files without adding regulatory interpretation. If a normalized passage and its raw source appear to disagree, use the raw source and report the discrepancy.

Writing PDF parsing or OCR is outside scope. You may use any language, libraries, models, and storage technologies. DeffAI will provide approved model access or agree on a local alternative before the timed work begins. You must not use personal paid model credits.

### Before the eight-hour clock starts

Read [MODEL_ACCESS.md](MODEL_ACCESS.md). DeffAI supplies the provider/endpoint and API format, a secure credential-delivery method, exact embedding and generation model IDs, a usage budget, rate/concurrency and request-size limits, structured-output support or its absence, an approved local fallback, and a named access-failure contact. The access sheet must state explicit limits or confirmed absence of limits; unset is not the same as unlimited.

Timed work starts only after complete materials and paid terms are available and you have successfully verified **both embedding and generation** with the agreed setup. Record this start time together. The two smoke calls, credential troubleshooting, and any agreed local-model download/setup needed to establish access are outside the eight hours. Use tiny non-sensitive test input; indexing this corpus, implementation, and evaluation are timed work.

If team-provided access fails during the exercise, record the error/time and contact the named person. Pause active-time accounting while blocked, or count only other exercise work you choose to continue. Resume after access is restored or an agreed alternative passes both checks. Do not spend personal credits or silently change models. The 14-day submission window starts at readiness and is extended for a confirmed team-provided access outage when needed.

The primary case materials are:

| Material | Official source |
|---|---|
| CASGEVY TDT approval letter, STN 125785 | <https://www.fda.gov/media/175482/download> |
| CASGEVY TDT Summary Basis for Regulatory Action | <https://www.fda.gov/media/175842/download> |
| CASGEVY TDT approval-history package | <https://www.fda.gov/media/176033/download?attachment=> |
| CASGEVY SCD approval letter, STN 125787 | <https://www.fda.gov/media/174618/download> |
| CASGEVY SCD Summary Basis for Regulatory Action | <https://www.fda.gov/media/175179/download> |
| ZYNTEGLO Summary Basis for Regulatory Action | <https://www.fda.gov/media/161472/download> |
| LYFGENIA Summary Basis for Regulatory Action | <https://www.fda.gov/media/175250/download> |
| CBER CMC BLA Review Memorandum | Included in the approval-history package and supplied corpus |

The CMC review memorandum is not the approval letter. Preserve that distinction.

The configured application is the TDT application, STN 125785. The product's potency-development and review history spans both the SCD and TDT indications. Model that shared exa-cel strategy while preserving which statements, assays, interactions, and agency actions apply to STN 125785, STN 125787, one indication, or both.

## Required work

### 1. Import and preserve evidence

Import the normalized records into a persistent local representation of your choice.

Preserve source and passage identity, scope, version, exact text, locator, authority, status, corpus snapshot, and content hash. An unchanged import after restart must not duplicate records or repeat paid embedding work.

Report imported, unchanged, failed, and deliberately omitted records with reconciled totals. All 791 records must be loaded and considered for indexing; a complete import indexes every valid record. Report partial processing as incomplete, with reasons. Keep ingestion/indexing counts separate from semantic-extraction coverage. Do not silently truncate input or turn failed processing into apparent success.

Run the supplied isolation fixture in the same store. It reuses document and passage identifiers across two project scopes. Evidence from one project must not appear in the other project's retrieval context, graph traversal, citations, or answers.

### 2. Build an inspectable evidence graph

Use the supplied graph core as a reference, not a fixed ontology. You may modify, merge, or replace its node and relationship types; document consequential choices and map your representation to the required evidence provenance, tenant/project isolation, source-version identity, assertion state, and human-review state. Equivalent representations are acceptable. Every source must still have an inspectable document representation, even if you rename SourceDocument. Do not treat the listed type names as a checklist.

Process the **entire supplied corpus**: every source must have a `SourceDocument` node or documented equivalent, and ingestion outcomes must account for every normalized record. Semantic entity, assertion, and relationship extraction may vary by relevance; document the selection method and any deliberate omissions. Structural document/passages coverage can be deterministic, and semantic extraction may use reusable rules, batched model calls, or a corpus-wide relevance pass. An LLM call for every passage and exhaustive extraction are not required. Report per-source extraction coverage and reasons for skipped semantic work. Do not pre-filter the corpus to documents already known to answer P1–P6 or use question IDs to select extraction rules.

The graph should represent, where relevant:

- Source documents and passages.
- Product, application, and indication.
- Potency strategy, assay, measured attribute, and assay role.
- Regulatory interaction, assertion, and review conclusion.
- Relationships such as proposed, requested, revised, supported, and concluded.

Preserve important distinctions:

- Regulation, final guidance, draft guidance, applicant statements, reviewer conclusions, and final agency actions.
- Proposed, requested, revised, accepted, rejected, unresolved, and redacted information.
- Different roles played by the same assay or attribute.
- Historical proposals versus later conclusions.

Every evidence-bearing assertion or relationship must retain stable scoped identity, source-passage references, extraction actor/version, timestamp, and review state. A document-level status of final_public_record does not make every quoted applicant proposal or historical reviewer position final. Preserve assertion-level status and, when supported, event dates separately from file retrieval time. Redacted names do not identify a single common assay: use passage-scoped placeholders rather than merging all `(b)(4)` mentions. Structural edges identify themselves as system-generated; they do not establish scientific claims. 

Automatically extracted assertions and relationships default to unreviewed. A model call or automated validation is not human review. Keep this independent of the regulatory state reported in the source: an accepted proposal or a reviewer conclusion may still be an unreviewed extraction. Human review and correction are production-design topics only; no working review operation or UI is required. Explain who can review/correct an extraction, what provenance is retained, and how an approved correction would reach indexes and later answers.

Export the graph in an inspectable documented format. A graph database or visualization is optional.

### 3. Retrieve and answer from evidence

Implement two retrieval modes:

- `vector`: semantic retrieval over source passages.
- `graph`: select seeds, traverse a bounded neighborhood, and use the traversal to select original source passages. Combining graph-selected and vector-selected evidence is allowed.

Answer the six questions in `questions.json`. Run both modes for P3 using the same corpus snapshot, generation model, and a maximum of 6,000 characters of source evidence per mode. One chosen mode is sufficient for P1, P2, P4, P5, and P6. This produces seven answer records. The 6,000-character P3 cap includes **all source-derived content supplied to generation**, including summaries or graph-derived descriptions; do not hide additional evidence in another prompt field. Count Unicode code points (equivalent to Python `len`), keep the instruction prompt and evidence selection policy comparable, and declare traversal direction/hop counting and seed/neighbor limits. Graph retrieval need not outperform vector retrieval; explain the observed contribution or failure.

For every answer, retain selected passage IDs, graph seeds and traversed edge IDs where applicable, exact evidence sent to the model, system versions, elapsed time, provider usage when available, and exact citations.

If support is absent, say what was not found in the inspected scope. Retrieval absence is not proof that something never existed.

### 3a. Demonstrate source correction and unresolved conflict

Run both scenarios in [fixtures/behavior-cases.json](fixtures/behavior-cases.json), following [fixtures/BEHAVIOR_CASES.md](fixtures/BEHAVIOR_CASES.md). They use four clearly fictional records in isolated test scopes, separate from the 791 official passages. Use the same reusable import/extraction/retrieval pipeline as the main case; do not implement answers or extraction branches keyed to these question IDs, fixture names, or expected labels.

- **Source correction:** import revision 1, answer the supplied question, apply the explicit replacement by revision 2, then answer again. Keep the earlier evidence and run artifacts inspectable. The current graph/evidence and new answer must use the corrected revision without mixing it with the superseded assertion. A full rebuild with preserved prior exports is acceptable; no incremental-update algorithm is required.
- **Unresolved conflict:** import both notes about the same fictional assay, plan and effective date. They have equal authority and no declared replacement relationship. Retrieve and cite both opposing claims. If the evidence does not resolve the disagreement, state that the conflict remains unresolved; the newer issue date alone does not settle it.

Use one chosen retrieval mode for each scenario and the same mode before/after the correction. Save three fixture answers and their actual generation traces separately from the seven main answers. Retain the before/after graph and source versions, report the checks and failures, and provide reproducible commands. Test doubles may support offline checks; identify them separately from these real embedding/generation runs. These are visible acceptance cases, not hidden tests or a candidate-authored holdout. They may be included in regression reporting if their supplied/visible origin remains explicit; they do not replace the candidate-authored evaluation cases.

Preserve inspectable old source records, assertions/graph exports, and the answers and traces generated before an update, connected by source, snapshot, index and run identifiers. Files are sufficient. No old-version or as-of retrieval/generation interface is required. Current source versions may still describe historical proposals: source revision state and the chronology described by a source are separate.

### 4. Design and run a compact evaluation

Build an evaluation that you would trust as an initial regression check after changing the model, prompt, ontology, extraction, or index.

Submit machine-readable cases, runnable scoring code or commands, results from your system, case-selection rationale, and a brief failure analysis. Evaluate retrieval separately from generated answers.

The six assignment questions alone are not a sufficient production evaluation. Create a small regression set, freeze a subset before the final run, and explain the limitations of candidate-authored labels. Explain when and how the holdout subset was frozen; no particular mechanism or minimum size is prescribed. We care more about what you decide is important to test and whether the checks work than about case count. Record enough provenance and per-case detail to reproduce reported metrics, including saved `evaluation/results.jsonl` and the commands that recompute aggregates **without new model calls**. Report failed, empty, and unscorable cases with explicit denominators; a truthful unavailable metric is preferable to a misleading success score. Keep candidate-authored development and holdout cases separate, and do not tune on the frozen holdout after the final run without disclosing it.

For eight-hour prioritization, a small set of roughly four to six additional cases is a useful target, not a mandatory quota. Aim to exercise distinct risks such as insufficient evidence, a misleading premise, redaction, authority/status, or scope. Select and implement a few meaningful checks rather than proposing a large unexecuted evaluation suite. Candidate-authored holdout results are a regression demonstration, not an unbiased production-accuracy estimate. Do not present an automated judge as regulatory expert confirmation.

### 5. Design the production architecture

Design how you would productionize your prototype on AWS for this pilot:

- 10 customers and 40 active projects.
- Shared public regulatory evidence plus confidential customer documents.
- 500,000 existing pages.
- 2,000 new pages/day, with occasional 20,000-page uploads.
- 95% extractable text; 5% OCR.
- 15,000 questions/month with bursts up to 5 requests/second.
- Routine new evidence normally searchable within 30 minutes. For a 20,000-page burst, estimate backlog clearance explicitly; do not assume it shares the routine freshness target.
- 99.5% monthly availability target.
- No cross-customer or cross-project leakage in evidence, indexes, caches, traces, analytics, or answers.
- Recurring AWS and model spend below **$1,000/month**, plus at most **$500** for initial backfill.
- A path to roughly 10x growth without a ground-up replacement.

Submit a proposed architecture supported by a diagram (Mermaid source is sufficient), a rough capacity/cost estimate, and a compact data-contract table. Aim for a concise design of about 2–4 pages plus the table/calculation; this is design work, not an AWS implementation.

Treat the spend limits as design targets that require evidence. State the AWS region, dated public pricing links, pages-to-tokens assumptions, retention, extraction policy, query-model token use, and what is inside/outside your estimate. Include shared baseline services, OCR, model calls, storage/indexes, logs and the backfill separately; do not depend on temporary credits to claim sustainable cost. If the stated target cannot be met under defensible assumptions, quantify the shortfall and propose the smallest change in service level or design. A justified infeasibility conclusion is acceptable; an invented sub-$1,000 total is not. Candidate hosted-access budget and this hypothetical production budget are separate. You may consult service/pricing documentation for architecture; factual potency answers still use only the frozen corpus.

Cover both the operational path—ingestion through online answers—and how generated data supports research, offline/online evaluation, monitoring, tracing, feedback, and future agent improvement.

Be explicit about canonical storage formats for raw and normalized evidence, graph records, retrieval/generation traces, reviewer feedback, and evaluation results. Explain how analytical consumers use these without querying the production application database.

Address service selection, scaling behavior, cost controls, tenant isolation, security, failure/retry behavior, observability, versioning, deletion/correction, and migration. Tie choices to your prototype. State what you would retain, replace, defer, or validate. No AWS deployment or infrastructure-as-code is required.

### 6. Handoff

Your README must include setup and run commands, implementation overview, tradeoffs, actual system versions, elapsed time and usage/cost, completion status, next three steps, and how AI-generated work was verified. Include a short time log by work area.

Submit everything an evaluator needs to run the solution you provide from a clean checkout: source code, dependency lock files, configuration templates, schema/migrations, prompts, test fixtures, evaluation code, and generated results. Do not include credentials. Large regenerable model caches or indexes may be omitted only when your documented commands rebuild them.

## Expected output

Equivalent structures are acceptable when documented.

```text
results/
  import-report.json
  graph/
    nodes.jsonl
    edges.jsonl
  answers.jsonl
  traces.jsonl
  evaluation/
    cases.jsonl
    results.jsonl
    report.json
  architecture/
    architecture.md
    architecture-diagram.<mmd|png|svg|pdf>
    data-contracts.md
  fixtures/
    answers.jsonl
    traces.jsonl
    report.json
    history/                    # preserved before/after sources, graphs and run metadata
  time-log.md
```

Include automated checks for unchanged re-import after restart, exact citation text and source version, isolation-fixture containment, valid graph endpoints/traversals, the source-correction and conflict scenarios, and repeatable evaluation scoring. Model calls may use test doubles in offline tests; distinguish those from real model runs.

## Scope boundaries

Not required: a UI, AWS deployment, infrastructure-as-code, PDF/OCR implementation, a graph database, a graph visualization, a complete regulatory ontology, exhaustive extraction, a generic client roadmap engine, or fine-tuning. A simple architecture diagram remains a required design deliverable. No independent second-product run, incremental-update algorithm, historical re-query feature, working human-review workflow, or source-deletion demonstration is required. Production design still addresses correction, review and deletion.

Do not ingest assignment text, questions, schemas, generated answers, or evaluation labels as evidence.

## Suggested time allocation

| Time | Focus |
|---|---|
| 0:00–0:30 | Inspect materials, record assumptions and priorities. |
| 0:30–1:30 | Persistent import and evidence identity. |
| 1:30–3:15 | Focused graph extraction. |
| 3:15–4:15 | Retrieval, answers, and P3 comparison. |
| 4:15–5:30 | Evaluation cases, scoring, and failure analysis. |
| 5:30–7:30 | Production architecture, cost model, and data contracts. |
| 7:30–8:00 | Verification and handoff. |

## What we evaluate

- Evidence grounding and uncertainty handling: **20%**.
- Knowledge-graph usefulness and graph-assisted retrieval: **20%**.
- Evaluation design and interpretation: **20%**.
- Production architecture, data formats, scale, security, and cost awareness: **25%**.
- Engineering quality, prioritization, and communication: **15%**.

P3 is scored once within question correctness; its two-mode comparison is assessed separately as retrieval/evaluation evidence. Supported partial work earns partial credit. Do not work beyond eight hours to hide missing requirements. A transparent incomplete submission may still reveal useful judgment, but missing core behavior remains incomplete.

We intentionally do not publish the complete evaluator rubric. Fabricated evidence, concealed failures, hardcoded answers, or cross-scope data leakage are critical failures. Framework choice and visual polish carry no independent credit.
