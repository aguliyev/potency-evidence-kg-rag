# Retrieval, Answering and Fixture Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use `- [ ]` checkboxes. Status: draft v10. Only `lib/potency_kg/retrieval_config.py` and its unit test exist so far (written at the user's request); everything else is not yet implemented.

**Goal:** Produce the required answer/trace artifacts (assignment items 3 and 3a): `vector` and `graph` retrieval over the existing store, bounded context building, citation-validated generation, a driver for the 6 questions, and real-model runs of the supplied fixtures (correction, conflict, isolation).

**Architecture:** One scope-driven retrieval pipeline uses a single frozen `HybridRetrievalConfig` for both modes. Vector mode selects passages by pgvector; graph mode uses vector hits as seeds, traverses the current evidence graph, and selects original source passages. A shared evidence-block renderer owns all character-budget accounting, and every attempted run writes a trace even when no valid answer is produced.

**Tech Stack:** Python, SQLAlchemy, PostgreSQL/pgvector, Apache AGE for the legacy smoke path, Pydantic structured generation, JSONL/JSON artifacts, and the existing `bin/runner` test workflow.

**Spec:** `ASSIGNMENT.md`, `README.md`, `schema/output-contract.json`, and `fixtures/behavior-cases.json`.

## Global Constraints

- Use only the supplied corpus for factual answers; never ingest assignment text, questions, schemas, fixture metadata, generated answers, or evaluation labels as evidence.
- Graph retrieval must select original source passages and preserve tenant/project isolation, source-version identity, and evidence provenance.
- P3 uses vector and graph modes under the same corpus snapshot, generation model, policy, and 6,000 Unicode-code-point source-evidence cap.
- Save seven main answer/trace records and three fixture answer/generation-trace records; preserve correction history and isolation retrieval artifacts separately.
- Failed attempts always produce a trace; only validated successful answers are written as answer records.
- Freeze retrieval configuration and prompt version/hash before the final real-model run; disclose any later change and rerun both P3 modes.

---

**Not in scope (later plans):** evaluation set/scoring, AWS architecture docs, README handoff, time log.

## Decisions (confirmed with the user)

- Graph mode: vector hits seed a similarity-pruned beam traversal (variant D below); entities contribute their best-matching cited passage. Graph candidates are scored `similarity_weight*cosine + hop_bonus[hop] + edge_type_bonus[edge]`; graph-only passages get a reserved share of the character budget (`graph_share=0.4`). Vector mode uses the whole budget for vector hits. EVERY production tunable value lives in one documented frozen class, `HybridRetrievalConfig` (`retrieval_config.py`, done), whose hash goes into every trace; values are frozen before the first real run. No separate `VectorOnlyRetrievalConfig` is needed: vector mode ignores graph fields but records the complete shared config.
- Every question runs in BOTH modes (12 runs). The required seven records are P3 (both modes) plus P1/P2/P4/P5/P6 in `graph` mode. The five extra `vector` runs go to a separate directory.
- The LLM returns JSON; quotes are validated locally as exact substrings of the evidence actually sent.
- The context truncates the last passage to fit. The budget comes from the question when it carries a cap (`max_evidence_chars_per_mode` in `questions.json`, i.e. 6000 for P3, applied identically in both modes, no question ID in code) and otherwise from `HybridRetrievalConfig.default_budget_chars` (16000). Both the budget and the config are recorded in every trace. `max_chars` remains an explicit argument to retrieval/context functions so tests and callers can exercise different budgets.
- Graph traversal: measure first on real vector hits for P1/P2/P4/P5/P6 only, then freeze the traversal choice before running P3 generation. Compare only two variants: A (existing AGE traversal plus hub skipping) and D (similarity-pruned beam over the current projection). D is preferred; A is the fallback only if A is materially better on the predefined measurement rule. Hubs are detected generically by degree (above 20 edges), never by product name or question ID.
- Execution is staged. Stage 1 = Tasks 0-4 and 6 (main answers). Stage 2 = Task 5 (fixtures), only after Stage 1 is verified.
- Fixtures use `graph` mode, with the same mode before and after the correction. They are graded deliverables (real-model runs), not only tests.
- No reset of the DB. The main scope `public-fda/casgevy-potency` is read-only for this plan. Verified: it is the only scope in the DB.
- Every attempted run is persisted. A valid answer gets an answer record and a successful generation trace; provider, schema, or citation failures get a failed trace and no successful answer record. Empty evidence is a valid answer that says what was not found in the inspected scope.
- Only compact source metadata necessary to preserve source-role distinctions is sent in evidence headers: `source_id`, `source_version`, `source_type`, `authority`, page/section when available, and `passage_id`. Titles and graph descriptions are not sent unless a test demonstrates they are necessary.
- Evidence-bearing semantic nodes and edges must have valid current passage references. Structural/system-generated nodes or edges may have empty references, but are excluded from graph evidence selection. Hop-0 candidates have `via_edge_id=null`; no unrelated edge is invented.

## Reuse (already in `lib/potency_kg/`)

| Need | Existing piece |
|---|---|
| Query embedding and vector search | `embeddings.embed_query`, `vector_search` |
| Active graph, index binding, bounded AGE traversal | `retrieval._active_graph_binding`, `age_projection.bounded_graph_traversal` |
| Node/edge to passage links | tables `kg_node_source_ref`, `kg_edge_source_ref`; evidence candidates require current bound-index references |
| Structured LLM call plus usage | `llm_client.InstructorExtractionClient.extract(model_text, response_model, prompt)`, `usage_for_current_thread()` |
| Config/scope, protected-input guard, path guard | `cli._scope_from_descriptor`, `output_safety.guard_output_path` |
| Fixture import/embed/extract/graph/export, including declared replacements | `fixture_runs.run_kg_fixtures` (test-double only today), `ingest` replacement declarations |
| Real clients | `cli._embedding_client`, `_jev_client`, `_generation_client` |

No new LLM client is needed: pass a Pydantic `AnswerModel` to `extract`.

## Design facts that shape the plan

- **Budget:** the median passage is about 1,800 chars (90th percentile about 3,300). A 6,000-char cap fits only about 3 passages, so ordering decides the vector vs graph comparison.
- **Graph does not select passages today:** `retrieve_graph_smoke` returns `source_refs` only for vector hits. This is the main code change.
- **Traversal is unranked and structural:** `bounded_graph_traversal` returns node and edge IDs without hop info. Its adjacent-edge query is `ORDER BY edge.logical_id LIMIT n` with no relevance order, and it also follows structural edges (`contains_passage`) and `SourceDocument` nodes. Collecting passages from those would pull in arbitrary passages of a document. Graph passages must therefore come only from evidence edges and semantic nodes, and be ranked by query similarity.
- **The graph is a star around one hub (measured on `results/kg/current/graph/`):** one canonical `Product` node has 456 edges; no other non-document node has more than 4. About 880 of the 1,979 `Assertion` nodes are leaves with an `applies_to` edge to `Product`/`Application`/`Indication`. `_semantic_seed_candidates` takes any non-document node referencing a hit passage (ordered by node ID), so the hub can become a seed, and one hop from it reaches 456 assertions, of which an arbitrary 50 survive the edge limit.
- **`kg_node` holds many entities that are not in the graph (verified in the DB; corrected from an earlier wrong diagnosis):** the graph is the extraction output, projected by `graph_store.load_current_projection(scope, "semantic")`. There is one extraction run per passage (746 runs, 746 passages) and all 1,979 assertions come from active runs, so nothing is stale in the main scope. But `kg_node` has about 2,700 entity nodes with source refs and NO relations (541 `MeasuredAttribute` vs 62 in the graph, 222 `Assay` vs 69, 341 `ReviewConclusion` vs 87, ...). The projection keeps only documents, passages, current assertions and nodes with a current relation, so isolated entities are outside the graph, the AGE graph and the export. Consequences: seeds taken from `kg_node_source_ref` can be isolated entities (dead ends), and the extraction found many entities but linked few of them (an extraction-coverage finding for the README). Variant D reuses the projection so it sees exactly the graph a reviewer inspects, and so it stays correct in the source-correction fixture, where superseded v1 rows do exist.
- **Potency edges are sparse:** `proposed_in` 1, `requested_in` 1, `revised_by` 0, `used_as` 6, `measures` 68, `concluded_in` 87. `Assay` and `PotencyStrategy` nodes are all labelled just with their type. P3 graph mode may add little over vector mode; that is an acceptable, reportable result, not something to hide.
- **Limits:** the existing AGE function keeps its smoke maxima (8 hits, 8 seeds, 2 hops, 30 nodes, 50 edges). Variant D has its own declared parameters (see 1b). All of them are recorded in every trace.
- **Model-reproduced IDs are fragile:** passage IDs are long (`casgevy-125785-analytical-method-review:page-0001-c01`). The model cites short tags `S1..Sn`, which code maps back to passage IDs.

## File map

- Modify `lib/potency_kg/age_projection.py` ONLY if variant A is chosen: return `node_hops`/`edge_hops` from `bounded_graph_traversal` (keep existing fields so current callers and tests still pass).
- Create `lib/potency_kg/graph_retrieval.py`: variant D. It loads the CURRENT projection once per (scope, `graph_version`) with `load_current_projection(connection, scope, "semantic", ...)` plus its node/edge source refs restricted to current source heads, builds in-memory adjacency (3,472 nodes and 1,979 edges, trivial), computes the cosine of the question to every index passage with ONE pgvector query (about 800 rows, joined to `index_member` for the bound `index_version`), and runs the beam in Python. No per-hop SQL, no AGE caps, and the current-graph rule is not duplicated. In production this becomes a graph store or per-hop queries (architecture plan).
- DONE: `lib/potency_kg/retrieval_config.py` defines `HybridRetrievalConfig` (frozen dataclass, validated, heavily commented). Fields: `vector_hits`, `vector_min_cosine`, `max_passages_per_document`, `max_seeds`, `seed_vector_hits`, `traversal` (beam/age), `neighbors_per_entity`, `frontier_cap_per_hop`, `direction`, `evidence_edges_only`, `excluded_node_types`, `expand_hubs`, `passages_per_entity`, `max_hops`, `hub_degree_cap`, `candidate_cap`, `similarity_weight`, `hop_bonus`, `edge_type_bonus`, `graph_min_cosine`, `graph_share`, `return_unused_graph_share`, `context_order`, `min_truncated_chars`, `truncate_last_passage`, `default_budget_chars`, `prompt_version`, `citation_repair_attempts`, `allow_normalized_quote_match`. Helpers: `graph_score`, `edge_bonus`, `hop_weight`, `as_dict`, `config_hash`. A unit test checks defaults, hash stability, scoring, validation, and that every field appears in the trace dict. All code below takes the config as an argument instead of constants. Deliberately not configurable (documented in the module): cosine metric, tenant/project and bound-index filtering, headers counting against the budget, and any source-type/authority pre-filter.
- Modify `lib/potency_kg/retrieval.py`: add `retrieve(...)` (vector and graph modes) using `HybridRetrievalConfig` as the production configuration. Keep `retrieve_graph_smoke` as a thin compatibility wrapper around the vector/legacy smoke behavior; do not introduce a second production limits object.
- Create `lib/potency_kg/answering.py`: a shared evidence-block renderer/budget allocator, `merge_candidates`, `build_context`, `AnswerModel`, `validate_citations`, `answer_question`, record builders, and `run_questions`.
- Modify `lib/potency_kg/cli.py`: add subcommands `answer` and `run-fixture-answers`.
- Modify `lib/potency_kg/fixture_runs.py`: allow real clients (gated), and add an `on_stage` callback.
- Create `lib/potency_kg/prompt_resources/answer_system.md` (`answer-prompt-v1`).
- Create tests: `tests/unit/test_answering.py`, `tests/integration/test_retrieve_modes.py`, `tests/integration/test_answer_cli.py`.
- Add or extend tests for source-reference completeness, failed-run persistence, prompt hashing, fixture history manifests, and reconnect call accounting.

## Task 0: preflight (read-only)

- [ ] DB container up; `bin/runner` works; env has `LLM_*`, `EMBEDDING_LLM_*`, `TYPESAFE_API_KEY` (report presence only, never values); `KG_RUN_REAL_MODELS=true`.
- [ ] One real query embedding and one tiny real `extract` call succeed (this is a smoke test, not the timed access check). Note the model IDs for the trace.
- [ ] Confirm `graph_scope.index_version` is bound to the index in `import-report.json`.

## Task 1: graph-selected passages in retrieval

- [ ] **1a. Measure first (read-only, 5 real query embeddings; throwaway script, output saved to `docs/graph-signal-measurement.md`).** Compare A and D on the same real vector hits for P1, P2, P4, P5, and P6. P3 is held out from traversal choice and is run only after the configuration is frozen. For each measured question, record whether each variant adds at least one directly relevant non-vector passage. Choose D if it adds relevant graph-only evidence for at least 3 of the 5 questions and is not clearly worse than A; choose A only if it meets that criterion while D does not; if neither does, ship D and report that graph retrieval added little.
  - **A**: existing AGE traversal with hub skipping (a node with degree above 20 is never a seed or an expansion point).
  - **D (preferred)**: similarity-pruned beam. Seeds: up to `max_seeds` entities (not in `excluded_node_types`, not hubs) that cite one of the top `seed_vector_hits` vector hits, ordered by that hit's vector score, then entity ID. For every frontier entity keep its top `neighbors_per_entity` neighbors (relations followed in `direction`, `evidence_edges_only`), ranked by `config.graph_score(cosine of the neighbor's best cited passage to the QUESTION, hop, edge_type)`: the hop bonus decreases with distance from the seed and the edge-type bonus favors potency relations. After each hop only the best `frontier_cap_per_hop` entities (over all seeds) continue; hubs (degree above `hub_degree_cap`) are not expanded unless `expand_hubs`. Repeat up to `max_hops`. The beam runs over the in-memory current projection (see file map); per-entity top-N is a sort of its neighbors by graph score. Hub degree is the number of evidence relations in the current graph. An entity contributes its `passages_per_entity` best CURRENT bound-index passages from its node source references; evidence-edge source references may also contribute when explicitly selected. Dedup across hops; candidates under `graph_min_cosine` (if set) are dropped; the list is sorted by graph score.
  - Report per question: seeds, hub involvement, entities visited per hop, how many NON-vector passages each variant adds, and how many candidates were skipped for missing/non-current provenance. If candidates mostly come through the hub, say the variant is effectively filtered vector search.
  - **Choosing the bonus values.** Print the distribution of cosine scores (top hits vs graph candidates) for the 5 measurement questions and set `hop_bonus` / `edge_type_bonus` so the bonus is a nudge, not an override (cosine spreads are narrow, so 0.05 is already strong). Set them from this distribution only, never from P3 answers; then commit the config.
- [ ] **Hub accounting.** Per hop, record in the trace: relations available for each expanded entity, relations kept (N), and whether the entity or edge went through a hub. The report states how much of the graph contribution is hub-mediated.
- [ ] **1b. Declare the traversal (assignment requirement).** Undirected by default (`direction="both"`); a hop is one relation from the current frontier; every parameter is a `HybridRetrievalConfig` field and the full `as_dict()` is written to the trace and described in `docs/`. Defaults: 8 vector hits, 8 seed hits, 8 seeds, `N=5`, frontier cap 50, 3 hops, hub cap 20, candidate cap 500, hop bonus 0.05/0.03/0.01, potency edge bonus 0.05, graph share 0.4.
- [ ] Write the failing integration test (pattern: `tests/integration/test_retrieval_smoke.py`). Scope with passage A as a vector hit linked to a semantic node N; an evidence edge from N whose source-ref passage is B, and B is NOT a vector hit. Assert:
  - `retrieve(mode="graph")` returns B with `selected_via="graph"`; `mode="vector"` does not.
  - A structural `contains_passage` edge or a `SourceDocument` node never yields passages (add a passage C that is only reachable that way and assert it is absent).
  - A second scope with identical IDs never appears.
  - A passage whose source version is not in the bound index (superseded or not embedded) is never returned, even if a stale node or edge refers to it.
  - A node or edge that exists in `kg_node`/`kg_edge` but is not in the current projection (an isolated entity, or in the correction fixture a superseded v1 row) is never a seed, a neighbor or a passage source.
  - A semantic node or evidence edge without a valid current source reference is skipped and counted in retrieval diagnostics.
  - A hop-0 seed candidate has `hop=0`, `via_node_id=<seed>`, and `via_edge_id=null`; a candidate reached through an edge records that edge and does not invent a hop-0 edge.
- [ ] Only if A is chosen: extend `bounded_graph_traversal` to return `node_hops` / `edge_hops`.
- [ ] Implement the chosen variant. For D it returns candidates (key, text, text_sha256, locator, source metadata, score, hop, nullable `via_node_id`, nullable `via_edge_id`, supporting path IDs, and hub flag). For A, a `graph_passages(connection, scope, node_ids, edge_ids, hops)` helper joins the node/edge source refs (non-structural nodes, evidence edges only) to `source_passage` and to `index_member` for the bound index.
- [ ] Implement `retrieve(engine, scope, question, mode, embedding_client, index_version, config, query_embedding=None)`:
  - Optional precomputed `query_embedding`, so one question embedded once serves both modes. This also guarantees identical vector seeds for the comparison.
  - `vector`: vector search, load passages. No graph work.
  - `graph`: the variant chosen in 1a (default D). Every graph candidate carries a query-similarity score from pgvector, so all candidates are comparable with vector hits.
  - Return a `RetrievalResult` with the original trace fields plus `candidates[]`: key, `document_id` (source_id), `selected_via` in {vector, graph, both}, score, hop, via IDs, supporting paths, and source metadata. `via_edge_id` is nullable for hop-0 candidates.
- [ ] Run the new test and `tests/integration/test_retrieval_smoke.py`; commit.

## Task 2: merging and context building (pure functions)

- [ ] Unit tests first:
  - `merge_candidates(mode, max_chars, config)`: vector mode returns hits (at or above `vector_min_cosine` if set) by score. Graph mode adds vector hits in rank order until they use `(1-graph_share)*max_chars`, then graph-only passages in graph-score order until the budget is full; if graph-only passages run out and `return_unused_graph_share`, the remaining budget goes to the unused vector hits. The shared evidence-block renderer, including headers and separators, supplies the exact character cost used for both share allocation and final context construction. `max_passages_per_document` is enforced while filling. Dedup by passage key; a passage found by both is `both` and consumes the vector allocation unless it is explicitly selected as a graph-only candidate. Deterministic. Both modes get the same total budget. The reserved share is forced: no floor by default (`graph_min_cosine=None`, decided; see Risks).
  - `build_context(candidates, max_chars, config)`:
    - `context_order="document"` groups blocks by document in page order; `"score"` keeps best first. Selection is identical either way; tags `S1..Sn` follow the final order. `truncate_last_passage=False` drops instead of cutting.
    - Block format: `[S1] source_id | source_version | source_type | authority | page/section | passage_id` plus the exact text; omit unavailable metadata fields rather than emitting placeholders. Do not include graph descriptions or titles unless a test demonstrates they are necessary.
    - `len(evidence_text)` in Unicode code points is `<= max_chars`, including headers and separators.
    - Whole passages are added in order. The last passage is truncated to a verbatim prefix when at least 300 chars remain; `truncated=true` and the number of chars kept are recorded. Later candidates go to `dropped`.
    - Returns `evidence_text`, the `S#` to key map, `sent_text` per passage (the exact text sent, possibly truncated), `chars_used`, `dropped`.
- [ ] Implement the shared `EvidenceBlockRenderer`/budget allocator first, then make both `merge_candidates` and `build_context` call it. Only passage text and block headers are source-derived content; no graph descriptions are sent, so the P3 cap is auditable from `evidence_text` alone.
- [ ] `max_chars` is an argument everywhere (`merge_candidates`, `build_context`, `answer_question`). The driver resolves it per question: the question's own cap if present, else `config.default_budget_chars`; the CLI has `--default-budget-chars` to override the config default. The value used is stored in the trace (`max_evidence_chars`).
- [ ] Commit.

## Task 3: generation with local citation validation

- [ ] Unit tests with a fake `extract` client:
  - Valid answer passes.
  - A quote that is not a substring of the SENT text of the cited tag fails. Matching is exact first; if that fails, whitespace-collapsed matching on both sides is accepted and flagged `match="normalized"` in the citation record (passages come from PDFs and contain line breaks and form feeds). A quote that exists only in the truncated-away part of the passage also fails.
  - A tag not in the context fails.
  - After one repair retry that also fails, the generation trace is saved with `status="failed"` and `citation_check.status="failed"`; no invalid answer is written to the successful answers JSONL. The failed trace may retain the model attempts for audit, but must identify them as unvalidated.
  - Empty evidence yields an answer saying nothing was found in the inspected scope.
  - A provider or schema error on one question is recorded as a failed run (error type only for provider details; no credentials or raw provider error text) and does not stop the other questions.
- [ ] Write `answer-prompt-v1`:
  - Answer only from the evidence.
  - Cite with tags `S#` plus a verbatim quote.
  - Distinguish regulation, guidance, applicant proposal, reviewer conclusion and final agency action.
  - Say what is not found or is redacted; never infer `(b)(4)` text.
  - Report an unresolved conflict when equal-authority sources disagree and no replacement is declared.
- [ ] Message layout: system = `answer-prompt-v1`; user = question plus `evidence_text`. The question is not source-derived. The repair retry sends the same evidence plus feedback that names failed tags/quotes (no new source text); the trace stores the feedback text and both attempts.
- [ ] `AnswerModel = {answer, citations[{tag, quote}], evidence_gaps[]}`. `answer_question(client, question, context) -> (answer, usage, elapsed_ms, attempts)`; elapsed time measured around `extract`. Capture usage after each attempt and aggregate it explicitly because the concrete client replaces its thread-local usage state on each call.
- [ ] Expand each cited tag into the contract citation: scope, source_id, source_version, passage_id, page, section (from the locator; may be null), quote.
- [ ] Record `prompt_version` and `prompt_sha256` in every trace. The committed prompt file is the canonical system prompt; the hash prevents a prompt edit from silently sharing an old version label.
- [ ] Commit.

## Task 4: record writers and the `answer` CLI

- [ ] Test with fake embedding and generation clients: every field in `schema/output-contract.json` (`answer_required_fields`, `trace_required_fields`) is present; `trace_id` links answer and trace; IDs are unique; `evidence_text` equals the text sent; P3 records satisfy `len(evidence_text) <= 6000` and the others `<= 16000`; a failed run produces a trace with `status="failed"` and no answer record.
- [ ] Field semantics:
  - `selected_passage_ids` = passages actually sent to the model. `candidate_passage_ids` = everything retrieved before the budget (kept separately).
  - Also: `document_ids`, `selected_via` per passage, `seed_node_ids`, `node_ids`, `traversed_edge_ids`, `hops`, `traversal_limits`, `max_evidence_chars`, `truncated`, `dropped`, `prompt_version`, `retriever_version`, `model`, `usage`, `attempts`, `elapsed_ms`, `citation_check`, `index_version`, `graph_version`, `corpus_snapshot` (from the config).
- [ ] Failed-run semantics are explicit: every attempted question/stage writes one trace; `status="failed"` includes `failure_kind` (`provider`, `schema`, or `citation_validation`), safe error metadata, attempts, usage, and exact evidence sent. Only `status="succeeded"` traces have corresponding answer records. Empty-evidence answers are successful records with an explicit evidence gap.
- [ ] **Freeze before the first real run.** Commit the final `HybridRetrievalConfig` defaults (or a named config instance) and the prompt, then run. The trace carries `config.as_dict()`, `config.config_hash()` and the git commit. Any change after seeing P3 answers is disclosed in the comparison note and the run is repeated for both modes.
- [ ] CLI: `python -m potency_kg.cli answer --config configs/casgevy-potency.json --output results [--default-budget-chars 16000]`.
  - Reads `questions.json`; embeds each question once; runs vector and graph with the same budget, config and prompt (6000 for P3 from its cap, the config default for the rest). Gates on `KG_RUN_REAL_MODELS`.
  - `results/answers.jsonl`, `results/traces.jsonl`: required seven (P3 both modes; the others `graph`).
  - `results/extra-mode/answers.jsonl`, `results/extra-mode/traces.jsonl`: the five other `vector` runs.
  - Output through `guard_output_path`; protected input hashes must be unchanged afterwards.
- [ ] Main answer runs persist the exact selected evidence in traces and append no unvalidated answer rows. Prompt text remains in the committed prompt resource; traces carry its version and SHA-256 rather than duplicating the system prompt in every row.
- [ ] Commit.

## Task 5 (Stage 2, after Stage 1 is verified): real-model fixture runs (item 3a plus isolation)

- [ ] Before writing code: keep `_verify_reconnect` in real mode. It performs one additional small query-embedding call per fixture scope (five calls with the current catalog), does not generate answers, and is recorded in the run usage/cost note. Do not add complexity solely to avoid these calls; skip only if the supplied access budget or rate limits make them material.
- [ ] Precondition (verified): the DB holds only `public-fda/casgevy-potency`. Fixture scopes (correction, conflict, isolation project one/two) are built for the first time with real models. This is the only DB write in the plan, confined to those scopes. Earlier test-double exports under `results/kg/fixtures/` stay on disk as offline checks.
- [ ] Add `allow_real` to `run_kg_fixtures` (default stays double-only so existing offline tests pass). Real mode requires `KG_RUN_REAL_MODELS` and uses `cli._embedding_client/_jev_client/_generation_client`.
- [ ] Add `on_stage(configuration, report)` called right after the stage's graph refresh and before the next stage runs, so the answer is generated while that stage's evidence is current. Reuse the stage's already-computed retrieval result in `report` rather than embedding the same fixture question a second time. For the correction fixture the v1 answer happens before v2 replaces it.
- [ ] `run-fixture-answers` CLI:
  - Runs F-update before, F-update after (declared replacement) and F-conflict combined in `graph` mode, using the question from `fixtures/behavior-cases.json`. The same mode before and after.
  - The isolation scopes are built in the same run and in the same store (assignment item 1) but get no answers.
  - Writes `results/fixtures/answers.jsonl` and `results/fixtures/traces.jsonl` (3 generation records, with `scenario_id`, `stage`, `run_id`), `results/fixtures/history/<scenario>/<stage>/`, and `results/fixtures/report.json`.
  - Each history stage uses stable names: `manifest.json`, `source-records.jsonl`, `import-report.json`, `graph/nodes.jsonl`, `graph/edges.jsonl`, `retrieval-traces.jsonl`, `answers.jsonl`, and `traces.jsonl`. Correction and conflict stages contain one answer/generation trace; isolation stages contain an empty `answers.jsonl`/`traces.jsonl` plus a retrieval-only trace.
  - The history manifest explicitly records `generation_performed`, `answer_count`, `retrieval_trace_path`, `answer_path`, and `generation_trace_path`, so retrieval-only isolation stages cannot be confused with failed generation.
  - `report.json` has: commands, checks (passed/failed/not_run), observed results, failures, and a `history_artifacts` map linking snapshot, index, source versions, answers and traces.
- [ ] Checks computed from saved generation traces, saved retrieval-only traces, and the DB:
  - Before answer cites v1, after answer cites v2 (citation `source_version` equals the stage's current version); the v1 passage ID is absent from the after-stage evidence; versioned citations resolve to exact text.
  - Conflict: both claim passages cited. The wording that the conflict is unresolved is a keyword check, marked heuristic, not a human judgement.
  - Isolation: for every retrieval trace in the isolation scopes, all passage, node and edge IDs belong to that scope's own graph. Also run one query per scope and assert zero evidence from the other.
- [ ] Commit.

## Task 6: run, verify, document

- [ ] `bin/runner pytest tests/unit tests/integration` is green (including re-import and isolation tests).
- [ ] Real run: `answer`, then `run-fixture-answers`. Spot-check 3 answers by hand against the cited passages. Record token usage and cost.
- [ ] Verify:
  - Record counts are 7 main, 5 extra-mode, 3 fixture.
  - P3 `evidence_text` lengths per mode.
  - Query-embedding calls are approximately: 1 preflight, 5 traversal-measurement questions, 6 main questions, 5 fixture-stage retrievals, and 5 reconnect checks. Fixture answer generation reuses the stage retrieval result and adds no duplicate query embedding. Record actual counts and provider usage.
  - Every `passage_id`, node ID and edge ID in a trace resolves in the DB and in `results/kg/current/graph/*.jsonl`.
  - No secrets in outputs (`rg` for key names and values from the secrets `.env`).
- [ ] Map the assignment's automated-check list to tests: unchanged re-import after restart (`test_import.py`, `test_embeddings.py`), exact citation text and version (new, Task 3/4), isolation containment (`test_behavior_graphs.py` plus Task 5), valid graph endpoints/traversals (`test_age_refresh.py`, `test_exports.py`, new traversal test), correction and conflict (Task 5), repeatable evaluation (later plan). Note which are offline test-double checks and which real runs.
- [ ] Write the P3 vector vs graph comparison note (observed contribution or failure) in `docs/`.
- [ ] `graphify update .`.

## Risks and open items

- **Budget vs recall:** with about 3 passages per P3 answer, P3 graph and vector may end up with the same evidence. Report that honestly; do not tune the merge after seeing P3 answers, beyond what is written here (any change must be disclosed).
- **Hub and sparse potency edges:** see Design facts. Mitigations: degree-based hub skipping, similarity-pruned beam, configurable hop/edge bonuses chosen from the score distribution, the measurement gate in 1a. Bonuses only reorder candidates inside the graph share, and the potency edges are rare, so expect a modest effect. If graph passages are still mostly irrelevant, say so in the comparison note.
- **Large configuration surface:** 29 fields. The real runs use the frozen defaults; the other fields exist so later experiments (hubs on, document cap, other direction) are one change plus a new `config_hash`, not new code. Every field must appear in the trace (unit test) and every non-default value used in a reported run must be named in the README.
- **Forced graph share without a floor (decided: no floor):** graph-only passages are by definition below the vector top-8, so they may be weakly relevant. A forced 40% share can lower the graph-mode answer quality. That is a legitimate finding, but it is a design choice the comparison note must state. A declared floor was considered and rejected because it can make graph mode identical to vector mode.
- **Hours:** the assignment budgets about 1 hour for retrieval/answers/P3. Stage 1 fits that only if the traversal decision is quick; if the gate in 1a forces an own traversal, say so in the time log.
- **Quote validation:** the model may paraphrase. One repair retry is recorded. If many still fail, add a declared whitespace/hyphenation normalization rather than fuzzy matching.
- **Determinism:** `extract` exposes no temperature or seed, so reruns may differ. Saved traces, not reruns, are the record.
- **Fixture build** is the only DB write. The correction fixture must run in order. A re-run needs fresh fixture scopes or dropping those scopes only; out of scope here.
- **Limits:** if the caps prove too small, raise them in one place and re-run both modes; never change them for one mode only.
- **Configuration boundary:** `HybridRetrievalConfig` is the only production retrieval policy. The old `RetrievalLimits` remains only as a compatibility type for legacy smoke tests, if required.
- **Evidence budget:** candidate merging and final context rendering use the same block renderer, so graph/vector share allocation and the final Unicode character cap cannot disagree.
- **Prompt context:** source headers contain only the compact metadata needed to distinguish source role and authority; all source-derived header characters count toward the evidence budget.
- **Provenance gaps:** not every structural node needs a source reference, but every semantic node/evidence edge used for graph evidence must have a valid current passage reference. Missing or non-current references are skipped and counted.
- **Reconnect cost:** real-mode reconnect verification intentionally adds five small query-embedding calls with the current five fixture scopes. This is accepted for simplicity and must be included in usage/cost reporting.

## Changes from draft v1

- Removed the fixture "reset"; the DB has no fixture scopes (verified).
- Added Task 0 (preflight), per-hop traversal info, exclusion of structural edges and document nodes from graph passage collection, and similarity ranking of graph passages instead of hop-first.
- The model now cites `S#` tags, validated against the text actually sent (truncation-aware).
- Split `candidate_passage_ids` from `selected_passage_ids`; one embedding per question shared by both modes.
- Failed runs are recorded without stopping the batch.
- Added the isolation fixture to the real fixture run, with a containment check.
- v3: hub/sparse-edge findings, measurement gate and declared traversal (1a/1b), split budgets (6000 for P3, 16000 for the rest), staged execution, retry-feedback accounting, reconnect check.
- v4: beam traversal (variant D) with question-similarity scoring plus edge-type bonus, best-passage entity score, hub accounting in the trace, and a reserved graph share (0.4) instead of round-robin.
- v5: removed contradictions (decisions vs variants, duplicated gate), cut variants B and C (subsumed by D), fixed bonus `+0.05`, hub defined by degree, seeds ordered by hit score, graph candidates restricted to the bound index, whitespace-normalized quote matching, frozen params with hash in the trace, automated-check mapping, AGE changes only if A is chosen.
- v6: added `HybridRetrievalConfig` (all tunables in one commented, validated, hashable class; implemented), hop-weighted graph scoring (`similarity_weight*cosine + hop_bonus + edge_type_bonus`), 40% graph share kept, budgets resolved per question from `questions.json` caps (no question IDs in code) with `default_budget_chars` for the rest.
- v7: more fields in `HybridRetrievalConfig` (document diversity cap, vector/graph cosine floors defaulting to none, seed hits, per-hop frontier cap, direction, evidence-only edges, excluded node types, hub expansion switch, passages per entity, unused-share return, context order, truncation switch); fixed the visit-count claim (N^hops is per seed; a global per-hop cap is needed).
- v8: review found that `kg_node`/`kg_edge` contain stale rows from superseded extraction runs (verified), so variant D now runs in memory over `load_current_projection` plus one pgvector query for passage similarities instead of SQL per hop over raw tables; added a stale-row test.
- v9: corrected the v8 diagnosis (no stale extraction runs in the main scope; the extra `kg_node` rows are isolated entities left out of the projection), fixed graph size, kept projection reuse for consistency and for the correction fixture.
- v10: made `HybridRetrievalConfig` the sole production policy, centralized exact character accounting, held P3 out of traversal tuning, defined saved retrieval-only fixture traces and history filenames, persisted failed generation traces without invalid answer rows, added prompt hashing and compact source-role metadata, specified provenance handling including hop-0 candidates, and accepted five extra real reconnect query embeddings for simpler verification.
