# Retrieval, Answering and Fixture Runs Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Status: draft v2, not yet executed.

**Goal:** Produce the required answer/trace artifacts (assignment items 3 and 3a): `vector` and `graph` retrieval over the existing store, bounded context building, citation-validated generation, a driver for the 6 questions, and real-model runs of the supplied fixtures (correction, conflict, isolation).

**Not in scope (later plans):** evaluation set/scoring, AWS architecture docs, README handoff, time log.

## Decisions (confirmed with the user)

- Graph mode is merged: vector hits seed the traversal; traversed nodes and edges contribute their source-ref passages; both lists are merged and filled into the character budget.
- Every question runs in BOTH modes (12 runs). The required seven records are P3 (both modes) plus P1/P2/P4/P5/P6 in `graph` mode. The five extra `vector` runs go to a separate directory.
- The LLM returns JSON; quotes are validated locally as exact substrings of the evidence actually sent.
- The context truncates the last passage to fit. The character budget is an argument (`max_chars`, default 6000).
- Fixtures use `graph` mode, with the same mode before and after the correction. They are graded deliverables (real-model runs), not only tests.
- No reset of the DB. The main scope `public-fda/casgevy-potency` is read-only for this plan. Verified: it is the only scope in the DB.

## Reuse (already in `lib/potency_kg/`)

| Need | Existing piece |
|---|---|
| Query embedding and vector search | `embeddings.embed_query`, `vector_search` |
| Active graph, index binding, bounded AGE traversal | `retrieval._active_graph_binding`, `age_projection.bounded_graph_traversal` |
| Node/edge to passage links | tables `kg_node_source_ref`, `kg_edge_source_ref` |
| Structured LLM call plus usage | `llm_client.InstructorExtractionClient.extract(model_text, response_model, prompt)`, `usage_for_current_thread()` |
| Config/scope, protected-input guard, path guard | `cli._scope_from_descriptor`, `output_safety.guard_output_path` |
| Fixture import/embed/extract/graph/export, including declared replacements | `fixture_runs.run_kg_fixtures` (test-double only today), `ingest` replacement declarations |
| Real clients | `cli._embedding_client`, `_jev_client`, `_generation_client` |

No new LLM client is needed: pass a Pydantic `AnswerModel` to `extract`.

## Design facts that shape the plan

- **Budget:** the median passage is about 1,800 chars (90th percentile about 3,300). A 6,000-char cap fits only about 3 passages, so ordering decides the vector vs graph comparison.
- **Graph does not select passages today:** `retrieve_graph_smoke` returns `source_refs` only for vector hits. This is the main code change.
- **Traversal is unranked and structural:** `bounded_graph_traversal` returns node and edge IDs without hop info. Its adjacent-edge query is `ORDER BY edge.logical_id LIMIT n` with no relevance order, and it also follows structural edges (`contains_passage`) and `SourceDocument` nodes. Collecting passages from those would pull in arbitrary passages of a document. Graph passages must therefore come only from evidence edges and semantic nodes, and be ranked by query similarity.
- **Limits:** the smoke maxima (8 hits, 8 seeds, 2 hops, 30 nodes, 50 edges) are kept as defaults and recorded in every trace.
- **Model-reproduced IDs are fragile:** passage IDs are long (`casgevy-125785-analytical-method-review:page-0001-c01`). The model cites short tags `S1..Sn`, which code maps back to passage IDs.

## File map

- Modify `lib/potency_kg/age_projection.py`: have `bounded_graph_traversal` also return `node_hops` and `edge_hops` (the hop at which each was reached; 0 for roots). Keep existing fields so current callers and tests still pass.
- Modify `lib/potency_kg/retrieval.py`: add `graph_passages(...)` and `retrieve(...)`. Keep `retrieve_graph_smoke` as a thin wrapper.
- Create `lib/potency_kg/answering.py`: `merge_candidates`, `build_context`, `AnswerModel`, `validate_citations`, `answer_question`, record builders, `run_questions`.
- Modify `lib/potency_kg/cli.py`: add subcommands `answer` and `run-fixture-answers`.
- Modify `lib/potency_kg/fixture_runs.py`: allow real clients (gated), and add an `on_stage` callback.
- Create `lib/potency_kg/prompt_resources/answer_system.md` (`answer-prompt-v1`).
- Create tests: `tests/unit/test_answering.py`, `tests/integration/test_retrieve_modes.py`, `tests/integration/test_answer_cli.py`.

## Task 0: preflight (read-only)

- [ ] DB container up; `bin/runner` works; env has `LLM_*`, `EMBEDDING_LLM_*`, `TYPESAFE_API_KEY` (report presence only, never values); `KG_RUN_REAL_MODELS=true`.
- [ ] One real query embedding and one tiny real `extract` call succeed (this is a smoke test, not the timed access check). Note the model IDs for the trace.
- [ ] Confirm `graph_scope.index_version` is bound to the index in `import-report.json`.

## Task 1: graph-selected passages in retrieval

- [ ] Write the failing integration test (pattern: `tests/integration/test_retrieval_smoke.py`). Scope with passage A as a vector hit linked to a semantic node N; an evidence edge from N whose source-ref passage is B, and B is NOT a vector hit. Assert:
  - `retrieve(mode="graph")` returns B with `selected_via="graph"`; `mode="vector"` does not.
  - A structural `contains_passage` edge or a `SourceDocument` node never yields passages (add a passage C that is only reachable that way and assert it is absent).
  - A second scope with identical IDs never appears.
- [ ] Extend `bounded_graph_traversal` to return `node_hops` / `edge_hops`.
- [ ] Implement `graph_passages(connection, scope, node_ids, edge_ids, hops)`: join `kg_node_source_ref` (excluding `SourceDocument`/`SourcePassage` nodes) and `kg_edge_source_ref` (evidence edges only, `is_evidence`) to `source_passage`; return key, text, text_sha256, locator, min hop, and the via node/edge ID.
- [ ] Implement `retrieve(engine, scope, question, mode, embedding_client, index_version, limits, query_embedding=None)`:
  - Optional precomputed `query_embedding`, so one question embedded once serves both modes. This also guarantees identical vector seeds for the comparison.
  - `vector`: vector search, load passages. No graph work.
  - `graph`: existing flow, plus `graph_passages`. Query similarity for graph-only passages is computed in SQL against `embedding_run` (same cosine as `vector_search`), so every candidate has a score.
  - Return a `RetrievalResult` with the original trace fields plus `candidates[]`: key, `document_id` (source_id), `selected_via` in {vector, graph, both}, score, hop, via IDs.
- [ ] Run the new test and `tests/integration/test_retrieval_smoke.py`; commit.

## Task 2: merging and context building (pure functions)

- [ ] Unit tests first:
  - `merge_candidates(mode)`: vector mode returns hits by score. Graph mode interleaves round-robin the vector list (by score) and the graph list (by score, then hop), dedups by passage key, and marks a passage found by both as `both`. Deterministic.
  - `build_context(candidates, max_chars=6000)`:
    - Block format: `[S1] source_id | source_version | page/section | passage_id` plus the exact text.
    - `len(evidence_text)` in Unicode code points is `<= max_chars`, including headers and separators.
    - Whole passages are added in order. The last passage is truncated to a verbatim prefix when at least 300 chars remain; `truncated=true` and the number of chars kept are recorded. Later candidates go to `dropped`.
    - Returns `evidence_text`, the `S#` to key map, `sent_text` per passage (the exact text sent, possibly truncated), `chars_used`, `dropped`.
- [ ] Implement. Only passage text and block headers are source-derived content; no graph descriptions are sent, so the P3 cap is trivially auditable.
- [ ] `max_chars` is threaded through `answer_question`, the CLI (`--max-chars`) and the trace (`max_evidence_chars`).
- [ ] Commit.

## Task 3: generation with local citation validation

- [ ] Unit tests with a fake `extract` client:
  - Valid answer passes.
  - A quote that is not a substring of the SENT text of the cited tag fails. A quote that exists only in the truncated-away part of the passage also fails.
  - A tag not in the context fails.
  - After one repair retry that also fails, the record is saved with `citation_check.status="failed"`, never silently fixed.
  - Empty evidence yields an answer saying nothing was found in the inspected scope.
  - A provider error on one question is recorded as a failed run (error type only, no raw provider text) and does not stop the other questions.
- [ ] Write `answer-prompt-v1`:
  - Answer only from the evidence.
  - Cite with tags `S#` plus a verbatim quote.
  - Distinguish regulation, guidance, applicant proposal, reviewer conclusion and final agency action.
  - Say what is not found or is redacted; never infer `(b)(4)` text.
  - Report an unresolved conflict when equal-authority sources disagree and no replacement is declared.
- [ ] `AnswerModel = {answer, citations[{tag, quote}], evidence_gaps[]}`. `answer_question(client, question, context) -> (answer, usage, elapsed_ms, attempts)`; elapsed time measured around `extract`, usage from `usage_for_current_thread()`.
- [ ] Expand each cited tag into the contract citation: scope, source_id, source_version, passage_id, page, section (from the locator; may be null), quote.
- [ ] Commit.

## Task 4: record writers and the `answer` CLI

- [ ] Test with fake embedding and generation clients: every field in `schema/output-contract.json` (`answer_required_fields`, `trace_required_fields`) is present; `trace_id` links answer and trace; IDs are unique; `evidence_text` equals the text sent; P3 records satisfy `len(evidence_text) <= 6000`; a failed run produces a trace with `status="failed"` and no answer record.
- [ ] Field semantics:
  - `selected_passage_ids` = passages actually sent to the model. `candidate_passage_ids` = everything retrieved before the budget (kept separately).
  - Also: `document_ids`, `selected_via` per passage, `seed_node_ids`, `node_ids`, `traversed_edge_ids`, `hops`, `traversal_limits`, `max_evidence_chars`, `truncated`, `dropped`, `prompt_version`, `retriever_version`, `model`, `usage`, `attempts`, `elapsed_ms`, `citation_check`, `index_version`, `graph_version`, `corpus_snapshot` (from the config).
- [ ] CLI: `python -m potency_kg.cli answer --config configs/casgevy-potency.json --output results [--max-chars 6000]`.
  - Reads `questions.json`; embeds each question once; runs vector and graph with the same budget and prompt. Gates on `KG_RUN_REAL_MODELS`.
  - `results/answers.jsonl`, `results/traces.jsonl`: required seven (P3 both modes; the others `graph`).
  - `results/extra-mode/answers.jsonl`, `results/extra-mode/traces.jsonl`: the five other `vector` runs.
  - Output through `guard_output_path`; protected input hashes must be unchanged afterwards.
- [ ] Commit.

## Task 5: real-model fixture runs (item 3a plus isolation)

- [ ] Precondition (verified): the DB holds only `public-fda/casgevy-potency`. Fixture scopes (correction, conflict, isolation project one/two) are built for the first time with real models. This is the only DB write in the plan, confined to those scopes. Earlier test-double exports under `results/kg/fixtures/` stay on disk as offline checks.
- [ ] Add `allow_real` to `run_kg_fixtures` (default stays double-only so existing offline tests pass). Real mode requires `KG_RUN_REAL_MODELS` and uses `cli._embedding_client/_jev_client/_generation_client`.
- [ ] Add `on_stage(configuration, report)` called right after the stage's graph refresh and before the next stage runs, so the answer is generated while that stage's evidence is current. For the correction fixture the v1 answer happens before v2 replaces it.
- [ ] `run-fixture-answers` CLI:
  - Runs F-update before, F-update after (declared replacement) and F-conflict combined in `graph` mode, using the question from `fixtures/behavior-cases.json`. The same mode before and after.
  - The isolation scopes are built in the same run and in the same store (assignment item 1) but get no answers.
  - Writes `results/fixtures/answers.jsonl` and `traces.jsonl` (3 records, with `scenario_id`, `stage`, `run_id`), `results/fixtures/history/<scenario>/<stage>/` (source records, graph export, answer, trace), and `results/fixtures/report.json`.
  - `report.json` has: commands, checks (passed/failed/not_run), observed results, failures, and a `history_artifacts` map linking snapshot, index, source versions, answers and traces.
- [ ] Checks computed from saved traces and the DB:
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
  - Every `passage_id`, node ID and edge ID in a trace resolves in the DB and in `results/kg/current/graph/*.jsonl`.
  - No secrets in outputs (`rg` for key names and values from the secrets `.env`).
- [ ] Write the P3 vector vs graph comparison note (observed contribution or failure) in `docs/`.
- [ ] `graphify update .`.

## Risks and open items

- **Budget vs recall:** with about 3 passages per answer, P3 graph and vector may end up with the same evidence. Report that honestly; do not tune the merge after seeing P3 answers, beyond what is written here (any change must be disclosed).
- **Unranked traversal:** high-degree nodes (for example the CASGEVY product node) can fill the 50-edge limit with arbitrary edges. Graph passages are ranked by query similarity to limit the damage. If graph passages are mostly irrelevant, say so in the comparison note.
- **Quote validation:** the model may paraphrase. One repair retry is recorded. If many still fail, add a declared whitespace/hyphenation normalization rather than fuzzy matching.
- **Determinism:** `extract` exposes no temperature or seed, so reruns may differ. Saved traces, not reruns, are the record.
- **Fixture build** is the only DB write. The correction fixture must run in order. A re-run needs fresh fixture scopes or dropping those scopes only; out of scope here.
- **Limits:** if the caps prove too small, raise them in one place and re-run both modes; never change them for one mode only.

## Changes from draft v1

- Removed the fixture "reset"; the DB has no fixture scopes (verified).
- Added Task 0 (preflight), per-hop traversal info, exclusion of structural edges and document nodes from graph passage collection, and similarity ranking of graph passages instead of hop-first.
- The model now cites `S#` tags, validated against the text actually sent (truncation-aware).
- Split `candidate_passage_ids` from `selected_passage_ids`; one embedding per question shared by both modes.
- Failed runs are recorded without stopping the batch.
- Added the isolation fixture to the real fixture run, with a containment check.
