# Retrieval, Answering and Fixture Runs Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Produce the required answer/trace artifacts (assignment items 3 and 3a): `vector` and `graph` retrieval over the existing store, bounded context building, citation-validated generation, a driver for the 6 questions, and real-model runs of the correction and conflict fixtures.

**Not in scope (later plans):** evaluation set/scoring, AWS architecture docs, README handoff, time log.

**Decisions (confirmed with the user):**
- Graph mode is merged and graph-ranked: vector hits seed the traversal; traversed nodes and edges contribute their source-ref passages; both lists are merged and then filled into the character budget.
- Every question runs in BOTH modes (12 runs). The required seven records are P3 (both modes) plus P1/P2/P4/P5/P6 in `graph` mode. The five extra `vector` runs are kept in a separate directory.
- The LLM returns JSON `{answer, citations[{passage_id, quote}], evidence_gaps}`. Quotes are validated locally as exact substrings of the evidence sent.
- Fixtures use `graph` mode, with the same mode before and after the correction.

## Reuse (already in `lib/potency_kg/`)

| Need | Existing piece |
|---|---|
| Query embedding and vector search | `embeddings.embed_query`, `vector_search` |
| Active graph, index binding, bounded AGE traversal | `retrieval._active_graph_binding`, `age_projection.bounded_graph_traversal` |
| Node/edge to passage links | tables `kg_node_source_ref`, `kg_edge_source_ref` |
| Structured LLM call plus usage | `llm_client.InstructorExtractionClient.extract(model_text, response_model, prompt)`, `usage_for_current_thread()` |
| Config/scope, protected-input guard, path guard | `cli._scope_from_descriptor`, `output_safety.guard_output_path` |
| Fixture import/embed/extract/graph/export | `fixture_runs.run_kg_fixtures` (currently test-double only) |

No new LLM client is needed: pass a Pydantic `AnswerModel` to `extract`.

## Design facts that shape the plan

- The median passage is about 1,800 chars (90th percentile about 3,300). A 6,000-char cap therefore fits only about 3 passages. What gets ranked first decides the vector vs graph comparison, and the last passage should be truncated to fit rather than dropped.
- `retrieve_graph_smoke` returns `source_refs` only for vector hits, so the graph does not select passages today. That is the main code change.
- Retrieval limits are capped at the "smoke maxima" (8 hits, 8 seeds, 2 hops, 30 nodes, 50 edges). Keep them as defaults and declare them in the trace.

## File map

- Modify `lib/potency_kg/retrieval.py`: add `graph_passages(...)` and `retrieve(engine, scope, question, mode, embedding_client, index_version, limits)`. Keep `retrieve_graph_smoke` as a thin wrapper so existing tests still pass.
- Create `lib/potency_kg/answering.py`: `merge_candidates`, `build_context`, `AnswerModel`, `validate_citations`, `answer_question`, and JSONL record builders.
- Modify `lib/potency_kg/cli.py`: add subcommands `answer` and `run-fixture-answers`.
- Modify `lib/potency_kg/fixture_runs.py`: allow real clients behind the `KG_RUN_REAL_MODELS` gate, and add an `on_stage` callback so the answer is generated while the stage's state is current.
- Create `lib/potency_kg/prompt_resources/answer_system.md` (prompt text, versioned as `answer-prompt-v1`).
- Create tests: `tests/unit/test_answering.py`, `tests/integration/test_retrieve_modes.py`, `tests/integration/test_answer_cli.py`.

## Task 1: graph-selected passages in `retrieval.py`

- [ ] Write the failing integration test (pattern: `tests/integration/test_retrieval_smoke.py`). Build a tiny scope where passage A is a vector hit linked to a node, an edge from that node carries source-ref passage B, and B is NOT a vector hit. Assert `retrieve(mode="graph")` returns B with `selected_via="graph"` and `retrieve(mode="vector")` does not. Assert a second scope with the same IDs never appears.
- [ ] Implement `graph_passages(connection, scope, node_ids, edge_ids)`. Join `kg_node_source_ref`/`kg_edge_source_ref` to `source_passage` and return (passage key, text, text_sha256, locator, min hop, via edge/node ID). Hop distance comes from the traversal order already in `BoundedGraphTraversal` (extend it to return per-element hop if absent).
- [ ] Implement `retrieve(...)`:
  - `vector`: embed, `vector_search`, load passages. Skip graph work (`seed_node_ids=()`, no edges).
  - `graph`: run the existing flow, then add `graph_passages`. Compute query-passage similarity for graph-only passages in SQL against `embedding_run` so every candidate has a score.
  - Return a `RetrievalResult` holding the original `RetrievalTrace` fields plus `candidates[]` (key, `selected_via` in {vector, graph, both}, score, hop, via IDs) and the `document_id` (source_id) of every passage.
- [ ] Run the new test and the existing `tests/integration/test_retrieval_smoke.py`, then commit.

## Task 2: merging and context building (pure functions, `answering.py`)

- [ ] Unit tests first:
  - `merge_candidates` for vector mode returns hits in score order.
  - For graph mode it interleaves round-robin: the vector list by score, the graph list by (hop, score). It dedups by passage key and marks a passage found by both as `both`. It is deterministic.
  - `build_context(candidates, max_chars=6000)`:
    - Output is `[S1] source_id | version | page/section | passage_id` plus the exact text per block.
    - The total `len(evidence_text)` (Unicode code points) is `<= max_chars`.
    - Whole passages are added in order. The final passage is truncated to a verbatim prefix if at least 300 chars remain, and `truncated=true` is recorded for it.
    - It returns `evidence_text`, per-passage `[S#] -> key` map, `chars_used` and `dropped` keys.
- [ ] The character budget is a function argument (`max_chars`, default 6000) and is threaded through `answer_question`, the `answer` CLI (`--max-chars`) and the trace (`max_evidence_chars`). The P3 submission uses the default for both modes.
- [ ] Implement both. Only passage text and headers are source-derived content. No graph descriptions are sent. This keeps the P3 cap trivially auditable.
- [ ] Commit.

## Task 3: generation with local citation validation

- [ ] Unit tests with a fake `extract` client:
  - A valid answer passes.
  - A quote that is not a substring of the cited passage text fails.
  - A `passage_id` not in the context fails.
  - After one repair retry that also fails, the record is saved with `citation_check.status="failed"` and is never silently fixed.
  - An empty-evidence case yields an answer that states nothing was found in the inspected scope.
- [ ] Write the `answer-prompt-v1` system prompt:
  - Answer only from the evidence.
  - Cite `passage_id` plus a verbatim quote.
  - Distinguish regulation, guidance, applicant proposal, reviewer conclusion and final action.
  - Say what is not found or is redacted. Do not infer `(b)(4)` text.
  - Report an unresolved conflict when sources disagree and no replacement is declared.
- [ ] Implement `AnswerModel`, `validate_citations` and `answer_question(client, question, context) -> (answer, usage, elapsed_ms, attempts)`. Measure elapsed time around `extract`, and read usage with `usage_for_current_thread()`.
- [ ] Build the citations output with the contract fields: scope, source_id, source_version, passage_id, page, section, quote. Page and section come from the passage locator and may be null.
- [ ] Commit.

## Task 4: record writers and the `answer` CLI

- [ ] Test (fake embedding and generation clients, fixture scope): `answer` writes JSONL with every field in `schema/output-contract.json` (`trace_required_fields`, `answer_required_fields`). `trace_id` links answer to trace, and ids are unique. `evidence_text` equals the text sent to the model. The P3 records have `len(evidence_text) <= 6000`.
- [ ] Trace fields beyond the contract: `selected_via` per passage, `document_ids`, `node_ids`, `seed_node_ids`, `traversed_edge_ids`, `hops`, `limits`, `truncated`, `dropped`, `prompt_version`, `model`, `usage`, `attempts`, `citation_check`, and `corpus_snapshot` (from the config).
- [ ] CLI: `python -m potency_kg.cli answer --config configs/casgevy-potency.json --output results`.
  - It reads `questions.json`, runs all six questions in vector and graph mode with the same budget (6,000) and prompts, and gates on `KG_RUN_REAL_MODELS`.
  - It writes `results/answers.jsonl` and `results/traces.jsonl` (the required seven: P3 both modes, the rest graph).
  - It writes `results/extra-mode/answers.jsonl` and `results/extra-mode/traces.jsonl` (the other five vector runs).
  - Output goes through `guard_output_path`, and the protected input hashes must stay unchanged.
- [ ] Commit.

## Task 5: real-model fixture runs (item 3a)

- [ ] Precondition (verified): the DB holds only the main scope `public-fda/casgevy-potency` (24 sources, one embedding version, one active graph). The fixture scopes are not in the DB, so no reset or cleanup is needed. The main knowledge base is not touched by this plan. Fixture scopes are built for the first time with real models; this is the only DB write in the plan, and it is confined to the fixture scopes. The earlier test-double fixture exports under `results/kg/fixtures/` stay on disk as offline checks.
- [ ] Add `allow_real` to `run_kg_fixtures`. The default stays double-only so the existing offline tests still pass. Add the `on_stage(configuration, report)` callback so the answer is generated right after the stage's graph refresh, before the next stage replaces the source.
- [ ] `run-fixture-answers` CLI:
  - It runs F-update before, F-update after and F-conflict combined in `graph` mode, with the question text taken from the `behavior-cases.json` question field.
  - It writes `results/fixtures/answers.jsonl` and `results/fixtures/traces.jsonl`, with `scenario_id`, `stage` and `run_id` added.
  - It writes `results/fixtures/history/<scenario>/<stage>/` (source records, graph export, answer, trace) and `results/fixtures/report.json`.
  - The report holds the commands, the checks (grounded in v1, grounded in v2, v1 assertion not current in the after context, both conflict claims cited, conflict stated unresolved, versioned citations resolve exactly) as passed/failed/not_run, the failures, and the `history_artifacts` map.
- [ ] Checks are computed from the saved trace and the DB: citation `source_version` equals the stage's current version, and the superseded passage ID is absent from the after-stage evidence. Conflict resolution wording is checked by a keyword check plus the fact that both passage IDs were cited. Mark that part as heuristic, not a human judgement.
- [ ] Commit.

## Task 6: run, verify, document

- [ ] Run `bin/runner pytest tests/unit tests/integration`. It must be green, including the unchanged re-import and isolation tests.
- [ ] Real run: `bin/runner python -m potency_kg.cli answer ...`, then the fixture command. Spot-check 3 answers by hand: open the cited passages and confirm the quotes. Record token usage and cost.
- [ ] Verify: the 7 and 5 record counts, P3 `evidence_text` lengths per mode, that every `passage_id` and `edge_id` in a trace resolves in the DB and in `results/kg/current/graph/*.jsonl`, and that no secrets appear in the outputs (`github_run_secret_scanning` or `rg` on `LLM_API_KEY`).
- [ ] Write the P3 vector vs graph comparison note (observed contribution or failure) into `docs/` for the README later.
- [ ] Run `graphify update .`.

## Risks and open items

- **Budget vs recall:** about 3 passages fit in 6,000 chars. If P3 graph and vector end up with identical evidence, report that honestly. Do not tune the ranking after seeing P3 answers beyond what is documented here.
- **Quote validation:** the model may paraphrase. The single repair retry is recorded. If many quotes still fail, add a normalization of whitespace and hyphenation (declared, not hidden) rather than loosening to fuzzy matching.
- **Fixture scope build** is the only DB write. It is confined to the fixture scopes, which are new, and the correction fixture must run in order (import v1, answer, replace with v2, answer). Re-running it needs a fresh fixture scope or a drop of those scopes only, which is out of scope here.
- **Limits:** if the 8-hit cap proves too small for the extra-mode comparison, raise the caps in one place and re-run both modes. Do not change them for one mode only.
