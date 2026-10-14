# Task 2 report — merge and context building

## Result

Implemented a pure `lib/potency_kg/context.py` layer with `EvidenceBlockRenderer`, `merge_candidates(candidates, mode, max_chars, config)`, and `build_context(candidates, max_chars, config)`. The merge returns immutable `SelectedPassage` records carrying the original `RetrievalCandidate`, exact selected text, and truncation flag. The context result includes `evidence_text`, `tag_to_key`, exact `sent_text`, `chars_used`, kept character counts for truncated passages, dropped keys, and ordered selected passages.

The renderer counts Unicode code points in the actual headers, passage text, and `\n\n` block separators. Vector mode uses the full cap. Graph mode reserves the configured share, charges overlap (`both`) to vector allocation, then fills with graph-only candidates and optionally returns unused share to remaining vector candidates. Ranking and deduplication use stable passage-key tie breaks. The document cap applies during filling. Context selection happens before document-order presentation, so changing `context_order` cannot change the selected passages. Headers include available source ID/version, source type, authority, page/section, and passage ID. Titles and graph descriptions are absent.

## Red and green evidence

- First test run: `PYTHONPATH=lib .venv/bin/python -m pytest tests/unit/test_context.py -q` failed collection with `ModuleNotFoundError: No module named 'potency_kg.context'` before implementation.
- After the first implementation, the focused run reported 3 failures / 4 passes; the failures exposed incorrectly hand-counted test budgets. Corrected literal budgets and expected prefix length to account for the complete header and separator.
- A subsequent selection-invariance test failed: score order chose `high`, but document order chose `low` under the same cap. The renderer now selects before document ordering. Final focused run: `8 passed in 0.27s`.
- `PYTHONPATH=lib .venv/bin/python -m pytest tests/unit -q` reported `1 failed, 108 passed in 0.59s`. The only failure is unrelated: `tests/unit/test_alias_discovery_llm.py::test_llm_alias_discovery_is_disabled_unless_explicitly_enabled`, with `assert DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY is False` but actual value `True`. `lib/potency_kg/alias_discovery.py:27` defines it as `True`. Recent history includes `07652e4 feat: enable LLM alias discovery by default and update knowledge graph notebook outputs`; Task 2 did not edit that file, constant, or test.
- `graphify update .` exited 0 after code modification, rebuilding 1,695 nodes and 4,650 edges. Its generated files remain ignored. `git diff --check` exited 0.

## Files and self-review

- `lib/potency_kg/context.py`: pure selection, allocation, rendering, result types.
- `tests/unit/test_context.py`: eight focused tests covering exact sent text and budget, Unicode, truncation, skip versus cut, tag ordering, selection invariance, share reservation, share return, overlap ownership, floor, deduplication, and document cap.
- This report is stored at the explicitly requested `.superpowers/sdd/2026-10-13-retrieval-answering/task-2-report.md` path.

The functions perform no provider or database calls. All source-derived context is visible in `evidence_text`; there is no hidden graph-description channel. A `SelectedPassage` retains the original candidate even when its sent text is a shorter prefix. `max_chars` is explicit and validated in both functions. Task 3/4 will need to pass the question-specific cap and record it in their trace; that driver work is outside Task 2. The existing unrelated `.idea/deffai-potency-kg-rag-assignment.iml` modification was left untouched and excluded from the Task 2 commit.

## Commit

`Implement budgeted evidence merge and context rendering` (this report is included in the same commit).

## Review fixes

The review identified two allocation/presentation defects. Added two regression tests first. `PYTHONPATH=lib .venv/bin/python -m pytest tests/unit/test_context.py -q` returned **2 failed, 8 passed**: with share return disabled, a second graph passage consumed unused vector allocation; and `context_order="score"` left a lower-score vector passage before a higher-score graph passage. After the fixes, `PYTHONPATH=lib .venv/bin/python -m pytest tests/unit/test_context.py tests/unit/test_retrieval_config.py -q` returned **26 passed in 0.31s**.

Graph allocation now has a strict phase limit equal to its reserved share plus the vector cost already used when `return_unused_graph_share=False`. With the flag enabled, unused vector allocation can be returned to graph-only passages; if graph candidates run out, remaining capacity can still return to unused vector passages. Final score presentation sorts selected passages by descending score and passage key after selection, so score order and document order share the same selected set. No alias behavior, `.idea` file, or provider path was changed.

Fresh full unit run after the review fixes: `PYTHONPATH=lib .venv/bin/python -m pytest tests/unit -q` returned **1 failed, 110 passed in 0.54s**; the sole failure remains `test_llm_alias_discovery_is_disabled_unless_explicitly_enabled` with the same `True is False` assertion described above. `graphify update .` exited 0, rebuilding 1,704 nodes and 4,670 edges.
