# Task 1 report — graph-selected passages

Status: **offline implementation complete; real A/D measurement blocked by automatic approval review**. Do not run P3 or answer generation on these provisional score bonuses. Main implementation commit: `db6744c` (`Add graph-selected passage retrieval with scoped provenance`).

## Implementation

- Added `retrieve(...)` and immutable `RetrievalResult`/`RetrievalCandidate` records in `lib/potency_kg/retrieval.py`. Vector mode searches only the exact scoped bound index and loads original passage text. Graph mode reads the activated AGE topology, current scoped node/edge source refs, and pgvector query cosine for bound-index passages in one repeatable-read transaction. A short binding preflight after embedding must equal the binding in that transaction.
- Implemented the confirmed variant D: vector-ordered non-hub semantic seeds, undirected evidence-relation traversal, per-entity similarity-pruned top-N neighbors, per-hop global frontier cap, projected evidence-degree hub detection, candidate deduplication, graph score, provenance skips, supporting paths, hop/via IDs, and per-hop expansion diagnostics. Structural document/passage nodes never yield candidate passages. Source metadata comes from the exact scoped source-document version.
- Added `tests/integration/test_hybrid_retrieval.py` with A-vector/B-edge-evidence selection, structural/non-evidence exclusion, another scope with identical IDs, superseded source refs, rows outside the active projection, missing node/edge refs, hop-0 and traversed-edge provenance, source metadata, and a simulated binding mismatch. Added a focused hub-degree regression.
- Added `docs/retrieval-method.md`, `docs/graph-signal-measurement.md`, and an unexecuted one-off A/D measurement script at `.superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py`.

## Red/green evidence

| Check | Red result | Green result |
| --- | --- | --- |
| Graph-selected B outside vector top hit | `bin/runner pytest -q tests/integration/test_hybrid_retrieval.py` → 1 failed because `retrieve` was absent | Same command → 1 passed after implementation |
| Non-evidence relation must not return C | Same command → 1 failed because C was included | Same command → 1 passed after checking `is_evidence` |
| Source document metadata on B | Same command → 1 failed because candidate had no `source_metadata` | Combined suite → 5 passed after metadata join |
| Hub degree uses projected evidence relations, even when refs are missing | Focused `...::test_projected_evidence_degree_excludes_hub_seed_even_when_refs_are_missing` → 1 failed (`seeds == ('hub',)`) | Combined suite → 6 passed after degree accounting fix |

Final fresh verification: `bin/runner pytest -q tests/integration/test_hybrid_retrieval.py tests/integration/test_retrieval_smoke.py` → **6 passed, 5 Alembic deprecation warnings**, exit 0. `.venv/bin/python -m py_compile lib/potency_kg/retrieval.py tests/integration/test_hybrid_retrieval.py .superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py` → exit 0. `git diff --cached --check` → no whitespace errors before implementation commit. `graphify update .` → exit 0, rebuilt 1,648 nodes / 4,501 edges; generated graphify files remain ignored.

## Real measurement and exact blocker

Read-only local DB probe via `bin/runner python -c ...` initially returned `permission denied while trying to connect to the docker API`. Repeating through `bin/runner` with `sandbox_permissions=require_escalated` succeeded: one active binding for `public-fda/casgevy-potency`; counts were **791 index members, 4,154 nodes, 1,188 evidence edges**. The DB-backed tests were also run through the authorized runner.

After focused tests were green, `bin/runner python .superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py` was requested twice with escalation. Automatic approval review rejected both before execution. First reason: “The measurement sends corpus-derived questions to an external embedding service; the user authorized the implementation but did not specifically authorize this payload to that destination.” The retry quoted the Task 1 brief's explicit real-embedding permission; review still rejected it because “quoting the task brief does not establish the user’s specific authorization of that payload and destination.” No workaround or indirect provider call was attempted afterward.

Consequently there is no real cosine distribution, no A/D relevance verdict ledger, no observed per-question graph contribution or hub-mediated fraction, and no calibrated/frozen hop or edge bonuses. P3 remains held out. The unexecuted script is a prepared measurement tool, not measurement evidence. `docs/graph-signal-measurement.md` records the binary relevance rubric and blocker. The provisional `HybridRetrievalConfig` bonuses remain `(0.05, 0.05, 0.03, 0.01)` and `0.05` potency-edge bonus. Do not interpret them as selected from observed data.

## Self-review and remaining work

The integration test uses real migrated Postgres, pgvector, and AGE, including the activated graph snapshot. Original passage text is selected only through scoped relational source rows. Graph-only candidates carry their cosine, score, exact source version, path, hop, and hub marker. The repeated binding check fails before a retrieval result is returned; this retrieval function writes no answer record. Vector mode does not query AGE or source refs. The legacy smoke API remains present and its four tests pass.

Limitations: the measurement script has only syntax verification; its provider-backed run and candidate review are blocked. The A implementation in that script is for comparison only and is not a production retrieval branch. Bonus calibration, A/D threshold verdict, and P3 run require the blocked five embedding requests and reviewer pass. No answer generation was attempted. Root `README.md` and user-owned `docs/scratchpad.md` were not read or modified. PyCharm interpreter setup produced an unrelated dirty `.idea/deffai-potency-kg-rag-assignment.iml`; it was excluded from the commit.
