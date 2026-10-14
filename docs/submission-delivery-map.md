# Submission delivery map

Updated 2026-10-14. The delivery audit failed because the saved real fixture report lacks the contract-required `scenarios` field. No `results/delivery-manifest.json` was written. This map records the partial handoff and its limits; no successful manifest or artifact hash map is claimed.

| Deliverable | Producer/status | Check |
|---|---|---|
| `results/import-report.json`, `results/graph/nodes.jsonl`, `results/graph/edges.jsonl` | Existing real graph export, copied from `results/kg/current/` | Byte-for-byte copy check in `verify-deliverables` |
| `results/answers.jsonl`, `results/traces.jsonl` | Existing real model output, seven records each | Schema, counts, citation/trace links |
| `results/fixtures/answers.jsonl`, `traces.jsonl`, `report.json`, `history/` | Existing real fixture runs, three records each; **incomplete contract** | Saved `report.json` has no required `scenarios` field; preserved unchanged |
| `results/evaluation/cases.jsonl` | Frozen candidate-authored labels | E1–E5, with holdout freeze metadata; labels remain provisional |
| `results/evaluation/results.jsonl`, `report.json` | Offline evaluator | Five rows, all unscorable because no saved matching E-question trace; aggregates recompute from rows |
| `results/architecture/architecture.md`, `.mmd`, `data-contracts.md` | Manual design | Workload, isolation, lifecycle and dated AWS price assumptions documented |
| `results/time-log.md` | Manual log | Actual saved usage and exact known commands; earlier elapsed time unavailable |
| `results/delivery-manifest.json` | **Failed / not written** | `verify-deliverables` stopped on fixture report contract violation; downstream checks were not claimed |
| Root `README.md` handoff | **Blocked** | Repository instruction requires explicit user authorization to edit it; left unchanged |

Final Stage 3 commands and outcomes:

1. `bin/runner pytest tests/unit tests/integration` — final run: 353 passed; 135 Alembic warnings.
2. `answer` — not rerun in Stage 3; seven saved real answer/trace records retained.
3. `run-fixture-answers` — not rerun in Stage 3; three saved real fixture records and history retained.
4. `bin/runner python -m potency_kg.cli evaluate --results results --cases results/evaluation/cases.jsonl` — wrote five explicit unscorable case rows.
5. `bin/runner jupyter nbconvert --to notebook --execute notebooks/06_retrieval_and_answering_examples.ipynb --output /tmp/potency-task8-executed.ipynb` — executed, temporary copy not submitted.
6. `bin/runner python -m potency_kg.cli verify-deliverables --results results --output results/delivery-manifest.json` — failed on the missing fixture-report `scenarios`; no manifest written.
7. Secret scan — the verifier's full scan was not reached after the contract failure; a separate read-only scan is recorded in the Task 8 report.

The evaluation failure analysis is a coverage gap: E1–E5 were frozen before the main run but are different questions from P1–P6. The offline evaluator has no authorization to issue new model calls, so no retrieval or answer quality denominator exists for these cases. This is not a zero score and must not be reported as one. A future real E-case run would require a separate deliberate provider execution and preserved outputs. Other limitations: no independent human adjudication of candidate labels, provider dollar rates unavailable, production costs are hypothetical, and historical work intervals are unlogged. The next three steps are to repair the fixture report through an authorized provenance-preserving producer path, run/preserve the E-case attempts, and price/load-test the proposed AWS deployment before claiming its budget or freshness targets.

SHA-256 of current saved artifacts (captured after the final offline evaluation; these are inventory hashes, not a passing contract certification):

| Relative to `results/` | SHA-256 |
|---|---|
| `import-report.json` | `e9dcf8d6eefab2ef3c1ada6ae9695f59c389a251f86e7d3c8ca15a2c8773c03e` |
| `graph/nodes.jsonl` | `231fd4c94f35837e5f5a23f3855b4f6585396adc80c39ddb883faa32b39cf45f` |
| `graph/edges.jsonl` | `5d5ac69cf2dea9e05ceec32056b2cac14314b7fa27d98193d3c636968263d654` |
| `answers.jsonl` | `c6fd42f92fd2201a2a915c20bc39de34083602518bac5ceb099e7b3b96f4db45` |
| `traces.jsonl` | `2df646b506a91c1b66fc4f6b67b956df3cf099730385b096d842eb0640f9c09e` |
| `fixtures/answers.jsonl` | `4ffe0fb68a54b990b35f28c06a972367a7af786056636c9d8a0e758d8c28211b` |
| `fixtures/traces.jsonl` | `ff955aaf4e6b82031e4d7b9f02c0047323615d293786579850a8058a6d80b60a` |
| `fixtures/report.json` | `0d94df02ec94e56fe6d9ee4488af7757cb4b38ff90929ee0ae0eee4c4aeacdec` |
| `evaluation/cases.jsonl` | `0913887ad1acc30f0ac27778fae41715329a8f4dc2ad2542b4a051c8054df700` |
| `evaluation/results.jsonl` | `601cb27f05045e9a9727f3f2797f3dcccfe940d31f79032a9d6167db44b617f8` |
| `evaluation/report.json` | `3e56dd9e8ded13b0ad34c1dfefde1f4678a5bef5cdd9271147ded06d48549aaf` |
| `architecture/architecture.md` | `6fe0f2fd8b8244f4680075a30b2cf331ffb96034ea970c4b54acfa8b48f4c8f5` |
| `architecture/architecture-diagram.mmd` | `80c5273169c959a852f29c73517e242cf8c9a626ee16e6b81d0a5c7b4be207a0` |
| `architecture/data-contracts.md` | `931fde475021d9d1e5c1d4a779dd18cbc2298d02cbfbd5af3b1ae1622801fda6` |
| `time-log.md` | `250971d218690d6fe5c3fd12a0a88f9fd4196a350201cfd3068570f4639a411b` |
