# Graph signal measurement

## Review rule, set before the real query runs

A candidate is **directly relevant** only when its original passage directly supports a factual part of the question without an inference through another source. A passage that merely names a related product, assay, regulation, or guidance is negative. A candidate that requires another passage to establish the requested relationship is negative. Review uses the complete original passage, not graph attributes or a generated summary.

The planned comparison uses P1, P2, P4, P5, and P6. P3 remains held out until the traversal and scoring configuration are frozen. A and D receive the same query embedding and vector hits. The measurement script is set up to write `docs/graph-signal-candidates.json` with every non-vector candidate's rank, passage/source IDs, mode, cosine and graph score, hop, path, hub flag, and review fields. Reviewers must fill the verdict and one-sentence rationale for every row. This note will summarize counts and the traversal decision after that review.

## Measurement status

The focused DB integration test was green before the attempted real measurement. The local active graph is `public-fda/casgevy-potency`, and a read-only count found 791 bound index members and 1,188 evidence edges. The requested five-query measurement could not run: automatic approval review rejected `bin/runner python .superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py` twice. Its stated reason was that sending the five corpus-derived question texts to the configured external embedding provider lacked specific authorization for that payload and destination, including after the Task 1 brief's authorization was quoted. No real query embedding, candidate ledger, cosine distribution, A/D relevance verdict, or P3 retrieval was produced.

Variant D is implemented as the traversal decision. The existing hop and potency bonuses remain provisional defaults because the required real cosine distribution was blocked. They were held fixed for the later successful P3 pair, but must not be described as measurement-calibrated. If future approval permits the five-query measurement, review every ledger row under the rule above. Any resulting bonus change would require disclosure and a new paired P3 run.

## Task 7 follow-up

The completed main answer batch later ran both P3 modes with the unchanged provisional configuration, and its observed comparison is recorded in [P3 vector versus graph comparison](p3-vector-graph-comparison.md). This does not retroactively satisfy the planned pre-P3 A/D calibration. A Task 7 attempt to run the saved five-question measurement was again rejected by automatic approval review because it would send corpus-derived question texts to the external embedding provider without separate authorization for that payload and destination. No `docs/graph-signal-candidates.json` ledger or per-candidate reviewer verdicts were created. The script was not rerouted or retried indirectly. Any future A/D measurement after P3 can be descriptive only; choosing different bonuses would require disclosure and a fresh paired P3 run.

The prepared A comparator walks the same activated AGE node and edge snapshot as D. It uses the smoke traversal's edge-ID ordered breadth-first bounds (two hops, at most 30 nodes and 50 edges), with the specified hub rule: an entity with more than 20 projected evidence relations cannot seed or expand, but it can be visited as a neighbor and contribute a valid source passage. It uses only current scoped, bound-index source references for candidate text. Its hub-neighbor behavior is covered by the offline regression test; no provider-backed A/D result is claimed.
