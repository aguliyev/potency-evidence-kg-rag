# Graph signal measurement

## Review rule, set before the real query runs

A candidate is **directly relevant** only when its original passage directly supports a factual part of the question without an inference through another source. A passage that merely names a related product, assay, regulation, or guidance is negative. A candidate that requires another passage to establish the requested relationship is negative. Review uses the complete original passage, not graph attributes or a generated summary.

The planned comparison uses P1, P2, P4, P5, and P6. P3 remains held out until the traversal and scoring configuration are frozen. A and D receive the same query embedding and vector hits. The measurement script is set up to write `docs/graph-signal-candidates.json` with every non-vector candidate's rank, passage/source IDs, mode, cosine and graph score, hop, path, hub flag, and review fields. Reviewers must fill the verdict and one-sentence rationale for every row. This note will summarize counts and the traversal decision after that review.

## Measurement status

The focused DB integration test was green before the attempted real measurement. The local active graph is `public-fda/casgevy-potency`, and a read-only count found 791 bound index members and 1,188 evidence edges. The requested five-query measurement could not run: automatic approval review rejected `bin/runner python .superpowers/sdd/2026-10-13-retrieval-answering/measure_task1.py` twice. Its stated reason was that sending the five corpus-derived question texts to the configured external embedding provider lacked specific authorization for that payload and destination, including after the Task 1 brief's authorization was quoted. No real query embedding, candidate ledger, cosine distribution, A/D relevance verdict, or P3 retrieval was produced.

Variant D is implemented as the confirmed traversal decision. The existing hop and potency bonuses remain provisional defaults because the required real cosine distribution was blocked. They must not be described as measurement-calibrated or frozen. After approval for the external embedding payload and destination, run the saved measurement script, review every ledger row under the rule above, set bonuses from the observed spread, and re-run A/D before any P3 or answer generation.
