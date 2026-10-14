# P3 vector versus graph comparison

Both successful P3 records used the same `potency-assignment-2026-09-21-v3` corpus snapshot, active graph version `graph-307b3797ee0fe8db7151aabddc53c8616003d64a8e6b4f6cad79da6331d98c41`, bound index version `index-563432d174c8967fa01b316ce20e425d93b249e5a20ec52f68ce3b4cda129f45`, config hash `4ab6cf693c09b656623c2c024deddb016dfe44c870d0617f3e01eca54c06b029`, and `answer-prompt-v1` hash `3c10d8aee0345061c2de0eef84be1d7cfc18298c3bcb20b252485285aacd7586`. Both used the configured `muse-spark-1.3-contributor` generation model. Each sent exactly 6,000 Unicode code points of source evidence, including headers.

| Mode | Retrieval candidates | Selected passages | Graph-only selected | Traversed edges | Generation tokens (prompt / completion) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 8 | 3 | 0 | 0 | 1,937 / 2,979 |
| Graph | 13 | 4 | 2 | 7 | 1,995 / 2,841 |

Graph traversal contributed two passages absent from the vector hit set: `sbra-casgevy-tdt-2024-01:page-0004-c02`, on approval and off-target safety PMRs, and `sbra-casgevy-scd-2023-12:page-0013`, on container-closure testing. Neither directly addresses P3's potency proposal, FDA challenge, change, and reviewer conclusion. The other two graph selections were also vector hits. The vector answer likewise reported insufficient potency evidence in its selected passages. Thus graph mode demonstrably changed evidence selection, but did not improve the P3 answer under this configuration; it spent reserved context on adjacent safety and container information. Both saved answers state the potency chronology cannot be determined from their inspected evidence. This is a judgment about the supplied P3 evidence blocks, not a claim that the full corpus lacks relevant passages.

The exact evidence, selected passage versions, citations, and retrieval paths are in `results/traces.jsonl` and `results/answers.jsonl`. See [retrieval method and configuration](retrieval-method.md) for the budget rule and frozen run settings. The planned five-question A/D candidate ledger and human relevance verdicts were not produced after the earlier approval rejection, as [graph-signal measurement](graph-signal-measurement.md) records. The bonuses therefore remain provisional; no P3-informed tuning was applied.
