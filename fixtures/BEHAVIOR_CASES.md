# Source correction and conflict — synthetic acceptance cases

These are fictional engineering fixtures, not FDA or customer facts. They supplement the existing project-isolation fixture. Keep all fixture scopes separate from the official corpus and from each other.

The machine-readable input contract is [behavior-cases.json](behavior-cases.json). Paths in `input_records` are relative to the candidate package root. Only the four JSONL records under `fixtures/inputs/` named by the scenarios are evidence. Do not embed questions, expected attributes, check lists, or this guide. Revision activation and scope metadata may drive your ordinary source-management code.

## Source correction

1. Import `source-correction-v1.jsonl` into the correction scope; run F-update with one chosen retrieval mode. Preserve its source record, graph/assertions, answer and trace with version/run identifiers.
2. Apply `source-correction-v2.jsonl` as the explicit replacement for the same logical source and passage. The new record changes one assertion: the measured attribute. Apply the declared replacement through your generic configuration/import path.
3. Run the same question and mode using the current revision. Save the new answer, trace and graph. The old assertion must not remain active or enter the default current evidence context; the old source/assertion/answer/trace remain inspectable in archived artifacts. Use source-versioned citations that resolve to the actual texts.

Full reconstruction of the current store is acceptable if the old exports survive. Two exports and an explicit run/source map are enough for history. You do not need incremental update algorithms, an as-of query endpoint, or a history UI. A source-file correction is separate from an earlier proposal described inside an otherwise current FDA source; preserve that proposal as historical evidence where relevant.

## Unresolved conflict

Import both conflict notes into the conflict scope and run F-conflict. They concern the same fictional product, indication, plan and effective date, have equal source authority, and contain opposing assertions. Both remain current; no replacement is declared. The issue dates deliberately differ.

Use the retrieved original evidence to retain and cite both claims. State that the conflict is unresolved when there is no supported resolution. Do not invent an authority hierarchy, silently discard one claim, or use the later issue date alone as an override. This scenario requires preserving disagreement; no expert arbitration or human approval operation is required.

## Submit actual execution artifacts

Save three answers and matching actual generation traces in `results/fixtures/answers.jsonl` and `results/fixtures/traces.jsonl`: F-update before, F-update after, and F-conflict combined. Include the normal answer/trace fields plus `scenario_id`, `stage`, and `run_id`. Use your agreed model access; offline test doubles are additional checks, not a substitute for these runs.

In `results/fixtures/report.json`, include scenario IDs, reproducible commands, checks (passed/failed/not_run), observed results, failures, and artifact references. `history_artifacts` links both correction runs to their preserved source files, graph exports, answers and traces using snapshot, index and source-version identifiers. See [output-contract.json](../schema/output-contract.json) for logical fields; equivalent documented layouts are acceptable. Keep answer/trace IDs unique and resolve relative artifact paths from `results/`.

The supplied cases are visible acceptance fixtures. Report them as supplied/visible if included in regression evaluation; they are not candidate-authored holdout cases. Their three answers remain separate from the seven official-corpus answers and the existing project-isolation demonstration. Do not combine any synthetic evidence into P1–P6.
