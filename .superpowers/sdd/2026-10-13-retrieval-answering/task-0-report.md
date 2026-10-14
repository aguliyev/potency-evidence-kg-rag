# Task 0 preflight report

Updated: 2026-10-14 00:01 UTC. This records preflight observations, not a timed-work start.

## Result

Preflight remains incomplete. After Docker access was approved, the DB and runner worked, and the live `graph_scope.index_version` matched the import report. One real query embedding succeeded. The tiny real structured extraction was attempted once and failed because the provider reported that the workspace usage limit had been reached (access returns 2026-11-01 00:00 UTC). No retry followed. The full access sheet still lacks paid terms, a confirmed budget, complete materials, and readiness confirmations; the official timed-work readiness point has not begun.

The initial Docker failures recorded in the first report were sandbox access failures, not evidence that the DB or runner were down. The later approved Docker checks below supersede those environment-limited results.

## Checks

| Requirement | Outcome |
|---|---|
| DB container up | Passed after Docker access approval. Compose reported `deffai-db` running and healthy. |
| `bin/runner` works | Passed. `bin/runner /usr/local/bin/python --version` returned Python 3.12.11; the CLI help command ran successfully in the container. |
| Required environment fields | Present by name across `etc/.env` and `etc/secrets/.env`: `LLM_URI`, `LLM_MODEL`, `LLM_API_KEY`, `EMBEDDING_LLM_URI`, `EMBEDDING_LLM_MODEL`, `EMBEDDING_LLM_API_KEY`, and `TYPESAFE_API_KEY`. Presence only was recorded for credentials; their values were never printed or saved. |
| `KG_RUN_REAL_MODELS=true` | Configured as `true` in `etc/.env`. |
| Configured model IDs | `claude-sonnet-5-5` (`LLM_MODEL`) and `text-embedding-3-large` (`EMBEDDING_LLM_MODEL`). The embedding response also returned `text-embedding-3-large`. The generation provider rejected the request before returning a model response. |
| Real query embedding | Passed once for the synthetic sentence “Access check: a fictional assay measures a fictional attribute.” Response: one finite vector, 3,072 dimensions, 11 total tokens, model `text-embedding-3-large`. |
| Tiny real `extract` | Attempted once with the same synthetic sentence and a minimal `acknowledged: bool` response model. Failed on attempt 1: provider returned HTTP 400 `invalid_request_error` stating the workspace API usage limit was reached and access would return at 2026-11-01 00:00 UTC. No extraction response or provider usage was returned. |
| Live index binding | Passed. `inspect-kg` reports active scope `public-fda/casgevy-potency`, graph `graph-307b3797ee0fe8db7151aabddc53c8616003d64a8e6b4f6cad79da6331d98c41`, and index `index-563432d174c8967fa01b316ce20e425d93b249e5a20ec52f68ce3b4cda129f45`. The index matches both the existing history `manifest.json` and sibling `import-report.json`. Live DB state reports 791 passages and 24 sources. |
| Timed-work readiness | Not reached. `configs/model-access.template.json` remains `setup_status: not_configured`; complete materials, paid terms, budget, both smoke timestamps, candidate acknowledgement, and start timestamp are absent. No committed time-log template exists. |

## Commands and outcomes

- `graphify query "preflight real model smoke query embedding extract import-report index_version runner configuration"` — returned the scoped project graph.
- Initial `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env ps --format json` and `bin/runner --help` — sandbox denied Docker socket access. These were rerun after Docker approval.
- Approved `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env ps --format json` — DB reported running and healthy.
- Approved `bin/runner /usr/local/bin/python --version` — returned `Python 3.12.11`.
- Approved `bin/runner /usr/local/bin/python -m potency_kg.cli --help` — succeeded and listed the CLI commands.
- Approved `bin/runner /workspace/.venv/bin/python -m potency_kg.cli inspect-kg --config configs/casgevy-potency.json` — the host venv path was not mounted at that location in the container; no task ran. Retried successfully with the image interpreter:
  `bin/runner /usr/local/bin/python -m potency_kg.cli inspect-kg --config configs/casgevy-potency.json`.
- Approved one-shot synthetic model smoke command through `bin/runner /usr/local/bin/python -c ...` — embedding succeeded (model, dimensions, tokens above); extraction returned the workspace-limit error on the first attempt.
- Shell checks of `etc/.env`, `etc/secrets/.env`, and host environment — reported only variable presence for credential names. Model IDs were read from the non-secret `LLM_MODEL` and `EMBEDDING_LLM_MODEL` settings.
- `jq` over the history `manifest.json`, `import-report.json`, and `configs/model-access.template.json` — confirmed artifact index agreement and the unconfigured readiness sheet.
- `git ls-files` search — no committed time-log template. Existing `results/` has a history export and no `results/time-log.md`.

## Calls, costs, and side effects

Exactly one embedding request and one generation request were made, both using synthetic text and the configured models above. Embedding usage returned 11 tokens. Generation failed with the provider workspace usage-limit response after one attempt and returned no usage. Dollar cost cannot be determined from repository data; the access sheet has no budget or pricing terms. No Jev request or other model call was made.

`inspect-kg` was read-only. No DB writes or corpus changes were made. The runner containers were ephemeral. The IDE Python setup helper created a local `.venv` and temporary IDE metadata. I reverted the `.idea` metadata changes and moved the generated environment to `/tmp/task0-preflight-generated-venv`; the repository worktree has no IDE-setting changes. No `results/time-log.md` was created because official readiness has not begun; no elapsed time or active-work interval is inferred.

## Files and commit

- Required artifact: this report only.
- No production code, README, access configuration, or time log changed.
- Initial report commit: `eb549e6` (`docs: record retrieval preflight results`). This update will be committed separately.

## Self-review

- Credential values were not printed, copied, or added to this report.
- Sandbox denial is distinguished from later successful Docker checks.
- DB binding is verified against the live `graph_scope` row and both history export artifacts.
- The embedding success and generation limit failure are both reported; no unsupported readiness claim is made.
- No timed-work start, active duration, generation usage, or dollar cost is invented.
- No tests were run; Task 0 adds no production behavior.
