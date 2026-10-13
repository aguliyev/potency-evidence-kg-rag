# Task 0 preflight report

Observed: 2026-10-13 23:57 UTC. This records preflight observations, not a timed-work start.

## Result

Preflight is incomplete. Docker access is denied in this environment, so the database status, runner, live `graph_scope` binding, query embedding, and generation smoke call could not be verified. No real model calls were made. The provider access sheet is still `not_configured`; paid terms, complete materials, smoke verification timestamps, and an official start time are all absent. The official timed-work readiness point has not begun.

## Checks

| Requirement | Outcome |
|---|---|
| DB container up | Not verifiable. Docker commands fail with permission denied for `/home/anar/.docker/desktop/docker.sock`. |
| `bin/runner` works | Failed to start: `bin/runner --help` reaches Docker Compose and is denied access to the Docker socket. No runner container was started. |
| Required environment fields | Present by name across `etc/.env` and `etc/secrets/.env`: `LLM_URI`, `LLM_MODEL`, `LLM_API_KEY`, `EMBEDDING_LLM_URI`, `EMBEDDING_LLM_MODEL`, `EMBEDDING_LLM_API_KEY`, and `TYPESAFE_API_KEY`. Presence was checked only; credential values were never printed or recorded. |
| `KG_RUN_REAL_MODELS=true` | Configured as `true` in `etc/.env`. |
| Model IDs | `claude-sonnet-5-5` (`LLM_MODEL`) and `text-embedding-3-large` (`EMBEDDING_LLM_MODEL`), read from `etc/.env`. The real provider response IDs are unknown because no calls were made. |
| Real query embedding and tiny real `extract` | Not run. Docker access failed, and the committed model access sheet documents no provider smoke instructions or confirmed budget. |
| `graph_scope.index_version` binding | Live database binding not verifiable without DB access. The existing history export manifest and its sibling `import-report.json` both report `index-563432d174c8967fa01b316ce20e425d93b249e5a20ec52f68ce3b4cda129f45` for scope `public-fda/casgevy-potency`; this confirms the two artifacts agree, not that the current database row is bound to it. |
| Timed-work readiness | Not reached. `configs/model-access.template.json` has `setup_status: not_configured`, `complete_materials_confirmed: false`, `paid_terms_confirmed: false`, no budget limit, no embedding or generation verification timestamps, and no start timestamp. No committed time-log template exists in the tracked files. |

## Commands and outcomes

- `graphify query "preflight real model smoke query embedding extract import-report index_version runner configuration"` — returned the scoped project graph; no files changed.
- `docker compose -f docker/compose.yaml --env-file etc/.env --env-file etc/secrets/.env ps --format json` — failed with Docker socket permission denied.
- `bin/runner --help` — failed with the same Docker socket permission denied; the help command did not run inside a container.
- Shell presence checks over `etc/.env`, `etc/secrets/.env`, and the current process environment — confirmed the required variable names as shown above; only presence was reported for secrets.
- `sed -n '10p;14p' etc/.env` — returned only the two non-secret model IDs reported above.
- `jq` over `results/kg/history/graph-307b3797ee0fe8db7151aabddc53c8616003d64a8e6b4f6cad79da6331d98c41/{manifest.json,import-report.json}` — confirmed matching scope and `index_version` in the existing export artifacts.
- `jq` over `configs/model-access.template.json` — confirmed access is unconfigured and the readiness fields are incomplete.
- `git ls-files` search for time-log/import-report templates — found no committed time-log template. `results/` contains an existing history export but no `results/time-log.md`.

## Calls, costs, and side effects

No embedding, extraction, generation, or other provider requests were made, so there is no usage or cost to report. The configured budget is unknown; therefore the cost of any future call cannot be established from the repository. The Docker commands were read-only checks or a help invocation, but both stopped before reaching the daemon. No database mutation or provider side effect occurred.

No `results/time-log.md` was created: the official readiness point has not begun and no committed source template exists. No elapsed time or active-work interval is inferred or invented.

## Files and commit

- Added this required non-sensitive report only.
- No production code, README, access configuration, or time log changed.
- Commit: to be recorded after self-review.

## Self-review

- Secret values were not printed, copied, or added to this report.
- The lack of a live DB check is distinguished from the matching historical export artifacts.
- No real model call, readiness timestamp, elapsed time, usage, or cost is claimed.
- No tests were run; Task 0 adds no production behavior.
