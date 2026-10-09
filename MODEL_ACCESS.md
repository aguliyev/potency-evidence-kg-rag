# Model access and the start of timed work

DeffAI provides approved embedding and generation access, or agrees a local alternative. **Personal paid model credits are not required.** This file and `configs/model-access.template.json` are a handoff checklist; unset fields do not constitute working access.

## What DeffAI supplies before the clock starts

| Item | Required information |
|---|---|
| Provider and endpoints | Provider for each service, exact embedding/generation URLs or local service invocation, region/project if relevant, API style/version and a working minimal request. They may use different providers. |
| Credentials | The approved secure delivery channel, authentication scheme/header and environment-variable names, candidate scope and expiry. Credentials are delivered separately; no keys go in the ZIP, repository, screenshots or trace exports. |
| Models | Exact embedding and generation model IDs/versions, embedding dimensions when fixed, and any approved substitutes. Do not silently change the generation model between P4 modes. |
| Budget | Candidate-specific total allowance in currency and/or tokens, included services, expiry/reset, how usage is visible, and what happens near the cap. Include extraction, embedding, generation, evaluation/judge calls and necessary retries. No personal top-up is expected. |
| Throughput | Requests/minute, tokens/minute, max concurrent requests, embedding batch size, timeouts and retry/backoff guidance. State explicit limits or confirmed absence of provider-imposed limits; null means unknown. |
| API constraints | Embedding input limits, generation context/output limits, supported API/SDK format, available usage fields, tools/streaming if relevant, and structured output: JSON schema, JSON mode, or neither. Plain generation plus local JSON validation/retry is acceptable when structured output is unavailable. |
| Local alternative | A named, pre-approved embedding model and generation model, exact versions/quantization, license/download source, expected hardware and how it is supplied, startup command and any restrictions. If no local fallback is viable, say so; DeffAI restores or replaces hosted access. |
| Support | A named primary contact and reachable channel, backup contact, and expected response window during the candidate's work period. |
| Start conditions | Complete materials, confirmed paid terms and submission window, both successful smoke calls, and a recorded readiness/start timestamp. |

DeffAI should verify that the supplied limits can support this 791-record exercise and the documented P4 evidence budget. Candidates may choose efficient batching or selective semantic extraction; they are not expected to evade provider limits or fund extra capacity.

## Two small access checks

DeffAI supplies or demonstrates the provider-specific request commands. Candidates should not need to build an API adapter merely to establish access.

1. **Embedding:** submit a tiny synthetic sentence, such as “Access check: a fictional assay measures a fictional attribute.” Verify a successful response contains a nonempty finite numeric vector; record the actual model ID and vector dimension.
2. **Generation:** ask the generation model to return a short acknowledgement of that synthetic sentence. Verify nonempty generated content from the intended model, not merely an HTTP 200 or an authenticated model listing. If schema-constrained output is advertised, verify that capability with a tiny object. Its absence is acceptable when explicitly documented.

Record the provider/model IDs, time, sanitized result summary and any usage returned. Never record credentials. For a local alternative, run the same two checks against the agreed local setup. Test fixtures and smoke responses are not FDA evidence.

## When time starts, stops and resumes

The **eight-hour active-work clock starts only after both embedding and generation work**, the complete package and paid terms are available, and any material access/API constraints are communicated. Candidate and contact record the start time and the 14-calendar-day deadline.

Pre-start credential setup, these two tiny checks, fixing supplied access, and agreed local-model download/startup are outside timed work. Importing/indexing the FDA corpus, implementation, substantive experiments, evaluation, architecture, and handoff count toward the eight hours. A preflight result is not permission for unlimited untimed implementation.

For a team-provided outage, denied quota, unexpected model unavailability, or incorrect supplied constraints:

1. Save the time and a sanitized error/request ID; notify the access contact.
2. Pause while blocked. If you continue other exercise work, count that active work only.
3. Resume after restored access or a mutually agreed replacement passes both relevant checks. Record any model/version change and use a consistent model for the final P3 comparison.
4. Agree a deadline extension for confirmed access downtime if needed. No personal credits are required and there is no score penalty for documented team-provided access failure.

Normal development/debugging is timed. Avoid retry loops that consume the allowance; follow supplied backoff guidance and contact the team when the budget/limits block completion. If access cannot be restored, stop rather than fabricate results. Clearly separate offline test doubles from actual model runs.

## Minimal readiness record

Save these fields in your time log or a small local `access-check.json`:

```text
Package edition / corpus snapshot:
Paid terms and complete materials confirmed at:
Access mode: hosted | local
Embedding provider / model / success timestamp / vector dimension:
Generation provider / model / success timestamp:
Structured-output capability confirmed: json_schema | json_mode | none
Budget and rate/concurrency limits acknowledged:
Local fallback agreed, or unavailable with hosted recovery plan:
Primary and backup support contact:
Active-work start timestamp / timezone:
Submission deadline:
Any access interruptions and excluded duration:
```

No field in the shipped template is a promise that access has already been provisioned. DeffAI completes the access sheet before asking a candidate to start.
