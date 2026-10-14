"""Deterministic evaluation of saved answer and retrieval artifacts only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATOR_VERSION = "saved-artifact-evaluator-v1"
REQUIRED_RESULT_FIELDS = frozenset(("case_id", "case_version", "split", "run_id", "status",
                                    "retrieved_passage_ids", "expected_evidence_refs", "scores",
                                    "errors", "trace_id"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{number} must be a JSON object")
            rows.append(value)
    return rows


def _reference_key(value: Mapping[str, Any], scope: Mapping[str, Any]) -> tuple[str, ...]:
    ref_scope = value.get("scope", scope)
    return (str(ref_scope.get("tenant_id")), str(ref_scope.get("project_id")),
            str(value.get("source_id")), str(value.get("source_version")),
            str(value.get("passage_id")))


def _answer_checks(answer: Mapping[str, Any], checks: Mapping[str, Any]) -> dict[str, Any]:
    body = str(answer.get("answer", "")).casefold()
    outcomes: dict[str, bool] = {}
    for name in ("must_mention_all", "must_mention_any", "must_not_contain"):
        terms = checks.get(name, [])
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise ValueError(f"invalid {name} check")
        matches = [term.casefold() in body for term in terms]
        outcomes[name] = (all(matches) if name == "must_mention_all" else
                          any(matches) if name == "must_mention_any" and terms else
                          not any(matches) if name == "must_not_contain" else True)
    if checks.get("must_have_evidence_gap"):
        outcomes["must_have_evidence_gap"] = bool(answer.get("evidence_gaps"))
    return {"passed": all(outcomes.values()), "checks": outcomes}


def score_case(case: Mapping[str, Any], trace: Mapping[str, Any] | None,
               answer: Mapping[str, Any] | None) -> dict[str, Any]:
    """Score one frozen label against one saved attempt; never infer a missing run."""
    scope = case.get("scope")
    refs = case.get("expected_evidence_refs")
    if not isinstance(scope, dict) or not isinstance(refs, list):
        raise ValueError("case requires scope and expected evidence refs")
    expected = []
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("scope") != scope or not all(
            isinstance(ref.get(key), str) and ref[key] for key in ("source_id", "source_version", "passage_id")
        ):
            raise ValueError("expected evidence reference is invalid or outside case scope")
        expected.append(_reference_key(ref, scope))
    row: dict[str, Any] = {
        "case_id": case["case_id"], "case_version": case["case_version"],
        "split": case["split"], "run_id": None, "status": "unscorable",
        "retrieved_passage_ids": [], "expected_evidence_refs": refs,
        "scores": {"retrieval": None, "answer": None}, "errors": [], "trace_id": None,
    }
    if trace is None:
        row["errors"].append("no_saved_trace_for_case")
        return row
    if trace.get("scope") != scope:
        raise ValueError("saved trace outside case scope")
    row["trace_id"] = trace.get("trace_id")
    row["run_id"] = trace.get("run_id") or trace.get("trace_id")
    candidates = trace.get("candidate_passages") or trace.get("selected_passages") or []
    if not isinstance(candidates, list):
        raise ValueError("saved candidate passages must be a list")
    if not candidates and trace.get("status") != "succeeded":
        row["status"] = "failed"
        row["errors"].extend(("retrieval_not_recorded", "generation_trace_failed"))
        return row
    available = {_reference_key(candidate, scope) for candidate in candidates}
    if any(ref not in available for ref in expected):
        raise ValueError("expected evidence reference absent from saved trace")
    retrieved = [candidate.get("passage_id") for candidate in candidates]
    row["retrieved_passage_ids"] = retrieved
    selected = set(trace.get("selected_passage_ids", []))
    checks = case.get("retrieval_checks", {})
    if not isinstance(checks, dict):
        raise ValueError("retrieval checks must be an object")
    found = [ref[-1] in selected for ref in expected]
    outcomes = {}
    if checks.get("must_include_any_expected_ref"):
        outcomes["must_include_any_expected_ref"] = any(found)
    if checks.get("must_include_all_expected_refs"):
        outcomes["must_include_all_expected_refs"] = all(found)
    row["scores"]["retrieval"] = {"passed": all(outcomes.values()), "checks": outcomes,
                                   "expected_count": len(expected), "selected_expected_count": sum(found)}
    if trace.get("status") != "succeeded":
        row["status"] = "failed"
        row["errors"].append("generation_trace_failed")
    elif answer is None:
        row["errors"].append("missing_saved_answer")
    elif answer.get("trace_id") != trace.get("trace_id") or answer.get("scope") != scope:
        row["errors"].append("answer_trace_link_mismatch")
    else:
        row["status"] = "completed"
        answer_checks = case.get("deterministic_answer_checks", {})
        if not isinstance(answer_checks, dict):
            raise ValueError("answer checks must be an object")
        row["scores"]["answer"] = _answer_checks(answer, answer_checks)
    return row


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A pure reducer so a verifier can reproduce report metrics from results.jsonl."""
    splits = sorted({str(row["split"]) for row in rows})
    aggregate: dict[str, Any] = {}
    for split in splits:
        group = [row for row in rows if row["split"] == split]
        statuses = Counter(row["status"] for row in group)
        entry: dict[str, Any] = {"cases": len(group), "statuses": {
            status: statuses[status] for status in ("completed", "failed", "unscorable")}}
        for metric in ("retrieval", "answer"):
            scored = [row["scores"][metric] for row in group if row["scores"].get(metric) is not None]
            passed = sum(bool(score["passed"]) for score in scored)
            entry[metric] = {"passed": passed, "denominator": len(scored),
                             "rate": passed / len(scored) if scored else None}
        aggregate[split] = entry
    return aggregate


def _select_saved(case: Mapping[str, Any], traces: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    matches = [trace for trace in traces if trace.get("scope") == case.get("scope") and (
        trace.get("trace_id") == case.get("trace_id") if case.get("trace_id") else
        trace.get("question_id") == case.get("question_id") if case.get("question_id") else
        trace.get("question") == case.get("question")) and (
        case.get("mode") is None or trace.get("mode") == case.get("mode"))]
    if len(matches) > 1:
        raise ValueError("evaluation case matches multiple saved traces; specify trace_id or mode")
    return matches[0] if matches else None


def evaluate_saved(results: Path, cases_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = read_jsonl(cases_path)
    if not cases:
        raise ValueError("evaluation cases are missing or empty")
    source_rows = {"main": (read_jsonl(results / "traces.jsonl"),
                            read_jsonl(results / "answers.jsonl"))}
    rows: list[dict[str, Any]] = []
    rationale: dict[str, str] = {}
    used: set[str] = set()
    for case in cases:
        if case["case_id"] in used:
            raise ValueError("duplicate evaluation case ID")
        used.add(case["case_id"])
        origin = case.get("origin", "main")
        if origin not in ("main", "fixture"):
            raise ValueError("case origin must be main or fixture")
        if origin == "fixture" and origin not in source_rows:
            source_rows[origin] = (read_jsonl(results / "fixtures/traces.jsonl"),
                                   read_jsonl(results / "fixtures/answers.jsonl"))
        traces, answers = source_rows[origin]
        chosen = _select_saved(case, traces)
        linked = [answer for answer in answers if chosen is not None and
                  answer.get("trace_id") == chosen.get("trace_id")]
        if len(linked) > 1:
            raise ValueError("duplicate saved answers for trace")
        rows.append(score_case(case, chosen, linked[0] if linked else None))
        rationale[str(case["case_id"])] = str(case.get("risk_rationale", "not supplied"))
    aggregates = aggregate_results(rows)
    failures = [{"case_id": row["case_id"], "status": row["status"],
                 "errors": row["errors"], "scores": row["scores"]} for row in rows
                if row["status"] != "completed" or not all(
                    score is None or score["passed"] for score in row["scores"].values())]
    hashes = {"cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()}
    config_hashes = sorted({str(trace["config_hash"]) for traces, _ in source_rows.values()
                            for trace in traces if trace.get("config_hash")})
    command = "evaluate --results " + str(results) + " --cases " + str(cases_path)
    hashes["command_sha256"] = hashlib.sha256(command.encode("utf-8")).hexdigest()
    report = {"version": EVALUATOR_VERSION, "command": command,
              "hashes": hashes, "config_hashes": config_hashes,
              "aggregates": aggregates, "case_selection_rationale": rationale,
              "limitations": ["Candidate-authored labels are provisional and not independent ground truth.",
                              "Unmatched case questions have no saved run and are unscorable."],
              "failure_analysis": failures}
    return rows, report
