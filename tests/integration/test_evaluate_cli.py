from __future__ import annotations

import json
from pathlib import Path

import potency_kg.cli as cli
from potency_kg.evaluation import aggregate_results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_uses_saved_artifacts_and_emits_one_row_per_case(tmp_path, monkeypatch):
    root = tmp_path / "results"
    scope = {"tenant_id": "tenant", "project_id": "project"}
    ref = {"scope": scope, "source_id": "s", "source_version": "v", "passage_id": "p"}
    trace = {"trace_id": "t", "question_id": "q", "question": "Question", "mode": "graph",
             "scope": scope, "status": "succeeded", "selected_passage_ids": ["p"],
             "candidate_passages": [{k: ref[k] for k in ("source_id", "source_version", "passage_id")}]}
    answer = {"trace_id": "t", "question_id": "q", "mode": "graph", "scope": scope,
              "answer": "Yes", "evidence_gaps": [], "citations": []}
    cases = [{"case_id": "c", "case_version": "v1", "split": "development", "scope": scope,
              "question_id": "q", "question": "Question", "expected_evidence_refs": [ref],
              "retrieval_checks": {"must_include_any_expected_ref": True},
              "deterministic_answer_checks": {"must_mention_all": ["yes"]}, "risk_rationale": "grounding"},
             {"case_id": "missing", "case_version": "v1", "split": "holdout", "scope": scope,
              "question_id": "other", "question": "Other", "expected_evidence_refs": [ref],
              "retrieval_checks": {}, "deterministic_answer_checks": {}, "risk_rationale": "coverage"}]
    write_jsonl(root / "traces.jsonl", [trace])
    write_jsonl(root / "answers.jsonl", [answer])
    write_jsonl(root / "evaluation/cases.jsonl", cases)
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: 1 / 0)
    monkeypatch.setattr(cli, "_embedding_client", lambda *_: 1 / 0)
    monkeypatch.setattr(cli, "_generation_client", lambda *_: 1 / 0)
    assert cli.main(["evaluate", "--results", str(root), "--cases", str(root / "evaluation/cases.jsonl")]) == 0
    rows = [json.loads(line) for line in (root / "evaluation/results.jsonl").read_text().splitlines()]
    report = json.loads((root / "evaluation/report.json").read_text())
    assert len(rows) == 2
    assert rows[0]["status"] == "completed" and rows[1]["status"] == "unscorable"
    assert report["aggregates"] == aggregate_results(rows)
