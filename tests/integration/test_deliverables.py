from __future__ import annotations

import json
from pathlib import Path

import pytest

from potency_kg.delivery import verify_deliverables
from potency_kg.evaluation import aggregate_results


CONTRACT = Path(__file__).resolve().parents[2] / "schema/output-contract.json"
SCOPE = {"tenant_id": "tenant", "project_id": "project"}


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def complete_tree(root: Path) -> None:
    report = {key: [] for key in json.loads(CONTRACT.read_text())["import_report_required_fields"]}
    report["scope"] = SCOPE
    report["index_version"] = "index-v1"
    for prefix in (root, root / "kg/current"):
        put(prefix / "import-report.json", report)
        jsonl(prefix / "graph/nodes.jsonl", [{"id": "n", "type": "SourceDocument", "label": "doc",
                                                "scope": SCOPE, "attributes": {}, "source_refs": []}])
        jsonl(prefix / "graph/edges.jsonl", [])
    traces, answers = [], []
    for i in range(7):
        mode = "vector" if i == 2 else "graph"
        question = "P3" if i in (2, 3) else f"P{i if i < 2 else i - 1}"
        t = {"trace_id": f"t{i}", "question_id": question, "mode": mode, "scope": SCOPE,
             "status": "succeeded", "seed_node_ids": [], "traversed_edge_ids": [],
             "selected_passage_ids": [], "evidence_text": "", "retriever_version": "r1",
             "prompt_version": "p1", "model": "model", "elapsed_ms": 1, "usage": {},
             "corpus_snapshot": "snapshot", "index_version": "index-v1", "traversal_limits": {},
             "graph_version": "graph-v1", "config_hash": "hash", "prompt_sha256": "prompt-hash"}
        traces.append(t)
        answers.append({"question_id": question, "mode": mode, "answer": "yes", "citations": [],
                        "trace_id": t["trace_id"], "scope": SCOPE, "corpus_snapshot": "snapshot"})
    jsonl(root / "traces.jsonl", traces)
    jsonl(root / "answers.jsonl", answers)
    fixture_traces, fixture_answers = [], []
    for i, stage in enumerate(("before", "after", "combined")):
        t = traces[i] | {"trace_id": f"f{i}", "question_id": "F-update" if i < 2 else "F-conflict",
                         "scenario_id": "source-correction" if i < 2 else "unresolved-conflict",
                         "stage": stage, "run_id": f"run{i}"}
        fixture_traces.append(t)
        fixture_answers.append(answers[i] | {"trace_id": t["trace_id"], "question_id": t["question_id"],
                                             "scenario_id": t["scenario_id"], "stage": stage, "run_id": t["run_id"]})
    jsonl(root / "fixtures/traces.jsonl", fixture_traces)
    jsonl(root / "fixtures/answers.jsonl", fixture_answers)
    put(root / "fixtures/report.json", {"scenarios": {}, "commands": [], "history_artifacts": {},
                                         "checks": {}, "failures": []})
    case = {"case_id": "c", "case_version": "v1", "split": "holdout", "scope": SCOPE,
            "question": "unrun", "expected_evidence_refs": [], "risk_rationale": "coverage"}
    jsonl(root / "evaluation/cases.jsonl", [case])
    result = {"case_id": "c", "case_version": "v1", "split": "holdout", "run_id": None,
              "status": "unscorable", "retrieved_passage_ids": [], "expected_evidence_refs": [],
              "scores": {"retrieval": None, "answer": None}, "errors": ["no_saved_trace_for_case"],
              "trace_id": None}
    jsonl(root / "evaluation/results.jsonl", [result])
    put(root / "evaluation/report.json", {"aggregates": aggregate_results([result])})
    for name in ("architecture.md", "architecture-diagram.mmd", "data-contracts.md"):
        path = root / "architecture" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("design\n", encoding="utf-8")
    (root / "time-log.md").write_text("recorded work\n", encoding="utf-8")


def test_complete_tree_produces_manifest_without_service_calls(tmp_path, monkeypatch):
    root = tmp_path / "results"
    complete_tree(root)
    manifest = verify_deliverables(root, root / "delivery-manifest.json", CONTRACT,
                                   environment={"MY_SECRET": "needle-value"})
    assert manifest["status"] == "verified"
    assert manifest["readme_handoff"]["status"] == "blocked"
    assert (root / "delivery-manifest.json").is_file()
    assert len(manifest["artifacts"]) == len(json.loads(CONTRACT.read_text())["required_artifacts"])


def test_incomplete_tree_fails_without_manifest(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    with pytest.raises(ValueError, match="missing required artifact"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})
    assert not (root / "delivery-manifest.json").exists()


def test_rejects_mismatched_copy_and_aggregation_and_secret(tmp_path):
    root = tmp_path / "results"
    complete_tree(root)
    (root / "graph/nodes.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="copy mismatch"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})
    complete_tree(root)
    put(root / "evaluation/report.json", {"aggregates": {}})
    with pytest.raises(ValueError, match="aggregation mismatch"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})
    complete_tree(root)
    (root / "architecture/architecture.md").write_text("Authorization: Bearer abc123\n")
    with pytest.raises(ValueError, match="sensitive content"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})
    (root / "architecture/architecture.md").write_text("needle-value\n")
    with pytest.raises(ValueError, match="sensitive content"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT,
                            environment={"MY_SECRET": "needle-value"})


def test_rejects_escaped_output_and_broken_answer_link(tmp_path):
    root = tmp_path / "results"
    complete_tree(root)
    with pytest.raises(ValueError, match="escapes results"):
        verify_deliverables(root, tmp_path / "outside.json", CONTRACT, environment={})
    answers = [json.loads(line) for line in (root / "answers.jsonl").read_text().splitlines()]
    answers[0]["trace_id"] = "absent"
    jsonl(root / "answers.jsonl", answers)
    with pytest.raises(ValueError, match="answer/trace link"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})


def test_rejects_broken_graph_reference_and_escaped_artifact(tmp_path):
    root = tmp_path / "results"
    complete_tree(root)
    node = {"id": "n", "type": "Assertion", "label": "claim", "scope": SCOPE,
            "attributes": {}, "source_refs": [{"source_id": "s"}],
            "actor": "model", "version": "v1", "created_at": "2026-10-13T00:00:00Z",
            "review_state": "unreviewed"}
    for prefix in (root, root / "kg/current"):
        jsonl(prefix / "graph/nodes.jsonl", [node])
    with pytest.raises(ValueError, match="source ref"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})
    complete_tree(root)
    (root / "architecture/architecture.md").unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (root / "architecture/architecture.md").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes results"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT, environment={})


def test_scans_saved_history_for_secret_values(tmp_path):
    root = tmp_path / "results"
    complete_tree(root)
    history = root / "fixtures/history/source-correction/before/traces.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text("needle-value\n")
    with pytest.raises(ValueError, match="sensitive content"):
        verify_deliverables(root, root / "delivery-manifest.json", CONTRACT,
                            environment={"GENERATION_API_KEY": "needle-value"})
