"""Read-only artifact contract audit, followed by one delivery-manifest write."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evaluation import aggregate_results, read_jsonl


MANIFEST_VERSION = "delivery-manifest-v1"
SENSITIVE_PATTERN = re.compile(r"(?im)authorization\s*:\s*(?:bearer|basic)\s+\S+|raw[_ -]?provider[_ -]?error|raw[_ -]?provider[_ -]?response")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fields(row: Mapping[str, Any], required: list[str], label: str) -> None:
    missing = set(required) - row.keys()
    if missing:
        raise ValueError(f"{label} missing contract fields: {', '.join(sorted(missing))}")


def _secure_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path escapes results")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("artifact path escapes results")
    return path


def _check_records(root: Path, contract: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    import_report = json.loads((root / "import-report.json").read_text(encoding="utf-8"))
    _fields(import_report, contract["import_report_required_fields"], "import report")
    graph_rows = {}
    for name, required in (("nodes", "node_required_fields"), ("edges", "edge_required_fields")):
        graph_rows[name] = read_jsonl(root / "graph" / f"{name}.jsonl")
        for row in graph_rows[name]:
            _fields(row, contract[required], f"graph {name}")
            if name == "nodes" and row.get("type") == "Assertion":
                _fields(row, contract["evidence_bearing_node_required_fields"], "assertion")
            for source_ref in row.get("source_refs", []):
                if not isinstance(source_ref, dict):
                    raise ValueError("graph source ref must be an object")
                _fields(source_ref, contract["source_ref_required_fields"], "source ref")
                if source_ref["scope"] != row["scope"]:
                    raise ValueError("graph source ref scope mismatch")
    node_keys = {(json.dumps(row["scope"], sort_keys=True), row["id"]) for row in graph_rows["nodes"]}
    for edge in graph_rows["edges"]:
        scope_key = json.dumps(edge["scope"], sort_keys=True)
        if (scope_key, edge["source"]) not in node_keys or (scope_key, edge["target"]) not in node_keys:
            raise ValueError("graph edge endpoint missing from scoped node set")
    for origin, count in (("", 7), ("fixtures/", 3)):
        answers = read_jsonl(root / origin / "answers.jsonl")
        traces = read_jsonl(root / origin / "traces.jsonl")
        if len(answers) != count or len(traces) != count:
            raise ValueError(f"{origin or 'main'} answer/trace count must be {count}")
        answer_fields = contract["fixture_answer_required_fields" if origin else "answer_required_fields"]
        trace_fields = contract["fixture_trace_required_fields" if origin else "trace_required_fields"]
        by_id = {}
        for trace in traces:
            _fields(trace, trace_fields, "trace")
            if trace["trace_id"] in by_id:
                raise ValueError("duplicate trace ID")
            by_id[trace["trace_id"]] = trace
        seen_links = set()
        for answer in answers:
            _fields(answer, answer_fields, "answer")
            trace = by_id.get(answer["trace_id"])
            if trace is None or answer["trace_id"] in seen_links or any(
                answer.get(key) != trace.get(key) for key in
                (("question_id", "mode", "scope", "corpus_snapshot", "scenario_id", "stage", "run_id")
                 if origin else ("question_id", "mode", "scope", "corpus_snapshot"))
            ):
                raise ValueError("answer/trace link mismatch")
            seen_links.add(answer["trace_id"])
            for citation in answer.get("citations", []):
                _fields(citation, contract["citation_required_fields"], "citation")
                if citation["scope"] != answer["scope"] or citation["passage_id"] not in trace["selected_passage_ids"]:
                    raise ValueError("citation is outside saved trace")
        if origin:
            fixture_report = json.loads((root / "fixtures/report.json").read_text())
            _fields(fixture_report, contract["fixture_report_required_fields"], "fixture report")
        else:
            main_answers, main_traces = answers, traces
    results = read_jsonl(root / "evaluation/results.jsonl")
    cases = read_jsonl(root / "evaluation/cases.jsonl")
    if len(results) != len(cases) or not cases:
        raise ValueError("evaluation result count differs from frozen cases")
    for result in results:
        _fields(result, contract["evaluation_result_required_fields"], "evaluation result")
        if result["status"] not in contract["evaluation_statuses"]:
            raise ValueError("invalid evaluation status")
    case_ids = {(case["case_id"], case["case_version"], case["split"]) for case in cases}
    result_ids = {(row["case_id"], row["case_version"], row["split"]) for row in results}
    if case_ids != result_ids or len(result_ids) != len(results):
        raise ValueError("evaluation results do not match cases")
    report = json.loads((root / "evaluation/report.json").read_text())
    if report.get("aggregates") != aggregate_results(results):
        raise ValueError("evaluation aggregation mismatch")
    return main_answers, main_traces


def _artifact_paths(root: Path, contract: Mapping[str, Any]) -> list[str]:
    paths = []
    for pattern in contract["required_artifacts"]:
        if "<mmd|png|svg|pdf>" in pattern:
            choices = [pattern.replace("<mmd|png|svg|pdf>", suffix)
                       for suffix in ("mmd", "png", "svg", "pdf")]
            matching = [choice for choice in choices if _secure_path(root, choice).is_file()]
            if not matching:
                raise ValueError(f"missing required artifact: {pattern}")
            paths.append(matching[0])
        else:
            if not _secure_path(root, pattern).is_file():
                raise ValueError(f"missing required artifact: {pattern}")
            paths.append(pattern)
    return paths


def _scan_sensitive(paths: list[Path], environment: Mapping[str, str]) -> None:
    secrets = [value for key, value in environment.items() if any(
        token in key.upper() for token in ("SECRET", "TOKEN", "PASSWORD", "API_KEY"))
               and len(value) >= 8]
    for path in paths:
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SENSITIVE_PATTERN.search(body) or any(secret in body for secret in secrets):
            raise ValueError(f"sensitive content in artifact: {path.name}")


def verify_deliverables(results: Path, output: Path, contract_path: Path,
                        *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    root = results.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("manifest output escapes results")
    if destination.exists():
        raise ValueError("manifest already exists; remove or archive it before re-verification")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    relatives = _artifact_paths(root, contract)
    for relative in ("import-report.json", "graph/nodes.jsonl", "graph/edges.jsonl"):
        current = _secure_path(root, "kg/current/" + relative)
        top = _secure_path(root, relative)
        if not current.is_file() or current.read_bytes() != top.read_bytes():
            raise ValueError(f"top-level copy mismatch: {relative}")
    answers, traces = _check_records(root, contract)
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(_secure_path(root, path.relative_to(root).as_posix()))
    _scan_sensitive(files, environment if environment is not None else os.environ)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=contract_path.parent.parent,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    artifacts = [{"path": relative, "sha256": _sha(_secure_path(root, relative)),
                  "producer_status": ("real_model" if relative.startswith(("answers", "traces", "fixtures/")) else
                                      "manual_design" if relative.startswith("architecture/") or relative == "time-log.md" else
                                      "offline_test")}
                 for relative in relatives]
    fixture_report = json.loads((root / "fixtures/report.json").read_text())
    eval_report = json.loads((root / "evaluation/report.json").read_text())
    manifest = {
        "status": "verified", "manifest_version": MANIFEST_VERSION,
        "package_version": contract["version"], "contract_version": contract["version"],
        "git_commit": commit, "commands": fixture_report.get("commands", []) + [eval_report.get("command")],
        "artifacts": artifacts, "graph_versions": sorted({str(t.get("graph_version")) for t in traces if t.get("graph_version")}),
        "index_versions": sorted({str(t.get("index_version")) for t in traces if t.get("index_version")}),
        "config_hashes": sorted({str(t.get("config_hash")) for t in traces if t.get("config_hash")}),
        "prompt_versions": sorted({str(t.get("prompt_version")) for t in traces if t.get("prompt_version")}),
        "prompt_hashes": sorted({str(t.get("prompt_sha256")) for t in traces if t.get("prompt_sha256")}),
        "evaluation_cases_sha256": _sha(root / "evaluation/cases.jsonl"),
        "checks": {"required_paths": "passed", "record_counts": "passed", "contract_fields": "passed",
                   "answer_trace_links": "passed", "current_export_copies": "passed",
                   "evaluation_aggregation": "passed", "secret_scan": "passed"},
        "failures": [],
        "readme_handoff": {"status": "blocked", "reason": "root README edit requires explicit user authorization"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
