"""The examples notebook must explain saved runs without invoking a pipeline."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "06_retrieval_and_answering_examples.ipynb"


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stage(root: Path, stage: str, version: str) -> None:
    base = root / "fixtures" / "history" / "source-correction" / stage
    base.mkdir(parents=True)
    passage = "assay-attribute-note:p0001"
    evidence = f"[S1] assay-attribute-note | {version} | {passage}\n{stage.title()} potency evidence."
    (base / "manifest.json").write_text(json.dumps({
        "scenario_id": "source-correction", "stage": stage,
        "source_versions": [{"source_id": "assay-attribute-note", "source_version": version}],
        "answer_path": "answers.jsonl", "generation_trace_path": "traces.jsonl",
        "retrieval_trace_path": "retrieval-traces.jsonl",
    }), encoding="utf-8")
    _jsonl(base / "answers.jsonl", [{
        "question_id": "F-update", "mode": "graph", "answer": f"{stage} answer",
        "citations": [{"source_id": "assay-attribute-note", "source_version": version,
                       "passage_id": passage, "quote": f"{stage.title()} potency evidence."}],
    }])
    _jsonl(base / "traces.jsonl", [{
        "question_id": "F-update", "mode": "graph", "status": "succeeded",
        "selected_passage_ids": [passage], "evidence_text": evidence,
        "selected_passages": [{"source_id": "assay-attribute-note", "source_version": version,
                               "passage_id": passage, "sent_text": f"{stage.title()} potency evidence."}],
    }])
    _jsonl(base / "retrieval-traces.jsonl", [{
        "status": "succeeded", "passages": [
            {"source_id": "assay-attribute-note", "source_version": version,
             "passage_id": passage}],
    }])


def synthetic_results(root: Path) -> Path:
    """Minimal saved artifacts with a normalized-match quote and a version replacement."""
    root.mkdir(parents=True)
    answers = []
    traces = []
    for mode, version, sent, quote in (
        ("vector", "source-v1", "Vector potency evidence.", "Vector potency evidence."),
        ("graph", "source-v2", "Graph  potency evidence.", "Graph potency evidence."),
    ):
        passage = f"p3-{mode}-p1"
        evidence = f"[S1] p3-source | {version} | {passage}\n{sent}"
        answers.append({"question_id": "P3", "mode": mode, "answer": f"{mode} answer",
                        "citations": [{"source_id": "p3-source", "source_version": version,
                                       "passage_id": passage, "quote": quote,
                                       "match": "exact" if mode == "vector" else "normalized"}]})
        traces.append({
            "question_id": "P3", "mode": mode, "status": "succeeded",
            "graph_version": "graph-v1", "index_version": "index-v1",
            "retriever_version": "retriever-v1", "config_hash": "config-hash",
            "prompt_sha256": "prompt-hash", "prompt_version": "answer-prompt-v1",
            "corpus_snapshot": "snapshot-v1", "max_evidence_chars": 6000,
            "selected_passage_ids": [passage], "selected_via": [
                {"source_id": "p3-source", "source_version": version,
                 "passage_id": passage, "selected_via": mode}],
            "selected_passages": [{"source_id": "p3-source", "source_version": version,
                                   "passage_id": passage, "selected_via": mode,
                                   "sent_text": sent}],
            "evidence_text": evidence,
        })
    _jsonl(root / "answers.jsonl", answers)
    _jsonl(root / "traces.jsonl", traces)
    _stage(root, "before", "fixture-v1")
    _stage(root, "after", "fixture-v2")
    return root


def _sources(notebook: dict[str, object]) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_is_read_only_and_uses_only_local_helpers() -> None:
    """Pipeline imports, filesystem writes, or credentials would break the viewer contract."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _sources(notebook)
    assert 'POTENCY_KG_RESULTS_ROOT' in source
    assert 'results' in source
    assert 'Missing saved artifacts' in source
    assert 'corpus/' not in source
    tree = ast.parse("\n".join("".join(cell["source"]) for cell in notebook["cells"]
                               if cell["cell_type"] == "code"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    allowed = set(sys.stdlib_module_names) | {"potency_kg"}
    for node in imports:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
        for name in names:
            assert name.split(".")[0] in allowed
            if name.startswith("potency_kg"):
                assert name == "potency_kg.notebook_present"
    forbidden = ("subprocess", "sqlalchemy", "requests", "urllib", "openai", "boto3",
                 "write_text", "write_bytes", "mkdir", "os.system", "getenv('API_KEY")
    assert all(token not in source for token in forbidden)


def _executed_text(notebook: dict[str, object]) -> str:
    chunks = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            chunks.append("".join(output.get("text", [])))
            chunks.append("".join(output.get("data", {}).get("text/plain", [])))
    return "\n".join(chunks)


def test_notebook_displays_saved_comparison_citations_and_history(tmp_path: Path) -> None:
    """A synthetic execution catches a missing mode, false quote match, or lost history."""
    root = synthetic_results(tmp_path / "results")
    saved_before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "executed.ipynb"
    env = os.environ.copy()
    env["POTENCY_KG_RESULTS_ROOT"] = str(root)
    env["PYTHONPATH"] = str(ROOT / "lib")
    subprocess.run([
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        str(NOTEBOOK), "--output", str(output),
    ], cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=120)
    rendered = _executed_text(json.loads(output.read_text(encoding="utf-8")))
    for expected in ("P3", "vector", "graph", "p3-vector-p1", "p3-graph-p1",
                     "chars_used", "6000", "config-hash", "prompt-hash", "snapshot-v1",
                     "normalized", "passed", "fixture-v1", "fixture-v2",
                     "superseded v1 passage absent after replacement: True"):
        assert expected in rendered
    assert "Graph  potency evidence." in rendered
    assert "Graph potency evidence." in rendered
    assert "normalized_match=True" in rendered
    assert saved_before == {path.relative_to(root): path.read_bytes()
                            for path in root.rglob("*") if path.is_file()}


def test_saved_citation_validation_rejects_quote_outside_sent_evidence(tmp_path: Path) -> None:
    """An original-source quote absent from saved context must not be presented as valid."""
    from potency_kg.notebook_present import validate_saved_citations

    root = synthetic_results(tmp_path / "results")
    answers = [json.loads(line) for line in (root / "answers.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (root / "traces.jsonl").read_text().splitlines()]
    answers[0]["citations"][0]["quote"] = "A passage never sent to the model."

    check, citations = validate_saved_citations(answers[0], traces[0])

    assert check["status"] == "failed"
    assert check["failures"][0]["reason"] == "quote_not_in_sent_text"
    assert citations[0]["match"] == "none"


def test_missing_artifacts_have_a_concise_message(tmp_path: Path) -> None:
    """Opening the notebook before runs exist should explain the missing inputs."""
    env = os.environ.copy()
    env["POTENCY_KG_RESULTS_ROOT"] = str(tmp_path / "empty-results")
    env["PYTHONPATH"] = str(ROOT / "lib")
    output = tmp_path / "empty.ipynb"
    subprocess.run([
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        str(NOTEBOOK), "--output", str(output),
    ], cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=120)
    rendered = _executed_text(json.loads(output.read_text(encoding="utf-8")))
    assert "Missing saved artifacts" in rendered
