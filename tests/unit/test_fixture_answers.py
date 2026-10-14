"""Offline contracts for fixture answer history and stage ordering."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from potency_kg.contracts import Scope
from potency_kg.fixture_catalog import FixtureRunConfiguration
from potency_kg.fixture_runs import ModelClients, run_kg_fixtures
from potency_kg.retrieval import RetrievalResult
from potency_kg.retrieval_config import HybridRetrievalConfig


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixture_usage_aggregates_provider_token_field_names() -> None:
    from potency_kg.fixture_answers import _summed_generation_usage

    traces = [
        {"usage": {"prompt_tokens": 653, "completion_tokens": 1514, "total_tokens": 2167}},
        {"usage": {"prompt_tokens": 653, "completion_tokens": 1821, "total_tokens": 2474}},
    ]

    assert _summed_generation_usage(traces) == {
        "prompt_tokens": 1306, "completion_tokens": 3335, "total_tokens": 4641,
    }


def _stage(tmp_path: Path, scenario: str, stage: str, question: str | None):
    from potency_kg.fixture_answers import FixtureAnswerRecorder

    root = tmp_path / "results" / "fixtures"
    recorder = FixtureAnswerRecorder(root, package_root=tmp_path)
    pipeline = tmp_path / "pipeline" / scenario / stage
    version = pipeline / "kg" / "history" / "graph-v1"
    (pipeline / "corpus").mkdir(parents=True)
    (version / "graph").mkdir(parents=True)
    (pipeline / "corpus" / "records.jsonl").write_text(
        json.dumps({"source_id": "source", "source_version": "fixture-v1",
                    "passage_id": "p1", "text": "Exact fixture evidence."}) + "\n",
        encoding="utf-8",
    )
    (version / "import-report.json").write_text("{}\n", encoding="utf-8")
    (version / "graph" / "nodes.jsonl").write_text("", encoding="utf-8")
    (version / "graph" / "edges.jsonl").write_text("", encoding="utf-8")
    configuration = FixtureRunConfiguration(scenario, stage, Scope("tenant", scenario),
                                            "snapshot-v1", (), (), "import", question)
    retrieval = RetrievalResult(
        "retriever", "query", "embedding", "index-v1", "graph-v1", "graph",
        HybridRetrievalConfig(), (), (), (), (), (), (), {},
    )
    report = SimpleNamespace(configuration=configuration, output_dir=pipeline,
                             retrieval=retrieval, export=SimpleNamespace(version_dir=version),
                             embeddings=SimpleNamespace(index_version="index-v1"))
    return recorder, configuration, report, root


def test_recorder_keeps_failed_generation_trace_and_separate_retrieval_only_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed answer cannot erase its trace or look like isolation's no-generation stage."""
    import potency_kg.fixture_answers as answers

    recorder, configuration, report, root = _stage(tmp_path, "source-correction", "before", "Q?")
    monkeypatch.setattr(answers, "answer_from_retrieval", lambda *args, **kwargs: (
        None, {"trace_id": "trace-1", "status": "failed", "failure_kind": "provider",
               "error_type": "RuntimeError", "usage": {"input_tokens": 2}},
    ))
    recorder.record_stage(configuration, report, object(), {"source-correction": "F-update"})

    history = root / "history" / "source-correction" / "before"
    manifest = json.loads((history / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_performed"] is True
    assert manifest["answer_count"] == 0
    assert manifest["retrieval_trace_path"] == "retrieval-traces.jsonl"
    assert manifest["answer_path"] == "answers.jsonl"
    assert manifest["generation_trace_path"] == "traces.jsonl"
    assert _rows(history / "answers.jsonl") == []
    assert _rows(history / "traces.jsonl")[0]["failure_kind"] == "provider"
    assert len(_rows(history / "retrieval-traces.jsonl")) == 1
    assert (history / "source-records.jsonl").exists()
    assert (history / "graph" / "nodes.jsonl").exists()

    isolation, configuration2, report2, _ = _stage(tmp_path, "isolation-project-one", "combined", None)
    isolation.record_stage(configuration2, report2, object(), {})
    manifest2 = json.loads((root / "history" / "isolation-project-one" / "combined" / "manifest.json").read_text())
    assert manifest2["generation_performed"] is False
    assert manifest2["answer_count"] == 0
    assert _rows(root / "history" / "isolation-project-one" / "combined" / "traces.jsonl") == []


def test_real_fixture_mode_requires_explicit_gate_before_database_access(tmp_path: Path) -> None:
    """Passing allow_real without the gate cannot reach a model or the database."""
    identity = SimpleNamespace(provider="external")
    adapter = SimpleNamespace(identity=identity)
    clients = ModelClients(adapter, adapter, adapter)
    catalog = SimpleNamespace(package_root=tmp_path, runs=())
    with pytest.raises(ValueError, match="KG_RUN_REAL_MODELS"):
        run_kg_fixtures(object(), catalog, tmp_path / "output", clients,
                        allow_real=True, env={})


def test_successful_stage_reuses_completed_retrieval_and_labels_both_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second retrieval or missing stage identity would make the saved answer unauditable."""
    import potency_kg.fixture_answers as answers

    recorder, configuration, report, root = _stage(tmp_path, "unresolved-conflict", "combined", "Q?")
    seen = []

    def generate(question_id, question, mode, retrieved, scope, snapshot, client, cap, config,
                 *, git_commit):
        seen.append(retrieved)
        return ({"question_id": question_id, "mode": mode, "citations": []},
                {"trace_id": "trace-2", "status": "succeeded", "usage": {}})

    monkeypatch.setattr(answers, "answer_from_retrieval", generate)
    recorder.record_stage(configuration, report, object(), {"unresolved-conflict": "F-conflict"})

    answer = _rows(root / "history/unresolved-conflict/combined/answers.jsonl")[0]
    trace = _rows(root / "history/unresolved-conflict/combined/traces.jsonl")[0]
    assert seen == [report.retrieval]
    assert (answer["question_id"], answer["mode"]) == ("F-conflict", "graph")
    assert (answer["scenario_id"], answer["stage"]) == ("unresolved-conflict", "combined")
    assert answer["run_id"] == trace["run_id"]
    assert len(_rows(root / "history/unresolved-conflict/combined/retrieval-traces.jsonl")) == 1


def test_cli_exposes_fixture_answers_with_default_result_location() -> None:
    from potency_kg.cli import build_parser

    args = build_parser().parse_args(["run-fixture-answers"])

    assert args.output == Path("results/fixtures")


def test_history_preflight_rejects_redirected_stage_before_writing(tmp_path: Path) -> None:
    """A symlinked history stage must not receive source or generation artifacts."""
    from potency_kg.fixture_answers import FixtureAnswerRecorder

    root = tmp_path / "results" / "fixtures"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "history" / "source-correction").mkdir(parents=True)
    (root / "history" / "source-correction" / "before").symlink_to(
        outside, target_is_directory=True)
    recorder = FixtureAnswerRecorder(root, package_root=tmp_path)
    run = FixtureRunConfiguration("source-correction", "before", Scope("t", "p"),
                                  "snap", (), (), "import", "Q?")

    with pytest.raises(ValueError, match="fixture answer output"):
        recorder.preflight((run,))

    assert list(outside.iterdir()) == []


def test_preflight_preserves_an_existing_answer_batch(tmp_path: Path) -> None:
    """A rerun cannot overwrite a prior fixture answer batch before importing sources."""
    from potency_kg.fixture_answers import FixtureAnswerRecorder

    root = tmp_path / "results" / "fixtures"
    root.mkdir(parents=True)
    (root / "answers.jsonl").write_text("prior run\n", encoding="utf-8")
    recorder = FixtureAnswerRecorder(root, package_root=tmp_path)

    with pytest.raises(FileExistsError, match="answers.jsonl"):
        recorder.preflight(())

    assert (root / "answers.jsonl").read_text(encoding="utf-8") == "prior run\n"


@pytest.mark.parametrize(
    ("passages", "expected"),
    [
        ([], False),
        ([{"source_id": "assay-attribute-note", "source_version": "fixture-v1",
           "passage_id": "assay-attribute-note:p0001"}], False),
        ([{"source_id": "assay-attribute-note", "source_version": "fixture-v2",
           "passage_id": "assay-attribute-note:p0001"}], True),
        ([{"source_id": "assay-attribute-note", "source_version": "fixture-v1",
           "passage_id": "assay-attribute-note:p0001"},
          {"source_id": "assay-attribute-note", "source_version": "fixture-v2",
           "passage_id": "assay-attribute-note:p0001"}], False),
    ],
)
def test_superseded_check_uses_saved_retrieval_and_requires_current_evidence(
    tmp_path: Path, passages: list[dict[str, str]], expected: bool,
) -> None:
    """An empty failed-generation trace cannot create a vacuous v1-absence pass."""
    recorder, _, _, root = _stage(tmp_path, "source-correction", "after", "Q?")
    saved = root / "history" / "source-correction" / "after" / "retrieval-traces.jsonl"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"status": "succeeded", "passages": passages}) + "\n",
                     encoding="utf-8")

    assert recorder.check_superseded_after_retrieval() is expected
