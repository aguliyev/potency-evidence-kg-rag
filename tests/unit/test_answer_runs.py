"""Contract tests for the persisted answer batch, using offline provider doubles."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from potency_kg.answering import AnswerModel
from potency_kg.contracts import Scope
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity, EmbeddingInputError
from potency_kg.retrieval import RetrievalCandidate, RetrievalResult
from potency_kg.retrieval_config import HybridRetrievalConfig


ROOT = Path(__file__).resolve().parents[2]


def _candidate(passage_id: str, via: str = "vector") -> RetrievalCandidate:
    return RetrievalCandidate(
        key=("source-1", "version-1", passage_id), document_id="source-1",
        source_id="source-1", source_version="version-1", passage_id=passage_id,
        text="A factual source passage supports this response.", text_sha256="digest",
        locator={"page": 3, "section": "A"}, selected_via=via,
        score=0.9, cosine=0.8, hop=1 if via == "graph" else None,
        via_node_id="node-1" if via == "graph" else None,
        via_edge_id="edge-1" if via == "graph" else None,
        supporting_path_ids=("edge-1",) if via == "graph" else (),
    )


class FakeEmbedding:
    identity = EmbeddingClientIdentity("fake", "text-embedding-3-large", 3072, "float", "provider_unit_length_v1", "local", "unused")

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        return EmbeddingBatch("text-embedding-3-large", (tuple([0.0] * 3072),), 3072,
                              "float", "provider_unit_length_v1", 1, 1)


class FakeGeneration:
    identity = SimpleNamespace(model="fake-generation", provider="fake", sanitized_origin="local")

    def __init__(self, *, fail_question: str | None = None, invalid_quote: bool = False) -> None:
        self.fail_question = fail_question
        self.invalid_quote = invalid_quote
        self.messages: list[str] = []
        self.last_usage = {"input_tokens": 5, "output_tokens": 3}

    def extract(self, model_text: str, response_model: type[AnswerModel], prompt: str) -> AnswerModel:
        self.messages.append(model_text)
        if self.fail_question and f"Question: {self.fail_question}\n" in model_text:
            raise RuntimeError("secret-provider-detail")
        tag = model_text.split("Evidence:\n", 1)[1].split("]", 1)[0].lstrip("[")
        quote = "Unsupported private detail" if self.invalid_quote else "A factual source passage"
        return response_model(answer="Supported.", citations=[{"tag": tag, "quote": quote}], evidence_gaps=[])


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_git_commit_uses_stable_fallback_when_git_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A hermetic runner without git must still produce a safe trace identity."""
    import potency_kg.answer_runs as runs

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git executable unavailable")

    monkeypatch.setattr(runs.subprocess, "run", missing_git)

    assert runs._git_commit(tmp_path) == "unavailable"


def test_batch_writes_contract_fields_exact_sent_evidence_and_distinct_mode_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Dropping a required field, duplicating IDs, or mixing candidates with sent passages must fail."""
    import potency_kg.answer_runs as runs

    embedding = FakeEmbedding()
    generation = FakeGeneration()
    unused = _candidate("candidate-not-sent", "graph")

    def fake_retrieve(_engine, _scope, question, mode, _client, index_version, config, query_embedding):
        assert query_embedding.model == embedding.identity.model
        candidates = (_candidate("sent"), unused) if mode == "graph" else (_candidate("sent"),)
        return RetrievalResult("retriever-v1", "query-hash", "embedding-v1", index_version,
                               "graph-v1", "graph", config, (), ("seed-1",), ("node-1",),
                               ("edge-1",), (), candidates, {})

    monkeypatch.setattr(runs, "retrieve", fake_retrieve)
    questions = [
        {"id": "P3", "question": "A", "compare_modes": True, "max_evidence_chars_per_mode": 6000},
        {"id": "P4", "question": "B", "compare_modes": False},
    ]
    descriptor = {"corpus_snapshot": "snapshot-1"}
    config = HybridRetrievalConfig(max_passages_per_document=1)
    summary = runs.run_answer_batch(object(), Scope("tenant", "project"), descriptor,
                                    questions, tmp_path / "results", embedding, generation,
                                    "index-v1", config, package_root=ROOT)

    main_answers = _rows(tmp_path / "results/answers.jsonl")
    main_traces = _rows(tmp_path / "results/traces.jsonl")
    extra_answers = _rows(tmp_path / "results/extra-mode/answers.jsonl")
    extra_traces = _rows(tmp_path / "results/extra-mode/traces.jsonl")
    contract = json.loads((ROOT / "schema/output-contract.json").read_text())
    assert summary["succeeded"] == 4
    assert len(main_answers) == len(main_traces) == 3
    assert len(extra_answers) == len(extra_traces) == 1
    assert len({row["trace_id"] for row in main_traces + extra_traces}) == 4
    assert [call[0] for call in embedding.calls] == ["A", "B"]
    for answer, trace in zip(main_answers + extra_answers, main_traces + extra_traces):
        assert set(contract["answer_required_fields"]) <= answer.keys()
        assert set(contract["trace_required_fields"]) <= trace.keys()
        assert answer["trace_id"] == trace["trace_id"]
        assert trace["status"] == "succeeded"
        assert trace["selected_passage_ids"] == ["sent"]
        assert trace["selected_via"] == [{"source_id": "source-1", "source_version": "version-1",
                                          "passage_id": "sent", "selected_via": "vector"}]
        assert trace["document_ids"] == ["source-1"]
        assert any(message == f"Question: {trace['question']}\n\nEvidence:\n{trace['evidence_text']}"
                   for message in generation.messages)
        assert trace["config_hash"] == config.config_hash()
        assert trace["config"] == config.as_dict()
        assert trace["prompt_sha256"] and trace["git_commit"]
        assert trace["model"] == "fake-generation"
        assert trace["provider"] == "fake" and trace["model_origin"] == "local"
        assert len(trace["evidence_text"]) <= (6000 if trace["question_id"] == "P3" else 16000)
        assert trace["max_evidence_chars"] == (6000 if trace["question_id"] == "P3" else 16000)
        assert all(citation["passage_id"] == "sent" for citation in answer["citations"])
    assert main_traces[0]["mode"] == "vector"
    assert main_traces[1]["mode"] == "graph"
    assert extra_traces[0]["question_id"] == "P4" and extra_traces[0]["mode"] == "vector"
    assert main_traces[1]["candidate_passage_ids"] == ["sent", "candidate-not-sent"]


def test_failed_generation_writes_safe_trace_without_answer_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Writing an invalid answer or losing the next question after a provider error must fail."""
    import potency_kg.answer_runs as runs

    def fake_retrieve(_engine, _scope, _question, _mode, _client, index_version, config, _query_embedding):
        return RetrievalResult("retriever-v1", "query-hash", "embedding-v1", index_version,
                               "graph-v1", "graph", config, (), (), (), (), (),
                               (_candidate("sent"),), {})

    monkeypatch.setattr(runs, "retrieve", fake_retrieve)
    questions = [{"id": "P1", "question": "FAIL", "compare_modes": False},
                 {"id": "P2", "question": "OK", "compare_modes": False}]
    runs.run_answer_batch(object(), Scope("tenant", "project"), {"corpus_snapshot": "snapshot-1"},
                          questions, tmp_path / "results", FakeEmbedding(),
                          FakeGeneration(fail_question="FAIL"), "index-v1",
                          HybridRetrievalConfig(), package_root=ROOT)

    traces = _rows(tmp_path / "results/traces.jsonl")
    answers = _rows(tmp_path / "results/answers.jsonl")
    assert [(row["question_id"], row["status"]) for row in traces] == [("P1", "failed"), ("P2", "succeeded")]
    assert [row["question_id"] for row in answers] == ["P2"]
    assert traces[0]["failure_kind"] == "provider"
    assert traces[0]["error_type"] == "RuntimeError"
    assert traces[0]["attempts"] and traces[0]["evidence_text"]
    assert "secret-provider-detail" not in (tmp_path / "results/traces.jsonl").read_text()


@pytest.mark.parametrize("stage,error_type", [
    ("embedding", "EmbeddingInputError"),
    ("retrieval", "ValueError"),
])
def test_internal_embedding_or_retrieval_validation_failure_is_schema_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, error_type: str,
) -> None:
    """Mislabeling a local input/binding validation error as a provider failure must fail."""
    import potency_kg.answer_runs as runs

    class InvalidEmbedding(FakeEmbedding):
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            raise EmbeddingInputError("private validation detail")

    def invalid_retrieval(*_args, **_kwargs):
        raise ValueError("private binding detail")

    if stage == "retrieval":
        monkeypatch.setattr(runs, "retrieve", invalid_retrieval)
    embedding = InvalidEmbedding() if stage == "embedding" else FakeEmbedding()
    output = tmp_path / stage
    runs.run_answer_batch(object(), Scope("tenant", "project"),
                          {"corpus_snapshot": "snapshot-1"},
                          [{"id": "P1", "question": "A", "compare_modes": False}],
                          output, embedding, FakeGeneration(), "index-v1",
                          HybridRetrievalConfig(), package_root=ROOT)

    traces = _rows(output / "traces.jsonl") + _rows(output / "extra-mode/traces.jsonl")
    assert len(traces) == 2
    assert all(trace["status"] == "failed" and trace["failure_kind"] == "schema"
               and trace["error_type"] == error_type and trace["failed_stage"] == stage
               for trace in traces)
    assert _rows(output / "answers.jsonl") == _rows(output / "extra-mode/answers.jsonl") == []
    assert "private" not in (output / "traces.jsonl").read_text()


def test_answer_command_is_gated_before_any_output_or_client_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Bypassing the model gate must fail before a provider or output path is touched."""
    import potency_kg.cli as cli

    monkeypatch.delenv(cli.MODEL_GATE, raising=False)
    monkeypatch.setattr(cli, "_embedding_client", lambda _env: pytest.fail("client constructed"))
    monkeypatch.setattr(cli, "_generation_client", lambda _env: pytest.fail("client constructed"))
    destination = tmp_path / "results"
    result = cli.main(["answer", "--config", str(ROOT / "configs/casgevy-potency.json"),
                       "--output", str(destination)])
    assert result == 1
    assert not destination.exists()


def test_invalid_citation_has_failed_trace_and_no_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Persisting a citation-rejected generation as an answer must fail."""
    import potency_kg.answer_runs as runs

    def fake_retrieve(_engine, _scope, _question, _mode, _client, index_version, config, _embedding):
        return RetrievalResult("retriever-v1", "query-hash", "embedding-v1", index_version,
                               "graph-v1", "graph", config, (), (), (), (), (),
                               (_candidate("sent"),), {})

    monkeypatch.setattr(runs, "retrieve", fake_retrieve)
    client = FakeGeneration(invalid_quote=True)
    runs.run_answer_batch(object(), Scope("tenant", "project"), {"corpus_snapshot": "snapshot-1"},
                          [{"id": "P1", "question": "A", "compare_modes": False}],
                          tmp_path / "results", FakeEmbedding(), client, "index-v1",
                          HybridRetrievalConfig(), package_root=ROOT)
    traces = _rows(tmp_path / "results/traces.jsonl")
    assert len(traces) == 1
    assert traces[0]["status"] == "failed"
    assert traces[0]["failure_kind"] == "citation_validation"
    assert traces[0]["citation_check"]["status"] == "failed"
    assert len(traces[0]["attempts"]) == 2
    assert _rows(tmp_path / "results/answers.jsonl") == []
    assert len(client.messages) == 4  # two bounded attempts in each mode
    assert _rows(tmp_path / "results/extra-mode/answers.jsonl") == []
    assert _rows(tmp_path / "results/extra-mode/traces.jsonl")[0]["failure_kind"] == "citation_validation"


def test_empty_evidence_is_successful_gap_without_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Treating an empty retrieval as a provider failure or invented answer must fail."""
    import potency_kg.answer_runs as runs

    def fake_retrieve(_engine, _scope, _question, _mode, _client, index_version, config, _embedding):
        return RetrievalResult("retriever-v1", "query-hash", "embedding-v1", index_version,
                               "graph-v1", "graph", config, (), (), (), (), (), (), {})

    monkeypatch.setattr(runs, "retrieve", fake_retrieve)
    client = FakeGeneration()
    runs.run_answer_batch(object(), Scope("tenant", "project"), {"corpus_snapshot": "snapshot-1"},
                          [{"id": "P1", "question": "A", "compare_modes": False}],
                          tmp_path / "results", FakeEmbedding(), client, "index-v1",
                          HybridRetrievalConfig(), package_root=ROOT)
    answer = _rows(tmp_path / "results/answers.jsonl")[0]
    trace = _rows(tmp_path / "results/traces.jsonl")[0]
    assert answer["citations"] == [] and answer["evidence_gaps"]
    assert trace["status"] == "succeeded" and trace["evidence_text"] == ""
    assert trace["citation_check"]["status"] == "passed"
    assert client.messages == []


def test_writer_rejects_symlink_into_protected_inputs_before_writing(
    tmp_path: Path,
) -> None:
    """Allowing output through a symlink into protected configs must fail."""
    import potency_kg.answer_runs as runs

    root = tmp_path / "package"
    (root / "configs").mkdir(parents=True)
    protected = root / "configs/input.json"
    protected.write_text("original")
    alias = root / "results"
    alias.symlink_to(root / "configs", target_is_directory=True)
    with pytest.raises(ValueError, match="protected output destination"):
        runs.run_answer_batch(object(), Scope("tenant", "project"),
                              {"corpus_snapshot": "snapshot-1"}, [], alias,
                              FakeEmbedding(), FakeGeneration(), "index-v1",
                              HybridRetrievalConfig(), package_root=root)
    assert protected.read_text() == "original"


def test_writer_preflights_all_output_files_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A protected extra-mode destination must not leave partial main output behind."""
    import potency_kg.answer_runs as runs

    root = tmp_path / "package"
    (root / "configs").mkdir(parents=True)
    output = root / "results"
    output.mkdir()
    (output / "extra-mode").symlink_to(root / "configs", target_is_directory=True)
    monkeypatch.setattr(runs, "_git_commit", lambda _root: "fixed-commit")
    with pytest.raises(ValueError, match="protected output destination"):
        runs.run_answer_batch(object(), Scope("tenant", "project"),
                              {"corpus_snapshot": "snapshot-1"}, [], output,
                              FakeEmbedding(), FakeGeneration(), "index-v1",
                              HybridRetrievalConfig(), package_root=root)
    assert not (output / "answers.jsonl").exists()
    assert not (output / "traces.jsonl").exists()


def test_prepared_retrieval_can_be_answered_without_embedding_or_database_access() -> None:
    """Forcing the fixture-stage callback to re-retrieve must fail this boundary test."""
    from potency_kg.answer_runs import answer_from_retrieval

    config = HybridRetrievalConfig()
    retrieved = RetrievalResult("retriever-v1", "query-hash", "embedding-v1", "index-v1",
                                "graph-v1", "graph", config, (), ("seed-1",), ("node-1",),
                                ("edge-1",), (), (_candidate("sent"),), {})
    answer, trace = answer_from_retrieval(
        "fixture-1", "What is supported?", "graph", retrieved,
        Scope("tenant", "project"), "snapshot-1", FakeGeneration(), 6000, config,
        git_commit="fixed-commit")

    assert answer is not None and answer["trace_id"] == trace["trace_id"]
    assert trace["selected_passage_ids"] == ["sent"]
    assert trace["index_version"] == "index-v1" and trace["graph_version"] == "graph-v1"
    assert trace["git_commit"] == "fixed-commit"


def test_post_generation_failure_preserves_generation_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after generation must retain returned telemetry and exact sent evidence."""
    import potency_kg.answer_runs as runs

    config = HybridRetrievalConfig()
    retrieved = RetrievalResult("retriever-v1", "query-hash", "embedding-v1", "index-v1",
                                "graph-v1", "graph", config, (), ("seed-1",), ("node-1",),
                                ("edge-1",), (), (_candidate("sent"),), {})
    usage = {"input_tokens": 17, "output_tokens": 4}
    attempts = ({"attempt": 1, "validated": True},)
    generated_context: list[str] = []

    def fake_answer_question(_client, _question, context, *, config):
        generated_context.append(context.evidence_text)
        return AnswerModel(answer="Supported.", citations=[], evidence_gaps=[]), usage, 23, attempts

    def fail_expansion(*_args, **_kwargs):
        raise ValueError("invalid citation record")

    monkeypatch.setattr(runs, "answer_question", fake_answer_question)
    monkeypatch.setattr(runs, "expand_citations", fail_expansion)
    answer, trace = runs.answer_from_retrieval(
        "fixture-1", "What is supported?", "graph", retrieved,
        Scope("tenant", "project"), "snapshot-1", FakeGeneration(), 6000, config,
        git_commit="fixed-commit")

    assert answer is None
    assert trace["status"] == "failed" and trace["failure_kind"] == "schema"
    assert trace["usage"] == usage
    assert trace["elapsed_ms"] == 23
    assert trace["attempts"] == list(attempts)
    assert trace["evidence_text"] == generated_context[0]


def test_record_construction_failure_keeps_trace_failed_and_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record construction error after validation must not leave a succeeded trace."""
    import potency_kg.answer_runs as runs

    class InvalidRecordAnswer:
        citations: list[object] = []
        evidence_gaps: list[object] = []

        @property
        def answer(self) -> str:
            raise ValueError("invalid answer record")

    config = HybridRetrievalConfig()
    retrieved = RetrievalResult("retriever-v1", "query-hash", "embedding-v1", "index-v1",
                                "graph-v1", "graph", config, (), ("seed-1",), ("node-1",),
                                ("edge-1",), (), (_candidate("sent"),), {})
    usage = {"input_tokens": 17, "output_tokens": 4}
    attempts = ({"attempt": 1, "validated": True},)
    generated_context: list[str] = []

    def fake_answer_question(_client, _question, context, *, config):
        generated_context.append(context.evidence_text)
        return InvalidRecordAnswer(), usage, 23, attempts

    monkeypatch.setattr(runs, "answer_question", fake_answer_question)
    monkeypatch.setattr(runs, "expand_citations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runs, "validate_citations",
                        lambda *_args, **_kwargs: {"status": "passed", "failures": []})
    answer, trace = runs.answer_from_retrieval(
        "fixture-1", "What is supported?", "graph", retrieved,
        Scope("tenant", "project"), "snapshot-1", FakeGeneration(), 6000, config,
        git_commit="fixed-commit")

    assert answer is None
    assert trace["status"] == "failed" and trace["failure_kind"] == "schema"
    assert trace["usage"] == usage
    assert trace["elapsed_ms"] == 23
    assert trace["attempts"] == list(attempts)
    assert trace["evidence_text"] == generated_context[0]


def test_answer_cli_offline_run_writes_seven_main_and_five_extra_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Changing question dispatch, mode routing, or CLI wiring must fail this run test."""
    import potency_kg.answer_runs as runs
    import potency_kg.cli as cli

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, *_args, **_kwargs):
            return self

        def scalar_one_or_none(self):
            return "index-v1"

    class FakeEngine:
        def connect(self):
            return FakeConnection()

        def dispose(self):
            pass

    def fake_retrieve(_engine, _scope, _question, _mode, _client, index_version, config, _embedding):
        return RetrievalResult("retriever-v1", "query-hash", "embedding-v1", index_version,
                               "graph-v1", "graph", config, (), (), (), (), (),
                               (_candidate("sent"),), {})

    embedding = FakeEmbedding()
    generation = FakeGeneration()
    monkeypatch.setattr(runs, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli, "_require_stage", lambda *_args: None)
    monkeypatch.setattr(cli, "_embedding_client", lambda _env: embedding)
    monkeypatch.setattr(cli, "_generation_client", lambda _env: generation)
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: FakeEngine())
    output = tmp_path / "answers"
    exit_code = cli.main(["answer", "--config", str(ROOT / "configs/casgevy-potency.json"),
                          "--output", str(output), "--default-budget-chars", "16000"])

    assert exit_code == 0
    assert len(embedding.calls) == 6
    assert len(_rows(output / "answers.jsonl")) == len(_rows(output / "traces.jsonl")) == 7
    assert len(_rows(output / "extra-mode/answers.jsonl")) == len(_rows(output / "extra-mode/traces.jsonl")) == 5
    assert [row["mode"] for row in _rows(output / "traces.jsonl") if row["question_id"] == "P3"] == ["vector", "graph"]
    assert {row["max_evidence_chars"] for row in _rows(output / "traces.jsonl") if row["question_id"] == "P3"} == {6000}
