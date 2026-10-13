from __future__ import annotations

import json
from pathlib import Path

import pytest

import potency_kg.cli as cli
import potency_kg.workflow_stages as stages
from potency_kg.contracts import Scope


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _descriptor(tmp_path: Path) -> Path:
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}})
    )
    return config


def test_stage_wrappers_verify_protected_inputs_and_dispose_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every independently callable stage owns the same finally safety boundary."""
    config = _descriptor(tmp_path)
    before = {"configs/config.json": "before"}
    checked: list[dict[str, str]] = []
    disposed: list[bool] = []

    class BrokenEngine:
        def dispose(self) -> None:
            disposed.append(True)

    monkeypatch.setattr(stages, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(stages, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(
        stages,
        "assert_protected_inputs_unchanged",
        lambda _root, snapshot: checked.append(dict(snapshot)),
    )
    monkeypatch.setattr(stages, "create_engine_from_env", BrokenEngine)
    monkeypatch.setattr(
        stages,
        "preflight_official_inputs",
        lambda _config: (_ for _ in ()).throw(RuntimeError("stage failed")),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        stages.preflight_stage(config)

    assert checked == [before]
    assert disposed == [True]


def test_deterministic_alias_discovery_never_constructs_a_model_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Credentials and generation clients belong only to the explicitly LLM-backed stage."""
    config = _descriptor(tmp_path)
    engine = type("Engine", (), {"dispose": lambda self: None})()
    monkeypatch.setattr(stages, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(stages, "protected_input_hashes", lambda _root: {})
    monkeypatch.setattr(stages, "assert_protected_inputs_unchanged", lambda *_args: None)
    monkeypatch.setattr(stages, "create_engine_from_env", lambda: engine)
    monkeypatch.setattr(
        stages,
        "discover_aliases_deterministic",
        lambda seen_engine, scope: {"scope": scope, "method": "deterministic"},
    )
    monkeypatch.setattr(stages, "_generation_client", lambda _env: pytest.fail("generation client constructed"))
    monkeypatch.setattr(stages, "load_generation_settings", lambda _env: pytest.fail("generation settings loaded"))

    report = stages.discover_aliases_deterministic_stage(config)

    assert report["method"] == "deterministic"


def test_explicit_llm_alias_discovery_requires_gate_and_generation_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The explicit LLM command must be unable to run under an implicit/default gate."""
    config = _descriptor(tmp_path)
    monkeypatch.setattr(stages, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(stages, "protected_input_hashes", lambda _root: {})
    monkeypatch.setattr(stages, "assert_protected_inputs_unchanged", lambda *_args: None)
    monkeypatch.setattr(stages, "create_engine_from_env", lambda: pytest.fail("engine constructed"))

    with pytest.raises(stages.RealModelsNotEnabled):
        stages.discover_aliases_llm_stage(config, enabled=True, env={})


def test_composite_stays_structural_when_real_model_settings_are_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An enabled-but-incomplete gate must retain the established structural fallback."""
    config = _descriptor(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(stages, "preflight_stage", lambda _config: {})
    monkeypatch.setattr(stages, "import_stage", lambda _config: {})
    monkeypatch.setattr(stages, "load_alias_seeds_stage", lambda _config: {})
    monkeypatch.setattr(stages, "discover_aliases_deterministic_stage", lambda _config: {})
    monkeypatch.setattr(stages, "discover_aliases_llm_stage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(stages, "load_embedding_settings", lambda _env: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(stages, "embed_stage", lambda *_args, **_kwargs: pytest.fail("embedding started"))
    monkeypatch.setattr(stages, "refresh_graph_stage", lambda _config, *, mode: events.append(mode) or {})
    monkeypatch.setattr(stages, "export_stage", lambda *_args: {})
    monkeypatch.setattr(stages, "inspect_stage", lambda _config, **_kwargs: {})

    report = stages.run_kg_workflow(config, tmp_path / "results", env={stages.MODEL_GATE: "true"})

    assert report["mode"] == "structural"
    assert events == ["structural"]


def test_cli_alias_review_parser_requires_complete_approval_target() -> None:
    """Review is explicit and an approval cannot invent an incomplete canonical target."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["review-alias", "--config", "config.json"])
    incomplete = parser.parse_args(
        [
            "review-alias",
            "--config",
            "config.json",
            "--alias-id",
            "alias-1",
            "--decision",
            "approved",
            "--reviewer",
            "operator",
        ]
    )
    assert incomplete.canonical_key is None
    parsed = parser.parse_args(
        [
            "review-alias",
            "--config",
            "config.json",
            "--alias-id",
            "alias-1",
            "--decision",
            "approved",
            "--reviewer",
            "operator",
            "--entity-type",
            "Product",
            "--canonical-key",
            "casgevy",
            "--canonical-name",
            "CASGEVY",
        ]
    )
    assert parsed.decision == "approved"


def test_stage_reports_are_json_serializable_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Public stage results must be safe for CLI and notebook display."""
    config = _descriptor(tmp_path)
    engine = type("Engine", (), {"dispose": lambda self: None})()
    monkeypatch.setattr(stages, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(stages, "protected_input_hashes", lambda _root: {})
    monkeypatch.setattr(stages, "assert_protected_inputs_unchanged", lambda *_args: None)
    monkeypatch.setattr(stages, "create_engine_from_env", lambda: engine)
    monkeypatch.setattr(stages, "load_alias_seeds", lambda *_args: {"status": "loaded"})

    report = stages.load_alias_seeds_stage(config)

    assert json.loads(json.dumps(cli._jsonable(report))) == {"status": "loaded"}


def test_cli_materialize_uses_the_public_protected_stage_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI and notebook must share the same materialization boundary."""
    config = _descriptor(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["materialize", "--config", str(config)])
    expected = {"materialized": 0}
    monkeypatch.setattr(cli, "materialize_stage", lambda actual: expected if actual == config else None)

    assert cli._execute(args, {}) == expected


def test_cli_alias_evidence_uses_the_read_only_scoped_stage_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Evidence inspection must use the same protected scope boundary as alias review."""
    config = _descriptor(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args([
        "list-alias-evidence", "--config", str(config), "--source-type", "deterministic_rule",
        "--entity-type", "Product", "--canonical-key", "casgevy", "--alias-id", "alias-1",
    ])
    expected = {"immutable": True}
    monkeypatch.setattr(
        cli, "inspect_alias_evidence_stage",
        lambda actual, **filters: expected if actual == config and filters == {
            "source_type": "deterministic_rule", "entity_type": "Product", "canonical_key": "casgevy",
            "alias_id": "alias-1",
        } else None,
    )

    assert cli._execute(args, {}) == expected


def test_scoped_inspection_reports_all_stage_counts_statuses_failures_and_versions(
    db_engine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A post-write inspection must expose every independently rerunnable stage's observable state."""
    scope = Scope("inspection-tenant", "inspection-project")
    monkeypatch.setattr(
        stages, "_guarded_stage",
        lambda config, operation: operation(config, scope, db_engine),
    )

    report = stages.inspect_stage(tmp_path / "config.json", output_root=tmp_path / "results")

    assert set(report["counts"]) == {
        "documents", "passages", "aliases", "alias_evidence", "alias_discovery_runs",
        "embedding_runs", "indexes", "jev_runs", "extraction_runs", "materializations",
        "graph_refreshes", "exports",
    }
    assert set(report["statuses"]) == {
        "aliases", "processing", "alias_discovery", "embedding", "jev", "extraction",
        "materialization", "graph_refresh",
        "export",
    }
    assert set(report["sanitized_failures"]) == {
        "alias_discovery", "embedding", "jev", "extraction", "materialization", "graph_refresh",
        "export",
    }
    assert report["versions"] == {
        "active_index": None, "active_embedding": None, "active_graph": None,
        "graph_index": None, "graph_embedding": None, "active_export": None,
    }


def test_build_notebook_has_ordered_independent_stage_and_inspection_cells() -> None:
    """The operator notebook must expose stages rather than hide them in one composite call."""
    notebook = json.loads(
        (PACKAGE_ROOT / "notebooks/01_build_knowledge_graph.ipynb").read_text()
    )
    cells = [
        (cell["cell_type"], "".join(cell.get("source", [])))
        for cell in notebook["cells"]
    ]
    stages_in_order = (
        "preflight",
        "import",
        "load_alias_seeds",
        "discover_aliases_deterministic",
        "discover_aliases_llm",
        "inspect_aliases",
        "embed",
        "classify_jev",
        "extract",
        "materialize",
        "refresh_graph",
        "export",
        "inspect",
    )
    call_positions: list[int] = []
    for stage_name in stages_in_order:
        positions = [
            index for index, (kind, source) in enumerate(cells)
            if kind == "code" and f"{stage_name}_stage(" in source
        ]
        position = positions[-1] if stage_name == "inspect" else positions[0]
        call_positions.append(position)
    assert call_positions == sorted(call_positions)

    write_stages = stages_in_order[:-1]
    for stage_name in write_stages:
        call_position = next(
            position
            for position, (kind, source) in enumerate(cells)
            if kind == "code" and f"{stage_name}_stage(" in source
        )
        explanation = cells[call_position - 1]
        inspection = cells[call_position + 1]
        assert explanation[0] == "markdown"
        assert all(word in explanation[1].casefold() for word in ("inputs", "outputs", "database", "rerun", "invalidation"))
        assert inspection[0] == "code" and "inspect" in inspection[1].casefold()

    build_code = "\n".join(source for kind, source in cells if kind == "code")
    assert "DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY" in build_code
    assert "enabled=enable_llm_alias_discovery" in build_code
    assert "alias_id=" in build_code
    assert "output_root=output_root" in build_code
    assert "def " not in build_code and "class " not in build_code
    assert "API_KEY" not in build_code and "Authorization" not in build_code
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
