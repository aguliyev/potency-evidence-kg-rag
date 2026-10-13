from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import potency_kg.cli as cli
import potency_kg.fixture_runs as fixture_runs
import potency_kg.workflow_stages as workflow_stages
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PACKAGE_ROOT / "configs" / "casgevy-potency.json"


def test_parser_exposes_only_derived_version_commands() -> None:
    """Adding caller-controlled version labels or omitting a required stage must break this test."""
    parser = cli.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert commands == {
        "hash-inputs",
        "preflight",
        "import",
        "embed",
        "classify-jev",
        "extract",
        "materialize",
        "refresh-graph",
        "export-kg",
        "inspect-kg",
        "run-kg",
        "run-kg-fixtures",
        "load-alias-seeds",
        "discover-aliases-deterministic",
        "discover-aliases-llm",
        "list-aliases",
        "list-alias-evidence",
        "review-alias",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["refresh-graph", "--config", str(CONFIG), "--graph-version", "caller-label"])
    with pytest.raises(SystemExit):
        parser.parse_args(["embed", "--config", str(CONFIG), "--embedding-version", "caller-label"])


def test_structural_workflow_orders_stages_and_never_constructs_model_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constructing any model adapter during the default structural workflow must break this test."""
    events: list[str] = []
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))

    monkeypatch.setattr(workflow_stages, "preflight_stage", lambda _path: events.append("preflight") or {"document_count": 1})
    monkeypatch.setattr(workflow_stages, "import_stage", lambda _path: events.append("import") or {"source_count": 1})
    monkeypatch.setattr(workflow_stages, "load_alias_seeds_stage", lambda _path: events.append("load-alias-seeds") or {})
    monkeypatch.setattr(workflow_stages, "discover_aliases_deterministic_stage", lambda _path: events.append("discover-aliases-deterministic") or {})
    monkeypatch.setattr(workflow_stages, "discover_aliases_llm_stage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow_stages, "refresh_graph_stage", lambda _path, *, mode: events.append(f"refresh:{mode}") or {"graph_version": "graph-" + "1" * 64})
    monkeypatch.setattr(workflow_stages, "export_stage", lambda *_args: events.append("export") or {"graph_version": "graph-" + "1" * 64})
    monkeypatch.setattr(workflow_stages, "inspect_stage", lambda _path, **_kwargs: events.append("inspect") or {"counts": {}})

    report = workflow_stages.run_kg_workflow(config, tmp_path / "results", env={})

    assert events == [
        "preflight",
        "import",
        "load-alias-seeds",
        "discover-aliases-deterministic",
        "refresh:structural",
        "export",
        "inspect",
    ]
    assert report["mode"] == "structural"
    assert report["semantic_status"] == "not_run"
    assert report["graph"]["graph_version"] == "graph-" + "1" * 64


def test_readiness_does_not_load_stage_settings_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading any credential-backed settings while the real-model gate is off must break this test."""
    monkeypatch.setattr(cli, "load_embedding_settings", lambda _env: pytest.fail("embedding settings loaded"))
    monkeypatch.setattr(cli, "load_generation_settings", lambda _env: pytest.fail("generation settings loaded"))
    monkeypatch.setattr(cli, "load_jev_settings", lambda _env: pytest.fail("Jev settings loaded"))

    readiness = cli.real_model_readiness({cli.MODEL_GATE: "false"})

    assert readiness == {
        "enabled": False,
        "embedding_present": False,
        "generation_present": False,
        "jev_present": False,
        "ready": False,
    }


def test_workflow_verifies_protected_inputs_even_when_a_stage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dropping the finally-protected post-run hash check must break this test."""
    checked: list[dict[str, str]] = []
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    before = {"configs/config.json": "before"}
    monkeypatch.setattr(cli, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(cli, "assert_protected_inputs_unchanged", lambda _root, value: checked.append(dict(value)))
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: object())
    monkeypatch.setattr(cli, "preflight_official_inputs", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        cli.run_kg_workflow(config, tmp_path / "results", env={})

    assert checked == [before]


def test_main_returns_nonzero_and_sanitizes_stage_failures(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Leaking an exception message or returning success on stage failure must break this test."""
    secret = "never-print-this-secret"
    monkeypatch.setattr(cli, "run_kg_workflow", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)))

    status = cli.main(["run-kg", "--config", str(CONFIG), "--output", "results/kg"])

    captured = capsys.readouterr()
    assert status != 0
    assert secret not in captured.out + captured.err
    payload = json.loads(captured.err)
    assert payload["status"] == "failed" and payload["error_type"] == "RuntimeError"


def test_hash_inputs_round_trip_detects_only_path_or_digest_changes(tmp_path: Path) -> None:
    """Accepting a changed protected snapshot must break this test."""
    root = tmp_path / "package"
    for directory in ("configs", "corpus", "fixtures", "schema"):
        (root / directory).mkdir(parents=True)
        (root / directory / "input.txt").write_text(directory)
    snapshot = tmp_path / "snapshot.json"

    written = cli.hash_inputs(root, output=snapshot)
    verified = cli.hash_inputs(root, verify=snapshot)

    assert written["status"] == verified["status"] == "verified"
    assert written["file_count"] == 4
    (root / "corpus" / "input.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="protected input snapshot differs"):
        cli.hash_inputs(root, verify=snapshot)


@pytest.mark.parametrize("protected_name", ["configs", "corpus", "fixtures", "schema"])
@pytest.mark.parametrize("through_symlink", [False, True])
def test_hash_output_guard_rejects_every_protected_root_before_write(
    tmp_path: Path, protected_name: str, through_symlink: bool
) -> None:
    """Direct and symlink-aliased destinations must be rejected before protected bytes change."""
    root = tmp_path / "package"
    for directory in ("configs", "corpus", "fixtures", "schema"):
        (root / directory).mkdir(parents=True)
        (root / directory / "input.txt").write_text(directory)
    protected = root / protected_name
    before = (protected / "input.txt").read_bytes()
    if through_symlink:
        alias = tmp_path / f"{protected_name}-alias"
        alias.symlink_to(protected, target_is_directory=True)
        destination = alias / "snapshot.json"
    else:
        destination = protected / "snapshot.json"

    with pytest.raises(ValueError, match="protected output"):
        cli.hash_inputs(root, output=destination)

    assert (protected / "input.txt").read_bytes() == before
    assert not (protected / "snapshot.json").exists()


def test_run_output_guard_fails_before_any_stage_or_engine_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A protected run root must be rejected before database or filesystem orchestration starts."""
    root = tmp_path / "package"
    config = root / "configs" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    for directory in ("corpus", "fixtures", "schema"):
        (root / directory).mkdir()
    monkeypatch.setattr(cli, "_package_root", lambda: root)
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: pytest.fail("engine constructed"))

    with pytest.raises(ValueError, match="protected output"):
        cli.run_kg_workflow(config, root / "schema" / "generated", env={})


def test_inspection_trace_guard_fails_before_engine_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A protected trace destination must be rejected before retrieval can make any request."""
    root = tmp_path / "package"
    config = root / "configs" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    for directory in ("corpus", "fixtures", "schema"):
        (root / directory).mkdir()
    monkeypatch.setattr(cli, "_package_root", lambda: root)
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: pytest.fail("engine constructed"))

    with pytest.raises(ValueError, match="protected output"):
        cli.inspect_kg_workflow(
            config,
            env={cli.SMOKE_GATE: "true"},
            question="question",
            trace_output=root / "fixtures" / "trace.json",
        )


def test_fixture_output_root_is_exact_and_confined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Writing fixture artifacts outside results/kg/fixtures must break this test."""
    seen: list[Path] = []
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: object())
    monkeypatch.setattr(cli, "load_fixture_catalog", lambda root: type("Catalog", (), {"package_root": root})())
    monkeypatch.setattr(cli, "fixture_model_clients", lambda: object())
    monkeypatch.setattr(
        cli,
        "run_kg_fixtures",
        lambda _engine, _catalog, output, _clients: seen.append(output) or type("Report", (), {"runs": (), "reconnect_verified": True})(),
    )

    output = tmp_path / "results" / "kg" / "fixtures"
    cli.run_fixture_workflow(tmp_path, output)

    assert seen == [output.resolve()]
    with pytest.raises(ValueError, match="fixture output must be exactly"):
        cli.run_fixture_workflow(tmp_path, tmp_path / "nested" / "results" / "kg" / "fixtures")


def test_fixture_output_root_rejects_a_symlink_that_escapes_the_repository(tmp_path: Path) -> None:
    """Resolving both sides through an escaping fixture symlink must not satisfy root confinement."""
    output = tmp_path / "results" / "kg" / "fixtures"
    output.parent.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="fixture output must be exactly"):
        cli.run_fixture_workflow(tmp_path, output)


def test_fixture_workflow_verifies_protected_inputs_when_a_stage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Moving the fixture post-hash check off the failure path must break this test."""
    before = {"fixtures/example.jsonl": "before"}
    checked: list[dict[str, str]] = []
    monkeypatch.setattr(cli, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(cli, "assert_protected_inputs_unchanged", lambda _root, value: checked.append(dict(value)))
    monkeypatch.setattr(cli, "load_fixture_catalog", lambda root: type("Catalog", (), {"package_root": root})())
    monkeypatch.setattr(cli, "create_engine_from_env", lambda: object())
    monkeypatch.setattr(cli, "fixture_model_clients", lambda: object())
    monkeypatch.setattr(
        cli,
        "run_kg_fixtures",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fixture stage failed")),
    )

    with pytest.raises(RuntimeError, match="fixture stage failed"):
        cli.run_fixture_workflow(tmp_path, tmp_path / "results" / "kg" / "fixtures")

    assert checked == [before]


def test_fixture_library_verifies_protected_inputs_when_a_stage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keeping the fixture library's hash check only on its success path must break this test."""
    before = {"fixtures/example.jsonl": "before"}
    checked: list[dict[str, str]] = []
    output = tmp_path / "results" / "kg" / "fixtures"
    existing_stage = output / "scenario" / "stage"
    existing_stage.mkdir(parents=True)
    configuration = type("Configuration", (), {"scenario_id": "scenario", "stage": "stage"})()
    catalog = type("Catalog", (), {"package_root": tmp_path, "runs": (configuration,)})()
    monkeypatch.setattr(fixture_runs, "_require_offline_test_doubles", lambda _clients: None)
    monkeypatch.setattr(fixture_runs, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(
        fixture_runs,
        "assert_protected_inputs_unchanged",
        lambda _root, value: checked.append(dict(value)),
    )

    with pytest.raises(FileExistsError):
        fixture_runs.run_kg_fixtures(object(), catalog, output, object())

    assert checked == [before]


def test_inspection_verifies_protected_inputs_even_when_database_access_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removing the inspection post-hash check must expose this protected-input regression."""
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    before = {"configs/config.json": "before"}
    checked: list[dict[str, str]] = []

    class BrokenEngine:
        def connect(self) -> object:
            raise RuntimeError("database unavailable")

        def dispose(self) -> None:
            pass

    monkeypatch.setattr(cli, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(cli, "assert_protected_inputs_unchanged", lambda _root, value: checked.append(dict(value)))
    monkeypatch.setattr(cli, "create_engine_from_env", BrokenEngine)

    with pytest.raises(RuntimeError, match="database unavailable"):
        cli.inspect_kg_workflow(config, env={})

    assert checked == [before]


def test_inspection_verifies_protected_inputs_when_engine_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Creating the engine outside inspection's finally boundary must break this test."""
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    before = {"configs/config.json": "before"}
    checked: list[dict[str, str]] = []
    monkeypatch.setattr(cli, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(cli, "assert_protected_inputs_unchanged", lambda _root, value: checked.append(dict(value)))
    monkeypatch.setattr(
        cli,
        "create_engine_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("engine creation failed")),
    )

    with pytest.raises(RuntimeError, match="engine creation failed"):
        cli.inspect_kg_workflow(config, env={})

    assert checked == [before]


def test_individual_stage_verifies_protected_inputs_when_engine_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Creating the engine outside an individual stage's finally boundary must break this test."""
    config = tmp_path / "configs" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"scope": {"tenant_id": "tenant", "project_id": "project"}}))
    before = {"configs/config.json": "before"}
    checked: list[dict[str, str]] = []
    monkeypatch.setattr(cli, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "protected_input_hashes", lambda _root: before)
    monkeypatch.setattr(cli, "assert_protected_inputs_unchanged", lambda _root, value: checked.append(dict(value)))
    monkeypatch.setattr(
        cli,
        "create_engine_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("engine creation failed")),
    )

    with pytest.raises(RuntimeError, match="engine creation failed"):
        cli._guarded_config_call(config, lambda *_args: pytest.fail("stage must not run"))

    assert checked == [before]


def test_notebooks_are_declarative_sanitized_and_smoke_is_explicitly_gated() -> None:
    """Putting implementations, credentials, or an ungated write in a notebook must break this test."""
    build = json.loads((PACKAGE_ROOT / "notebooks/01_build_knowledge_graph.ipynb").read_text())
    inspect = json.loads((PACKAGE_ROOT / "notebooks/02_inspect_knowledge_graph.ipynb").read_text())
    build_code = "\n".join("".join(cell.get("source", [])) for cell in build["cells"] if cell["cell_type"] == "code")
    inspect_code = "\n".join("".join(cell.get("source", [])) for cell in inspect["cells"] if cell["cell_type"] == "code")
    all_code = build_code + "\n" + inspect_code

    assert "from potency_kg.workflow_stages" in build_code
    assert "run_kg_workflow" not in build_code
    assert "from potency_kg" in inspect_code and "inspect_kg_workflow" in inspect_code
    assert "KG_ENABLE_RETRIEVAL_SMOKE" in inspect_code
    assert "if enable_retrieval_smoke" in inspect_code
    assert "def " not in all_code and "class " not in all_code
    assert "API_KEY" not in all_code and "Authorization" not in all_code
    assert all(not cell.get("outputs") for notebook in (build, inspect) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_alias_review_notebook_has_conservative_high_confidence_bulk_approval() -> None:
    """Bulk approval must remain preview-first and refuse incomplete targets."""
    review = json.loads((PACKAGE_ROOT / "notebooks/03_review_aliases.ipynb").read_text())
    review_code = "\n".join("".join(cell.get("source", [])) for cell in review["cells"] if cell["cell_type"] == "code")

    assert "def approve_high_confidence_aliases" in review_code
    assert "r.confidence > threshold" in review_code
    assert "dry_run=True" in review_code
    assert "missing target fields" in review_code
    assert "review_alias(r.alias_id, \"approved\"" in review_code


def test_runner_remains_executable() -> None:
    """Removing executable permissions from the supported operator boundary must break this test."""
    assert stat.S_IMODE((PACKAGE_ROOT / "bin/runner").stat().st_mode) & 0o111
