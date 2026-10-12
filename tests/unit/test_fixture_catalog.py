from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from potency_kg.fixture_catalog import load_fixture_catalog, resolve_fixture_run


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_translates_only_declared_fixture_evidence() -> None:
    """Fixture metadata must not be admitted as retrieval evidence."""
    catalog = load_fixture_catalog(PACKAGE_ROOT)

    assert {(run.scenario_id, run.stage) for run in catalog.runs} == {
        ("isolation-project-one", "combined"),
        ("isolation-project-two", "combined"),
        ("source-correction", "before"),
        ("source-correction", "after"),
        ("unresolved-conflict", "combined"),
    }
    assert {run.scenario_id for run in catalog.runs} == {
        "isolation-project-one",
        "isolation-project-two",
        "source-correction",
        "unresolved-conflict",
    }
    assert all(path.suffix == ".jsonl" for run in catalog.runs for path in run.input_records)
    assert all(path.parent == PACKAGE_ROOT / "fixtures" / "inputs" for run in catalog.runs for path in run.input_records)


def test_catalog_resolves_config_paths_from_descriptor_and_behavior_paths_from_package_root() -> None:
    """The two supplied descriptor formats have different declared path bases."""
    catalog = load_fixture_catalog(PACKAGE_ROOT)

    isolation = resolve_fixture_run(catalog, "isolation-project-one", "combined")
    correction = resolve_fixture_run(catalog, "source-correction", "after")

    assert isolation.input_records == (PACKAGE_ROOT / "fixtures/inputs/project-one.jsonl",)
    assert correction.input_records == (PACKAGE_ROOT / "fixtures/inputs/source-correction-v2.jsonl",)
    assert correction.replacements[0].previous_source_version == "fixture-v1"
    assert correction.replacements[0].source_version == "fixture-v2"


def test_catalog_does_not_treat_questions_or_expected_answers_as_evidence() -> None:
    """Behavior-case evaluation metadata is not a source-record declaration."""
    catalog = load_fixture_catalog(PACKAGE_ROOT)

    run = resolve_fixture_run(catalog, "source-correction", "before")

    assert all("question" not in path.name and "behavior" not in path.name for path in run.input_records)
    with pytest.raises(KeyError):
        resolve_fixture_run(catalog, "source-correction", "expected_attribute")


def test_catalog_rejects_isolation_contract_scopes_without_matching_config(tmp_path: Path) -> None:
    """Isolation checks must describe the same projects as the translated config runs."""
    package_root = tmp_path / "package"
    shutil.copytree(PACKAGE_ROOT / "fixtures", package_root / "fixtures")
    descriptor_path = package_root / "fixtures" / "isolation-cases.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["assertions"][0]["query_scope"]["project_id"] = "unknown-project"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ValueError, match="isolation assertions must match config scopes"):
        load_fixture_catalog(package_root)


def test_catalog_rejects_output_path_segments_that_escape_fixture_root(tmp_path: Path) -> None:
    """Scenario and stage identifiers must remain one safe output-directory segment."""
    package_root = tmp_path / "package"
    shutil.copytree(PACKAGE_ROOT / "fixtures", package_root / "fixtures")
    descriptor_path = package_root / "fixtures" / "behavior-cases.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["scenarios"][0]["id"] = "../outside-fixture-root"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ValueError, match="fixture output identifier must be one safe path segment"):
        load_fixture_catalog(package_root)
