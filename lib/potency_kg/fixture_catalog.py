"""Read-only translation of the supplied synthetic-fixture descriptors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Scope
from .project_profiles import protected_input_hashes


@dataclass(frozen=True)
class FixtureReplacement:
    """An explicit source-version activation declaration from a behavior stage."""

    source_id: str
    previous_source_version: str
    source_version: str


@dataclass(frozen=True)
class FixtureRunConfiguration:
    scenario_id: str
    stage: str
    scope: Scope
    corpus_snapshot: str
    input_records: tuple[Path, ...]
    replacements: tuple[FixtureReplacement, ...]
    operation: str
    question: str | None


@dataclass(frozen=True)
class FixtureCatalog:
    package_root: Path
    runs: tuple[FixtureRunConfiguration, ...]
    protected_hashes: dict[str, str]


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid fixture descriptor: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"fixture descriptor must be an object: {path}")
    return value


def _scope(value: object) -> Scope:
    if not isinstance(value, dict):
        raise ValueError("fixture scope is required")
    tenant_id, project_id = value.get("tenant_id"), value.get("project_id")
    if not isinstance(tenant_id, str) or not isinstance(project_id, str):
        raise ValueError("fixture scope must have tenant_id and project_id")
    return Scope(tenant_id, project_id)


def _output_segment(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError("fixture output identifier must be one safe path segment")
    return value


def _fixture_input_path(
    package_root: Path,
    descriptor_path: Path,
    value: object,
    *,
    config_relative: bool,
) -> Path:
    """Convert either declared path convention into one allowlisted evidence path."""
    if not isinstance(value, str) or not value:
        raise ValueError("fixture evidence path must be a nonempty string")
    base = descriptor_path.parent if config_relative else package_root
    candidate = (base / value).resolve()
    inputs = (package_root / "fixtures" / "inputs").resolve()
    try:
        candidate.relative_to(inputs)
    except ValueError as error:
        raise ValueError(f"fixture evidence path escapes named JSONL inputs: {value}") from error
    if candidate.parent != inputs or candidate.suffix != ".jsonl" or not candidate.is_file():
        raise ValueError(f"fixture evidence must be a named JSONL input: {value}")
    return candidate


def _records(path: Path, scope: Scope, snapshot: str) -> tuple[dict[str, Any], ...]:
    try:
        rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid fixture evidence: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"fixture evidence must contain JSON objects: {path}")
    for row in rows:
        if row.get("scope") != {"tenant_id": scope.tenant_id, "project_id": scope.project_id}:
            raise ValueError(f"fixture evidence scope does not match descriptor: {path}")
        if row.get("corpus_snapshot") != snapshot:
            raise ValueError(f"fixture evidence snapshot does not match descriptor: {path}")
        if not all(isinstance(row.get(key), str) and row[key] for key in ("source_id", "source_version", "passage_id", "text", "text_sha256")):
            raise ValueError(f"fixture evidence lacks a required source field: {path}")
    return rows


def _config_runs(package_root: Path) -> list[FixtureRunConfiguration]:
    result: list[FixtureRunConfiguration] = []
    for descriptor_path in sorted((package_root / "fixtures" / "configs").glob("*.json")):
        descriptor = _object(descriptor_path)
        scope = _scope(descriptor.get("scope"))
        snapshot = descriptor.get("corpus_snapshot")
        if not isinstance(snapshot, str) or not snapshot:
            raise ValueError("fixture corpus_snapshot is required")
        record = _fixture_input_path(
            package_root, descriptor_path, descriptor.get("source_records"), config_relative=True
        )
        _records(record, scope, snapshot)
        result.append(
            FixtureRunConfiguration(
                scenario_id=_output_segment(f"isolation-{descriptor_path.stem}"),
                stage=_output_segment("combined"),
                scope=scope,
                corpus_snapshot=snapshot,
                input_records=(record,),
                replacements=(),
                operation="import",
                question=None,
            )
        )
    if not result:
        raise ValueError("fixture catalog has no isolation descriptors")
    return result


def _validate_isolation_contract(
    package_root: Path,
    runs: list[FixtureRunConfiguration],
) -> None:
    """Reconcile the isolation checks with config-derived runs without admitting checks as evidence."""
    descriptor = _object(package_root / "fixtures" / "isolation-cases.json")
    version = descriptor.get("version")
    collision = descriptor.get("local_identity_collision")
    assertions = descriptor.get("assertions")
    if not isinstance(version, str) or not version:
        raise ValueError("isolation fixture version is required")
    if not isinstance(collision, dict):
        raise ValueError("isolation local identity collision is required")
    collision_identity = tuple(
        collision.get(key) for key in ("source_id", "source_version", "passage_id")
    )
    if not all(isinstance(value, str) and value for value in collision_identity):
        raise ValueError("isolation local identity collision is invalid")
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("isolation assertions must be a nonempty list")

    declared_scopes: list[tuple[str, str]] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            raise ValueError("isolation assertion must be an object")
        scope = _scope(assertion.get("query_scope"))
        declared_scopes.append((scope.tenant_id, scope.project_id))
    config_scopes = [(run.scope.tenant_id, run.scope.project_id) for run in runs]
    if sorted(declared_scopes) != sorted(config_scopes):
        raise ValueError("isolation assertions must match config scopes")

    for run in runs:
        if run.corpus_snapshot != version:
            raise ValueError("isolation contract version must match config snapshots")
        rows = tuple(row for path in run.input_records for row in _records(path, run.scope, version))
        if not rows or any(
            tuple(row.get(key) for key in ("source_id", "source_version", "passage_id"))
            != collision_identity
            for row in rows
        ):
            raise ValueError("isolation inputs must share the declared local identity collision")


def _behavior_runs(package_root: Path) -> list[FixtureRunConfiguration]:
    descriptor_path = package_root / "fixtures" / "behavior-cases.json"
    descriptor = _object(descriptor_path)
    scenarios = descriptor.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("behavior fixture scenarios must be a list")
    result: list[FixtureRunConfiguration] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("behavior fixture scenario must be an object")
        scenario_id = _output_segment(scenario.get("id"))
        scope = _scope(scenario.get("query_scope"))
        question = scenario.get("question")
        if question is not None and not isinstance(question, str):
            raise ValueError("behavior fixture question must be text")
        stages = scenario.get("stages")
        if not isinstance(stages, list):
            raise ValueError("behavior fixture stages must be a list")
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("behavior fixture stage must be an object")
            stage_id = _output_segment(stage.get("id"))
            operation, snapshot = stage.get("operation"), stage.get("corpus_snapshot")
            raw_inputs = stage.get("input_records")
            if not all(isinstance(value, str) and value for value in (operation, snapshot)):
                raise ValueError("behavior fixture operation and snapshot are required")
            if not isinstance(raw_inputs, list) or not raw_inputs:
                raise ValueError("behavior fixture stage needs named input_records")
            inputs = tuple(
                _fixture_input_path(package_root, descriptor_path, value, config_relative=False)
                for value in raw_inputs
            )
            source_versions = {
                (str(row["source_id"]), str(row["source_version"]))
                for path in inputs
                for row in _records(path, scope, snapshot)
            }
            replacements: list[FixtureReplacement] = []
            for previous in stage.get("supersedes_source_refs", []):
                if not isinstance(previous, dict) or _scope(previous.get("scope")) != scope:
                    raise ValueError("fixture replacement has an invalid scope")
                source_id, previous_version = previous.get("source_id"), previous.get("source_version")
                candidates = [version for candidate_id, version in source_versions if candidate_id == source_id]
                if not isinstance(source_id, str) or not isinstance(previous_version, str) or len(candidates) != 1:
                    raise ValueError("fixture replacement must explicitly identify one imported source version")
                replacements.append(FixtureReplacement(source_id, previous_version, candidates[0]))
            result.append(
                FixtureRunConfiguration(
                    scenario_id=scenario_id,
                    stage=stage_id,
                    scope=scope,
                    corpus_snapshot=snapshot,
                    input_records=inputs,
                    replacements=tuple(replacements),
                    operation=operation,
                    question=question,
                )
            )
    return result


def load_fixture_catalog(package_root: Path) -> FixtureCatalog:
    """Load supplied fixture descriptors without writing them or treating metadata as evidence."""
    root = package_root.resolve()
    if not (root / "fixtures").is_dir():
        raise ValueError("package root has no fixtures directory")
    isolation_runs = _config_runs(root)
    _validate_isolation_contract(root, isolation_runs)
    runs = tuple([*isolation_runs, *_behavior_runs(root)])
    keys = [(run.scenario_id, run.stage) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("fixture scenario stages must be unique")
    return FixtureCatalog(root, runs, dict(protected_input_hashes(root)))


def resolve_fixture_run(catalog: FixtureCatalog, scenario_id: str, stage: str) -> FixtureRunConfiguration:
    """Return exactly one prevalidated run configuration."""
    for run in catalog.runs:
        if (run.scenario_id, run.stage) == (scenario_id, stage):
            return run
    raise KeyError(f"unknown fixture run: {scenario_id}/{stage}")
