"""Secret-safe command-line orchestration for the evidence KG stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, SecretStr
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .age_projection import graph_counts, refresh_graph
from .contracts import Scope, ScopedPassageIdentity
from .db import create_engine_from_env
from .embedding_client import EmbeddingBatch, EmbeddingClientIdentity, OpenAIEmbeddingClient
from .embeddings import embed_active_passages
from .export import (
    ExportReport,
    export_graph_version,
    promote_history_for_current_graph,
)
from .extraction import build_response_model, extract_selected_passages
from .fixture_catalog import load_fixture_catalog
from .fixture_runs import ModelClients, run_kg_fixtures
from .ingest import ImportRequest, import_source_version, preflight_official_inputs
from .ids import canonical_json, derive_extraction_version
from .jev_client import JevClientIdentity, JevResponse, NoulQuestion, TypeSafeJevClient
from .jev_processing import classify_active_passages
from .llm_client import GenerationClientIdentity, InstructorExtractionClient
from .output_safety import guard_output_path
from .processing_policy import BUNDLE_ORDER, JevSelectionPolicy
from .prompt_resources.extraction import assemble_prompt
from .project_profiles import assert_protected_inputs_unchanged, protected_input_hashes
from .resolution import materialize_extraction
from .retrieval import RetrievalLimits, retrieve_graph_smoke
from .settings import load_embedding_settings, load_generation_settings, load_jev_settings


MODEL_GATE = "KG_RUN_REAL_MODELS"
SMOKE_GATE = "KG_ENABLE_RETRIEVAL_SMOKE"


class RealModelsNotEnabled(RuntimeError):
    """A real-model stage was requested without the explicit complete gate."""


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _protected_descriptor(config: Path, root: Path) -> Path:
    resolved = config.resolve()
    if not resolved.is_file() or not _inside(resolved, root / "configs"):
        raise ValueError("config must be an existing descriptor beneath the protected configs directory")
    return resolved


def _scope_from_descriptor(config: Path) -> Scope:
    try:
        value = json.loads(config.read_text(encoding="utf-8"))
        scope = value["scope"]
        tenant_id, project_id = scope["tenant_id"], scope["project_id"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("descriptor must contain a valid scope") from None
    if not isinstance(tenant_id, str) or not tenant_id or not isinstance(project_id, str) or not project_id:
        raise ValueError("descriptor scope must contain nonempty tenant_id and project_id")
    return Scope(tenant_id, project_id)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _emit(value: object, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")), file=stream)


def _snapshot_digest(hashes: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(hashes.items())), sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def hash_inputs(root: Path, *, output: Path | None = None, verify: Path | None = None) -> dict[str, object]:
    """Write or verify a path-and-content snapshot without exposing supplied contents."""
    if (output is None) == (verify is None):
        raise ValueError("exactly one of output or verify is required")
    resolved_root = root.resolve()
    destination = (
        guard_output_path(output, package_root=resolved_root) if output is not None else None
    )
    current = dict(protected_input_hashes(resolved_root))
    if output is not None:
        assert destination is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"files": current}, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        snapshot_path = destination
    else:
        snapshot_path = verify.resolve()  # type: ignore[union-attr]
        try:
            stored = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError("protected input snapshot is unreadable") from None
        expected = stored.get("files") if isinstance(stored, dict) else None
        if not isinstance(expected, dict) or expected != current:
            raise RuntimeError("protected input snapshot differs")
    return {
        "status": "verified",
        "file_count": len(current),
        "snapshot_sha256": _snapshot_digest(current),
        "snapshot_path": snapshot_path,
    }


def real_model_readiness(env: Mapping[str, str]) -> dict[str, bool]:
    """Return presence/validity booleans only; never return configuration values."""
    enabled = env.get(MODEL_GATE) == "true"
    if not enabled:
        return {
            "enabled": False,
            "embedding_present": False,
            "generation_present": False,
            "jev_present": False,
            "ready": False,
        }

    def valid(loader: Callable[[Mapping[str, str]], object]) -> bool:
        try:
            loader(env)
        except (ValueError, TypeError):
            return False
        return True

    embedding = valid(load_embedding_settings)
    generation = valid(load_generation_settings)
    jev = valid(load_jev_settings)
    return {
        "enabled": enabled,
        "embedding_present": embedding,
        "generation_present": generation,
        "jev_present": jev,
        "ready": enabled and embedding and generation and jev,
    }


def _require_real_models(env: Mapping[str, str]) -> None:
    if not real_model_readiness(env)["ready"]:
        raise RealModelsNotEnabled("real model stage is not enabled with complete valid settings")


def _require_stage(env: Mapping[str, str], loader: Callable[[Mapping[str, str]], object]) -> None:
    if env.get(MODEL_GATE) != "true":
        raise RealModelsNotEnabled("real model stage is not explicitly enabled")
    try:
        loader(env)
    except (ValueError, TypeError):
        raise RealModelsNotEnabled("real model stage settings are missing or invalid") from None


def _embedding_client(env: Mapping[str, str]) -> OpenAIEmbeddingClient:
    _require_stage(env, load_embedding_settings)
    settings = load_embedding_settings(env)
    return OpenAIEmbeddingClient(settings, SecretStr(env[settings.credential_env]))


def _jev_client(env: Mapping[str, str]) -> TypeSafeJevClient:
    _require_stage(env, load_jev_settings)
    return TypeSafeJevClient.from_environment(env)


def _generation_client(env: Mapping[str, str]) -> InstructorExtractionClient:
    _require_stage(env, load_generation_settings)
    return InstructorExtractionClient.from_environment(env)


def _active_graph_version(engine: Engine, scope: Scope) -> str:
    with engine.connect() as connection:
        value = connection.execute(
            text(
                "SELECT graph_version FROM graph_scope "
                "WHERE tenant_id=:tenant AND project_id=:project"
            ),
            {"tenant": scope.tenant_id, "project": scope.project_id},
        ).scalar_one_or_none()
    if value is None:
        raise ValueError("scope has no active graph")
    return str(value)


def export_or_reuse_graph(engine: Engine, scope: Scope, output_root: Path, graph_version: str) -> ExportReport:
    """Publish once, or validate and repoint to an identical immutable version."""
    root = output_root.resolve()
    version_dir = root / "history" / graph_version
    if version_dir.exists() or version_dir.is_symlink():
        return promote_history_for_current_graph(engine, scope, root, graph_version)
    return export_graph_version(engine, scope, root, graph_version)


def _pending_materializations(engine: Engine, scope: Scope) -> list[tuple[ScopedPassageIdentity, UUID]]:
    with engine.connect() as connection:
        rows = list(connection.execute(
            text(
                """
                SELECT pp.source_id,pp.source_version,pp.passage_id,run.extraction_run_id,
                       run.extraction_version,run.model,run.instructor_mode,
                       run.prompt_hash,run.schema_hash,run.normalization,
                       selection.selected_bundles,response.normalization AS active_normalization,
                       response.profile_name,response.profile_aliases,
                       response.profile_source_sha256,response.profile_version
                FROM passage_processing AS pp
                JOIN source_head AS head
                  ON (head.tenant_id,head.project_id,head.source_id,head.source_version)=
                     (pp.tenant_id,pp.project_id,pp.source_id,pp.source_version)
                JOIN jev_selection AS selection
                  ON (selection.selection_id,selection.tenant_id,selection.project_id,
                      selection.source_id,selection.source_version,selection.passage_id)=
                     (pp.active_jev_selection_id,pp.tenant_id,pp.project_id,
                      pp.source_id,pp.source_version,pp.passage_id)
                JOIN jev_response AS response
                  ON (response.response_id,response.tenant_id,response.project_id,
                      response.source_id,response.source_version,response.passage_id)=
                     (selection.response_id,selection.tenant_id,selection.project_id,
                      selection.source_id,selection.source_version,selection.passage_id)
                JOIN llm_extraction_run AS run
                  ON (run.extraction_run_id,run.tenant_id,run.project_id,run.source_id,
                      run.source_version,run.passage_id)=
                     (pp.validated_extraction_run_id,pp.tenant_id,pp.project_id,pp.source_id,
                      pp.source_version,pp.passage_id)
                WHERE pp.tenant_id=:tenant AND pp.project_id=:project AND pp.state='selected'
                  AND run.status='succeeded'
                  AND run.selected_bundles=selection.selected_bundles
                  AND run.input_sha256=response.input_sha256
                  AND run.profile_version=response.profile_version
                  AND run.profile_name IS NOT DISTINCT FROM response.profile_name
                  AND run.profile_aliases=response.profile_aliases
                  AND run.profile_source_sha256 IS NOT DISTINCT FROM response.profile_source_sha256
                  AND run.normalization=response.normalization
                ORDER BY pp.source_id,pp.source_version,pp.passage_id
                """
            ),
            {"tenant": scope.tenant_id, "project": scope.project_id},
        ).mappings())
    current_rows = []
    for row in rows:
        bundles = tuple(row["selected_bundles"] or ())
        prompt_hash = sha256(assemble_prompt(bundles).encode()).hexdigest()
        schema_hash = sha256(
            canonical_json(build_response_model(bundles).model_json_schema()).encode()
        ).hexdigest()
        profile = {
            "name": row["profile_name"],
            "aliases": dict(row["profile_aliases"] or {}),
            "descriptor_sha256": row["profile_source_sha256"],
            "profile_version": row["profile_version"],
        }
        expected_version = derive_extraction_version(
            model=str(row["model"]),
            instructor_mode=str(row["instructor_mode"]),
            selected_bundles=bundles,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            normalization=str(row["active_normalization"]),
            profile=profile,
        )
        if (
            row["prompt_hash"] == prompt_hash
            and row["schema_hash"] == schema_hash
            and row["extraction_version"] == expected_version
        ):
            current_rows.append(row)
    return [
        (
            ScopedPassageIdentity(
                scope.tenant_id,
                scope.project_id,
                str(row["source_id"]),
                str(row["source_version"]),
                str(row["passage_id"]),
            ),
            row["extraction_run_id"],
        )
        for row in current_rows
    ]


def materialize_pending_extractions(engine: Engine, scope: Scope) -> dict[str, object]:
    reports = [materialize_extraction(engine, passage, run_id) for passage, run_id in _pending_materializations(engine, scope)]
    return {
        "materialized": len(reports),
        "materialization_run_ids": [report.materialization_run_id for report in reports],
        "extraction_run_ids": [report.extraction_run_id for report in reports],
    }


def _semantic_audit(engine: Engine, scope: Scope) -> dict[str, object]:
    params = {"tenant": scope.tenant_id, "project": scope.project_id}
    with engine.connect() as connection:
        embedding = [
            dict(row)
            for row in connection.execute(
                text(
                    """SELECT model AS returned_model,dimensions,encoding_format,normalization,
                                      sanitized_origin,credential_env,count(*) AS run_count,
                                      sum(COALESCE((usage->>'prompt_tokens')::bigint,0)) AS prompt_tokens,
                                      sum(COALESCE((usage->>'total_tokens')::bigint,0)) AS total_tokens
                       FROM embedding_run
                       WHERE tenant_id=:tenant AND project_id=:project AND status='succeeded'
                       GROUP BY model,dimensions,encoding_format,normalization,sanitized_origin,credential_env
                       ORDER BY model"""
                ),
                params,
            ).mappings()
        ]
        jev = [
            dict(row)
            for row in connection.execute(
                text(
                    """SELECT model,request_id,usage,probabilities,status
                       FROM jev_response
                       WHERE tenant_id=:tenant AND project_id=:project
                       ORDER BY created_at,response_id"""
                ),
                params,
            ).mappings()
        ]
        generation = [
            dict(row)
            for row in connection.execute(
                text(
                    """SELECT model,usage,status,extraction_version
                       FROM llm_extraction_run
                       WHERE tenant_id=:tenant AND project_id=:project
                       ORDER BY created_at,extraction_run_id"""
                ),
                params,
            ).mappings()
        ]
    return {
        "embedding": {
            "requested_model": "text-embedding-3-large",
            "dimensions": 3072,
            "encoding_format": "float",
            "credential_env": "EMBEDDING_LLM_API_KEY",
            "runs": embedding,
        },
        "jev": {
            "probability_fields": list(BUNDLE_ORDER),
            "responses": jev,
        },
        "generation": {"runs": generation},
    }


def _report_value(report: object, name: str) -> object:
    return report.get(name) if isinstance(report, Mapping) else getattr(report, name)


def run_kg_workflow(config: Path, output_root: Path, *, env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Run the official workflow, with semantic work only behind the complete real-model gate."""
    environment = os.environ if env is None else env
    root = _package_root().resolve()
    descriptor = _protected_descriptor(config, root)
    output_root = guard_output_path(output_root, package_root=root)
    before = dict(protected_input_hashes(root))
    engine: Engine | object | None = None
    stages: list[str] = []
    try:
        scope = _scope_from_descriptor(descriptor)
        engine = create_engine_from_env()
        preflight = preflight_official_inputs(descriptor)
        stages.append("preflight")
        imported = import_source_version(engine, ImportRequest(descriptor))  # type: ignore[arg-type]
        stages.append("import")
        readiness = real_model_readiness(environment)
        semantic: dict[str, object] | None = None
        if readiness["ready"]:
            embedded = embed_active_passages(engine, scope, _embedding_client(environment))  # type: ignore[arg-type]
            stages.append("embed")
            jev = classify_active_passages(engine, scope, _jev_client(environment), JevSelectionPolicy())  # type: ignore[arg-type]
            stages.append("classify-jev")
            extraction = extract_selected_passages(engine, scope, _generation_client(environment), 1)  # type: ignore[arg-type]
            stages.append("extract")
            materialized = materialize_pending_extractions(engine, scope)  # type: ignore[arg-type]
            stages.append("materialize")
            graph = refresh_graph(engine, scope, "semantic")  # type: ignore[arg-type]
            mode = "semantic"
            semantic = {
                "embedding": embedded,
                "jev": jev,
                "extraction": extraction,
                "materialization": materialized,
                "audit": _semantic_audit(engine, scope),  # type: ignore[arg-type]
            }
        else:
            graph = refresh_graph(engine, scope, "structural")  # type: ignore[arg-type]
            mode = "structural"
        stages.append("refresh-graph")
        exported = export_or_reuse_graph(engine, scope, output_root, str(_report_value(graph, "graph_version")))  # type: ignore[arg-type]
        stages.append("export-kg")
        return {
            "status": "succeeded" if mode == "semantic" else "completed_structural",
            "mode": mode,
            "semantic_status": "succeeded" if mode == "semantic" else "not_run",
            "real_model_readiness": readiness,
            "scope": scope,
            "stages": stages,
            "preflight": preflight,
            "import": imported,
            "semantic": semantic,
            "graph": graph,
            "graph_version": _report_value(graph, "graph_version"),
            "export": exported,
        }
    finally:
        try:
            assert_protected_inputs_unchanged(root, before)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()


@dataclass
class _FixtureEmbedding:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity(
        provider="test-double",
        model="fixture-embedding",
        dimensions=3072,
        encoding_format="float",
        normalization="provider_unit_length_v1",
        sanitized_origin="https://test.invalid/v1",
        credential_env="OFFLINE_FIXTURE",
    )

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch(
            self.identity.model,
            tuple((1.0,) * 3072 for _ in texts),
            3072,
            "float",
            self.identity.normalization,
            len(texts),
            len(texts),
        )


@dataclass
class _FixtureJev:
    identity: JevClientIdentity = JevClientIdentity(
        provider="test-double",
        model="fixture-jev",
        credential_env="OFFLINE_FIXTURE",
        max_attempts=1,
        max_concurrency=1,
    )

    def classify(self, text_value: str, questions: Sequence[NoulQuestion]) -> JevResponse:
        return JevResponse(self.identity.model, {name: 1.0 for name in BUNDLE_ORDER})


@dataclass
class _FixtureExtraction:
    identity: GenerationClientIdentity = GenerationClientIdentity(
        provider="test-double",
        model="fixture-extraction",
        sanitized_origin="https://test.invalid/v1",
        credential_env="OFFLINE_FIXTURE",
        instructor_mode="test-double",
        max_attempts=1,
    )

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
        sentences = tuple(part.strip().rstrip(".!?") for part in model_text.split(". ") if part.strip())
        claim = max(sentences, key=len)
        return response_model.model_validate(
            {
                "entities": [],
                "assertions": [{"local_id": "assertion", "claim": claim, "supporting_quote": claim}],
                "relationships": [],
                "source_refs": [],
            }
        )


def fixture_model_clients() -> ModelClients:
    return ModelClients(_FixtureEmbedding(), _FixtureJev(), _FixtureExtraction())


def run_fixture_workflow(package_root: Path, output_root: Path) -> dict[str, object]:
    root = package_root.resolve()
    requested_output = output_root.absolute()
    expected_output = (root / "results" / "kg" / "fixtures").absolute()
    output = guard_output_path(requested_output, package_root=root)
    if requested_output != expected_output or not _inside(output, root):
        raise ValueError("fixture output must be exactly the package results/kg/fixtures directory")
    before = dict(protected_input_hashes(root))
    engine: Engine | object | None = None
    try:
        catalog = load_fixture_catalog(root)
        engine = create_engine_from_env()
        report = run_kg_fixtures(engine, catalog, output, fixture_model_clients())
        return {
            "status": "succeeded",
            "output_root": output,
            "run_count": len(report.runs),
            "reconnect_verified": report.reconnect_verified,
            "runs": report.runs,
        }
    finally:
        try:
            assert_protected_inputs_unchanged(root, before)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()


def inspect_kg_workflow(
    config: Path,
    *,
    env: Mapping[str, str] | None = None,
    question: str | None = None,
    trace_output: Path | None = None,
) -> dict[str, object]:
    """Inspect persisted state; retrieval is the only optional write and is doubly gated."""
    environment = os.environ if env is None else env
    root = _package_root().resolve()
    descriptor = _protected_descriptor(config, root)
    guarded_trace = (
        guard_output_path(trace_output, package_root=root) if trace_output is not None else None
    )
    before = dict(protected_input_hashes(root))
    engine: Engine | object | None = None
    try:
        scope = _scope_from_descriptor(descriptor)
        engine = create_engine_from_env()
        with engine.connect() as connection:
            counts = connection.execute(
                text(
                    """SELECT
                         (SELECT count(*) FROM source_head WHERE tenant_id=:tenant AND project_id=:project) AS sources,
                         (SELECT count(*) FROM source_passage AS passage JOIN source_head AS head
                           ON (head.tenant_id,head.project_id,head.source_id,head.source_version)=
                              (passage.tenant_id,passage.project_id,passage.source_id,passage.source_version)
                          WHERE passage.tenant_id=:tenant AND passage.project_id=:project) AS passages"""
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).mappings().one()
            active = connection.execute(
                text(
                    "SELECT graph_version,active_graph_name,index_version,embedding_version FROM graph_scope "
                    "WHERE tenant_id=:tenant AND project_id=:project"
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).mappings().one_or_none()
        report: dict[str, object] = {
            "status": "inspected",
            "scope": scope,
            "source_count": int(counts["sources"]),
            "passage_count": int(counts["passages"]),
            "active_graph": dict(active) if active else None,
            "retrieval_smoke": "not_run",
        }
        enabled = environment.get(SMOKE_GATE) == "true"
        if enabled:
            if not question or trace_output is None:
                raise ValueError("retrieval smoke requires a question and trace output")
            index_version = active["index_version"] if active is not None else None
            if index_version is None:
                raise ValueError("retrieval smoke requires an index bound to the active graph")
            trace = retrieve_graph_smoke(
                engine,
                scope,
                question,
                _embedding_client(environment),
                str(index_version),
                RetrievalLimits(),
            )
            assert guarded_trace is not None
            destination = guarded_trace
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(_jsonable(trace), sort_keys=True, separators=(",", ":")) + "\n")
            report["retrieval_smoke"] = "succeeded"
            report["trace_output"] = destination
            report["trace"] = trace
        return report
    finally:
        try:
            assert_protected_inputs_unchanged(root, before)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()


def _guarded_config_call(config: Path, operation: Callable[[Path, Scope, Engine], object]) -> object:
    root = _package_root().resolve()
    descriptor = _protected_descriptor(config, root)
    before = dict(protected_input_hashes(root))
    engine: Engine | object | None = None
    try:
        engine = create_engine_from_env()
        return operation(descriptor, _scope_from_descriptor(descriptor), engine)
    finally:
        try:
            assert_protected_inputs_unchanged(root, before)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m potency_kg.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hashes = subparsers.add_parser("hash-inputs")
    destination = hashes.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify", type=Path)

    for name in ("preflight", "import", "embed", "classify-jev", "extract", "materialize"):
        _add_config(subparsers.add_parser(name))

    refresh = subparsers.add_parser("refresh-graph")
    _add_config(refresh)
    refresh.add_argument("--mode", choices=("structural", "semantic"), required=True)
    refresh.add_argument("--allow-partial", action="store_true")
    refresh.add_argument("--force", action="store_true")

    export = subparsers.add_parser("export-kg")
    _add_config(export)
    export.add_argument("--output", required=True, type=Path)

    inspect = subparsers.add_parser("inspect-kg")
    _add_config(inspect)
    inspect.add_argument("--question")
    inspect.add_argument("--trace-output", type=Path)

    run = subparsers.add_parser("run-kg")
    _add_config(run)
    run.add_argument("--output", required=True, type=Path)

    fixtures = subparsers.add_parser("run-kg-fixtures")
    fixtures.add_argument("--output", type=Path, default=Path("results/kg/fixtures"))
    return parser


def _execute(args: argparse.Namespace, env: Mapping[str, str]) -> object:
    command = args.command
    if command == "hash-inputs":
        return hash_inputs(_package_root(), output=args.output, verify=args.verify)
    if command == "run-kg":
        return run_kg_workflow(args.config, args.output, env=env)
    if command == "run-kg-fixtures":
        return run_fixture_workflow(_package_root(), args.output)
    if command == "inspect-kg":
        return inspect_kg_workflow(
            args.config,
            env=env,
            question=args.question,
            trace_output=args.trace_output,
        )

    def operation(config: Path, scope: Scope, engine: Engine) -> object:
        if command == "preflight":
            return preflight_official_inputs(config)
        if command == "import":
            return import_source_version(engine, ImportRequest(config))
        if command == "embed":
            return embed_active_passages(engine, scope, _embedding_client(env))
        if command == "classify-jev":
            return classify_active_passages(engine, scope, _jev_client(env), JevSelectionPolicy())
        if command == "extract":
            report = extract_selected_passages(engine, scope, _generation_client(env), 1)
            return {"report": report, "audit": _semantic_audit(engine, scope)["generation"]}
        if command == "materialize":
            return materialize_pending_extractions(engine, scope)
        if command == "refresh-graph":
            return refresh_graph(engine, scope, args.mode, args.allow_partial, args.force)
        if command == "export-kg":
            version = _active_graph_version(engine, scope)
            return export_or_reuse_graph(engine, scope, args.output, version)
        raise ValueError("unknown command")

    return _guarded_config_call(args.config, operation)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result = _execute(args, os.environ)
        _emit(result)
        return 0
    except SystemExit:
        raise
    except Exception as error:
        _emit(
            {
                "status": "failed",
                "command": getattr(locals().get("args"), "command", None),
                "error_type": type(error).__name__,
                "detail": "stage failed; no credential values or raw provider responses are emitted",
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
