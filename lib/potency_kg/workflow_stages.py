"""Protected, independently rerunnable owners for KG workflow stages."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .age_projection import refresh_graph
from .alias_discovery import (
    DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY,
    AliasDiscoveryReport,
    discover_aliases_deterministic,
    discover_aliases_llm,
)
from .aliases import (
    AliasCatalogStatus,
    AliasDecision,
    AliasRecord,
    AliasSeedReport,
    alias_catalog_status,
    decide_alias,
    list_aliases,
    load_alias_seeds,
)
from .contracts import Scope
from .db import create_engine_from_env
from .embedding_client import OpenAIEmbeddingClient
from .embeddings import embed_active_passages
from .export import export_graph_version, promote_history_for_current_graph
from .extraction import extract_selected_passages
from .ingest import ImportRequest, import_source_version, preflight_official_inputs
from .jev_client import TypeSafeJevClient
from .jev_processing import classify_active_passages
from .llm_client import InstructorExtractionClient
from .output_safety import guard_output_path
from .processing_policy import JevSelectionPolicy
from .project_profiles import assert_protected_inputs_unchanged, protected_input_hashes
from .resolution import materialize_extraction
from .settings import load_embedding_settings, load_generation_settings, load_jev_settings


MODEL_GATE = "KG_RUN_REAL_MODELS"


class RealModelsNotEnabled(RuntimeError):
    """A credential-backed stage was called without its explicit gate."""


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _protected_descriptor(config: Path, root: Path) -> Path:
    descriptor = config.resolve()
    if not descriptor.is_file() or not _inside(descriptor, root / "configs"):
        raise ValueError("config must be an existing descriptor beneath the protected configs directory")
    return descriptor


def _scope_from_descriptor(config: Path) -> Scope:
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        tenant_id = payload["scope"]["tenant_id"]
        project_id = payload["scope"]["project_id"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("descriptor must contain a valid scope") from None
    if not isinstance(tenant_id, str) or not tenant_id or not isinstance(project_id, str) or not project_id:
        raise ValueError("descriptor scope must contain nonempty tenant_id and project_id")
    return Scope(tenant_id, project_id)


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


def _semantic_workflow_ready(env: Mapping[str, str]) -> bool:
    """Keep the convenience workflow structural until every semantic stage is configured."""
    if env.get(MODEL_GATE) != "true":
        return False
    for loader in (load_embedding_settings, load_jev_settings, load_generation_settings):
        try:
            loader(env)
        except (ValueError, TypeError):
            return False
    return True


def _guarded_stage(config: Path, operation: Callable[[Path, Scope, Engine], object]) -> object:
    root = _package_root().resolve()
    descriptor = _protected_descriptor(config, root)
    before = dict(protected_input_hashes(root))
    engine: Engine | object | None = None
    try:
        engine = create_engine_from_env()
        return operation(descriptor, _scope_from_descriptor(descriptor), engine)  # type: ignore[arg-type]
    finally:
        try:
            assert_protected_inputs_unchanged(root, before)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()


def preflight_stage(config: Path) -> object:
    return _guarded_stage(config, lambda descriptor, _scope, _engine: preflight_official_inputs(descriptor))


def import_stage(config: Path) -> object:
    return _guarded_stage(config, lambda descriptor, _scope, engine: import_source_version(engine, ImportRequest(descriptor)))


def load_alias_seeds_stage(config: Path) -> AliasSeedReport:
    return _guarded_stage(config, lambda _descriptor, scope, engine: load_alias_seeds(engine, scope))  # type: ignore[return-value]


def discover_aliases_deterministic_stage(config: Path) -> AliasDiscoveryReport:
    return _guarded_stage(config, lambda _descriptor, scope, engine: discover_aliases_deterministic(engine, scope))  # type: ignore[return-value]


def discover_aliases_llm_stage(
    config: Path, *, enabled: bool, env: Mapping[str, str] | None = None
) -> AliasDiscoveryReport:
    environment = os.environ if env is None else env
    if not enabled:
        return _guarded_stage(
            config,
            lambda _descriptor, _scope, _engine: AliasDiscoveryReport(
                "llm", 0, 0, 0, 0, 0, 0, ()
            ),
        )  # type: ignore[return-value]
    if enabled:
        _require_stage(environment, load_generation_settings)
    return _guarded_stage(
        config,
        lambda _descriptor, scope, engine: discover_aliases_llm(
            engine, scope, _generation_client(environment), enabled=enabled
        ),
    )  # type: ignore[return-value]


def inspect_aliases_stage(
    config: Path, *, status: str | None = None, entity_type: str | None = None
) -> tuple[AliasRecord, ...]:
    def inspect(_descriptor: Path, scope: Scope, engine: Engine) -> tuple[AliasRecord, ...]:
        with engine.connect() as connection:
            return tuple(list_aliases(connection, scope, status=status, entity_type=entity_type))
    return _guarded_stage(config, inspect)  # type: ignore[return-value]


def review_alias_stage(config: Path, decision: AliasDecision) -> AliasRecord:
    return _guarded_stage(config, lambda _descriptor, scope, engine: decide_alias(engine, scope, decision))  # type: ignore[return-value]


def alias_status_stage(config: Path) -> AliasCatalogStatus:
    def inspect(_descriptor: Path, scope: Scope, engine: Engine) -> AliasCatalogStatus:
        with engine.connect() as connection:
            return alias_catalog_status(connection, scope)
    return _guarded_stage(config, inspect)  # type: ignore[return-value]


def embed_stage(config: Path, *, env: Mapping[str, str] | None = None) -> object:
    environment = os.environ if env is None else env
    return _guarded_stage(config, lambda _descriptor, scope, engine: embed_active_passages(engine, scope, _embedding_client(environment)))


def classify_jev_stage(config: Path, *, env: Mapping[str, str] | None = None) -> object:
    environment = os.environ if env is None else env
    return _guarded_stage(config, lambda _descriptor, scope, engine: classify_active_passages(engine, scope, _jev_client(environment), JevSelectionPolicy()))


def extract_stage(config: Path, *, env: Mapping[str, str] | None = None) -> object:
    environment = os.environ if env is None else env
    return _guarded_stage(config, lambda _descriptor, scope, engine: extract_selected_passages(engine, scope, _generation_client(environment), 1))


def materialize_stage(config: Path) -> object:
    # Each validated run remains an independently materializable write unit.
    from .cli import materialize_pending_extractions
    return _guarded_stage(config, lambda _descriptor, scope, engine: materialize_pending_extractions(engine, scope))


def refresh_graph_stage(config: Path, *, mode: Literal["structural", "semantic"], allow_partial: bool = False, force: bool = False) -> object:
    return _guarded_stage(config, lambda _descriptor, scope, engine: refresh_graph(engine, scope, mode, allow_partial, force))


def _active_graph_version(engine: Engine, scope: Scope) -> str:
    with engine.connect() as connection:
        version = connection.execute(text("SELECT graph_version FROM graph_scope WHERE tenant_id=:tenant AND project_id=:project"), {"tenant": scope.tenant_id, "project": scope.project_id}).scalar_one_or_none()
    if version is None:
        raise ValueError("scope has no active graph")
    return str(version)


def export_stage(config: Path, output_root: Path) -> object:
    root = _package_root().resolve()
    output = guard_output_path(output_root, package_root=root)
    def export(_descriptor: Path, scope: Scope, engine: Engine) -> object:
        version = _active_graph_version(engine, scope)
        location = output / "history" / version
        return promote_history_for_current_graph(engine, scope, output, version) if location.exists() or location.is_symlink() else export_graph_version(engine, scope, output, version)
    return _guarded_stage(config, export)


def inspect_stage(config: Path) -> object:
    return alias_status_stage(config)


def run_kg_workflow(config: Path, output_root: Path, *, env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Convenience composition of the same independently callable stage owners."""
    environment = os.environ if env is None else env
    preflight = preflight_stage(config)
    imported = import_stage(config)
    seeds = load_alias_seeds_stage(config)
    deterministic = discover_aliases_deterministic_stage(config)
    llm = discover_aliases_llm_stage(
        config, enabled=DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY, env=environment
    )
    readiness = _semantic_workflow_ready(environment)
    semantic: dict[str, object] | None = None
    if readiness:
        semantic = {
            "embedding": embed_stage(config, env=environment),
            "jev": classify_jev_stage(config, env=environment),
            "extraction": extract_stage(config, env=environment),
            "materialization": materialize_stage(config),
        }
        graph = refresh_graph_stage(config, mode="semantic")
        mode = "semantic"
    else:
        graph = refresh_graph_stage(config, mode="structural")
        mode = "structural"
    exported = export_stage(config, output_root)
    return {
        "status": "succeeded" if mode == "semantic" else "completed_structural",
        "mode": mode,
        "semantic_status": "succeeded" if mode == "semantic" else "not_run",
        "stages": ["preflight", "import", "load-alias-seeds", "discover-aliases-deterministic", "refresh-graph", "export-kg"],
        "preflight": preflight,
        "import": imported,
        "alias_seeds": seeds,
        "alias_discovery": deterministic,
        "llm_alias_discovery": llm,
        "semantic": semantic,
        "graph": graph,
        "export": exported,
    }
