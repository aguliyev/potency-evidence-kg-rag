"""Immutable, validated publication of the current scoped KG projection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .age_projection import active_graph_name, graph_counts
from .contracts import Scope
from .graph_store import load_current_projection
from .ids import canonical_json
from .project_profiles import trusted_descriptor_path_for_profile


_GRAPH_VERSION = re.compile(r"^graph-[a-f0-9]{64}$")
_REQUIRED_FILES = frozenset(
    {
        "manifest.json",
        "import-report.json",
        "extraction-coverage.json",
        "graph-validation.json",
        "graph/nodes.jsonl",
        "graph/edges.jsonl",
    }
)


@dataclass(frozen=True)
class ExportReport:
    graph_version: str
    version_dir: Path
    current_path: Path
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class ExportValidationReport:
    valid: bool
    node_count: int
    edge_count: int
    errors: tuple[str, ...]


def _validate_graph_version(graph_version: str) -> str:
    if not isinstance(graph_version, str) or _GRAPH_VERSION.fullmatch(graph_version) is None:
        raise ValueError("invalid internally derived graph version")
    return graph_version


def _scope_dict(scope: Scope) -> dict[str, str]:
    return {"tenant_id": scope.tenant_id, "project_id": scope.project_id}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _source_refs(connection: Connection, scope: Scope, table: str, id_column: str) -> dict[str, list[dict[str, object]]]:
    if (table, id_column) not in {
        ("kg_node_source_ref", "node_id"),
        ("kg_edge_source_ref", "edge_id"),
    }:
        raise ValueError("invalid source-reference relation")
    if table == "kg_node_source_ref":
        provenance_join = """
            LEFT JOIN assertion AS evidence_assertion
              ON (evidence_assertion.tenant_id,evidence_assertion.project_id,evidence_assertion.assertion_id) =
                 (ref.tenant_id,ref.project_id,ref.node_id)
        """
    else:
        provenance_join = """
            LEFT JOIN kg_edge AS evidence_edge
              ON (evidence_edge.tenant_id,evidence_edge.project_id,evidence_edge.edge_id) =
                 (ref.tenant_id,ref.project_id,ref.edge_id)
            LEFT JOIN assertion AS evidence_assertion
              ON (evidence_assertion.tenant_id,evidence_assertion.project_id,evidence_assertion.assertion_id) =
                 (evidence_edge.tenant_id,evidence_edge.project_id,evidence_edge.assertion_id)
        """
    rows = connection.execute(
        text(
            f"""SELECT ref.{id_column} AS logical_id,ref.source_id,ref.source_version,
                       ref.passage_id,COALESCE(evidence_assertion.original_quote,passage.text) AS quote,
                       passage.locator
                FROM {table} AS ref
                JOIN source_passage AS passage
                  ON (passage.tenant_id,passage.project_id,passage.source_id,
                      passage.source_version,passage.passage_id) =
                     (ref.tenant_id,ref.project_id,ref.source_id,ref.source_version,ref.passage_id)
                {provenance_join}
                WHERE ref.tenant_id=:tenant AND ref.project_id=:project
                ORDER BY ref.{id_column},ref.source_id,ref.source_version,ref.passage_id"""
        ),
        {"tenant": scope.tenant_id, "project": scope.project_id},
    ).mappings()
    result: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        locator = row["locator"] if isinstance(row["locator"], dict) else {}
        result.setdefault(str(row["logical_id"]), []).append(
            {
                "scope": _scope_dict(scope),
                "source_id": str(row["source_id"]),
                "source_version": str(row["source_version"]),
                "passage_id": str(row["passage_id"]),
                "page": locator.get("page"),
                "section": locator.get("section"),
                "quote": str(row["quote"]),
            }
        )
    return result


def _export_rows(connection: Connection, scope: Scope, mode: str, allow_partial: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshot = load_current_projection(connection, scope, mode, allow_partial)
    node_refs = _source_refs(connection, scope, "kg_node_source_ref", "node_id")
    edge_refs = _source_refs(connection, scope, "kg_edge_source_ref", "edge_id")
    assertion_rows = {
        str(row["assertion_id"]): dict(row)
        for row in connection.execute(
            text(
                """SELECT assertion.assertion_id,assertion.authority,assertion.status,
                          assertion.review_state,assertion.created_at,run.extraction_version
                   FROM assertion
                   LEFT JOIN llm_extraction_run AS run
                     ON (run.extraction_run_id,run.tenant_id,run.project_id) =
                        (assertion.extraction_run_id,assertion.tenant_id,assertion.project_id)
                   WHERE assertion.tenant_id=:tenant AND assertion.project_id=:project"""
            ),
            {"tenant": scope.tenant_id, "project": scope.project_id},
        ).mappings()
    }
    nodes: list[dict[str, object]] = []
    for row in snapshot.nodes:
        node_id = str(row["node_id"])
        provenance = assertion_rows.get(node_id, {})
        item: dict[str, object] = {
            "id": node_id,
            "type": str(row["node_type"]),
            "label": str(row["node_type"]),
            "scope": _scope_dict(scope),
            "attributes": row["attributes"] or {},
            "source_refs": node_refs.get(node_id, []),
        }
        if provenance:
            item.update(
                {
                    "actor": provenance.get("authority") or "instructor",
                    "version": provenance.get("extraction_version"),
                    "created_at": provenance["created_at"].isoformat(),
                    "review_state": provenance["review_state"],
                }
            )
        nodes.append(item)
    nodes.sort(key=lambda row: str(row["id"]))
    edges: list[dict[str, object]] = []
    for row in snapshot.edges:
        edge_id = str(row["edge_id"])
        provenance = assertion_rows.get(str(row["assertion_id"]), {}) if row.get("assertion_id") else {}
        edges.append(
            {
                "id": edge_id,
                "type": str(row["edge_type"]),
                "source": str(row["source_node_id"]),
                "target": str(row["target_node_id"]),
                "scope": _scope_dict(scope),
                "status": provenance.get("status") or "system_generated",
                "source_refs": edge_refs.get(edge_id, []),
                "actor": provenance.get("authority") or "system",
                "version": provenance.get("extraction_version") or "structural-v1",
                "created_at": provenance["created_at"].isoformat() if provenance else None,
                "review_state": provenance.get("review_state") or "not_applicable",
                "attributes": row["attributes"] or {},
                "assertion_id": str(row["assertion_id"]) if row.get("assertion_id") else None,
            }
        )
    edges.sort(key=lambda row: str(row["id"]))
    return nodes, edges


def _metadata(connection: Connection, scope: Scope, graph_version: str) -> dict[str, object]:
    params = {"tenant": scope.tenant_id, "project": scope.project_id, "version": graph_version}
    active = connection.execute(
        text(
            """SELECT active_graph_name,graph_version FROM graph_scope
               WHERE tenant_id=:tenant AND project_id=:project"""
        ),
        params,
    ).mappings().one_or_none()
    if active is None or active["graph_version"] != graph_version:
        raise ValueError("requested graph version is not active for scope")
    refresh = connection.execute(
        text(
            """SELECT mode,allow_partial,omitted,validation FROM graph_refresh_run
               WHERE tenant_id=:tenant AND project_id=:project AND graph_version=:version
                 AND shadow_graph_name=:graph_name AND status='succeeded'
               ORDER BY finished_at DESC LIMIT 1"""
        ),
        {**params, "graph_name": active["active_graph_name"]},
    ).mappings().one()
    heads = [
        {"source_id": str(row.source_id), "source_version": str(row.source_version)}
        for row in connection.execute(
            text(
                """SELECT source_id,source_version FROM source_head
                   WHERE tenant_id=:tenant AND project_id=:project ORDER BY source_id"""
            ),
            params,
        )
    ]
    index = connection.execute(
        text(
            """SELECT index_version,embedding_version FROM index_snapshot
               WHERE tenant_id=:tenant AND project_id=:project ORDER BY created_at DESC LIMIT 1"""
        ),
        params,
    ).mappings().one_or_none()
    extraction = connection.execute(
        text(
            """SELECT DISTINCT run.extraction_version,run.profile_name,run.profile_version,
                              run.profile_source_sha256,run.prompt_hash,run.schema_hash
               FROM passage_processing AS processing
               JOIN source_head AS head
                 ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                    (processing.tenant_id,processing.project_id,processing.source_id,processing.source_version)
               JOIN llm_extraction_run AS run
                 ON (run.extraction_run_id,run.tenant_id,run.project_id,run.source_id,
                     run.source_version,run.passage_id) =
                    (processing.active_extraction_run_id,processing.tenant_id,processing.project_id,
                     processing.source_id,processing.source_version,processing.passage_id)
               WHERE processing.tenant_id=:tenant AND processing.project_id=:project
               ORDER BY run.extraction_version"""
        ),
        params,
    ).mappings().all()
    jev = connection.execute(
        text(
            """SELECT DISTINCT response.jev_response_version,selection.jev_selection_version,
                              response.prompt_hash,response.bundle_hash
               FROM passage_processing AS processing
               JOIN source_head AS head
                 ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                    (processing.tenant_id,processing.project_id,processing.source_id,processing.source_version)
               JOIN jev_selection AS selection
                 ON (selection.selection_id,selection.tenant_id,selection.project_id) =
                    (processing.active_jev_selection_id,processing.tenant_id,processing.project_id)
               JOIN jev_response AS response
                 ON (response.response_id,response.tenant_id,response.project_id) =
                    (selection.response_id,selection.tenant_id,selection.project_id)
               WHERE processing.tenant_id=:tenant AND processing.project_id=:project
               ORDER BY response.jev_response_version,selection.jev_selection_version"""
        ),
        params,
    ).mappings().all()
    source_hashes = [
        {
            "source_id": str(row.source_id),
            "source_version": str(row.source_version),
            "source_sha256": str(row.source_sha256),
        }
        for row in connection.execute(
            text(
                """SELECT document.source_id,document.source_version,document.source_sha256
                   FROM source_document AS document JOIN source_head AS head
                     ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                        (document.tenant_id,document.project_id,document.source_id,document.source_version)
                   WHERE document.tenant_id=:tenant AND document.project_id=:project
                   ORDER BY document.source_id"""
            ),
            params,
        )
    ]
    return {
        "graph_name": str(active["active_graph_name"]),
        "mode": str(refresh["mode"]),
        "allow_partial": bool(refresh["allow_partial"]),
        "omitted": refresh["omitted"],
        "graph_validation": refresh["validation"],
        "active_heads": heads,
        "index_version": str(index["index_version"]) if index else None,
        "embedding_version": str(index["embedding_version"]) if index else None,
        "extraction": [dict(row) for row in extraction],
        "jev": [dict(row) for row in jev],
        "source_hashes": source_hashes,
    }


def _fsync_tree(root: Path) -> None:
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file())):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    for directory in sorted((candidate for candidate in root.rglob("*") if candidate.is_dir()), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_kg_artifacts(version_dir: Path) -> ExportValidationReport:
    errors: list[str] = []
    if not version_dir.is_dir():
        return ExportValidationReport(False, 0, 0, ("version directory is missing",))
    files = {path.relative_to(version_dir).as_posix() for path in version_dir.rglob("*") if path.is_file()}
    missing = sorted(_REQUIRED_FILES - files)
    if missing:
        errors.append(f"missing artifacts: {missing}")
        return ExportValidationReport(False, 0, 0, tuple(errors))
    extra = sorted(files - _REQUIRED_FILES)
    if extra:
        errors.append(f"unexpected artifacts: {extra}")
    try:
        manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        nodes = [json.loads(line) for line in (version_dir / "graph/nodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        edges = [json.loads(line) for line in (version_dir / "graph/edges.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError, TypeError):
        return ExportValidationReport(False, 0, 0, ("artifact JSON is unreadable",))
    if not isinstance(manifest, dict):
        return ExportValidationReport(False, 0, 0, ("manifest is malformed",))
    required_manifest_fields = {
        "scope", "graph_version", "graph_name", "projection_mode", "active_heads",
        "index_version", "embedding_version", "jev_response_versions",
        "jev_selection_versions", "extraction_versions", "profile_provenance",
        "source_hashes", "prompt_hashes", "statuses", "hashes", "artifact_hashes",
        "counts", "later_phase_exclusions",
    }
    if not required_manifest_fields.issubset(manifest):
        errors.append("manifest required metadata is missing")
    version = manifest.get("graph_version")
    if (
        not isinstance(version, str)
        or _GRAPH_VERSION.fullmatch(version) is None
        or (
            version_dir.name != version
            and not version_dir.name.startswith(f".staging-{version}-")
        )
    ):
        errors.append("manifest graph version differs from history directory")
    statuses = manifest.get("statuses")
    flat_statuses = {
        "import": manifest.get("import_status"),
        "semantic": manifest.get("semantic_status"),
        "index": manifest.get("index_status"),
        "graph": manifest.get("graph_status"),
    }
    if statuses != flat_statuses or flat_statuses["import"] != "succeeded" or flat_statuses["graph"] != "succeeded":
        errors.append("manifest statuses are missing or inconsistent")
    if manifest.get("projection_mode") == "structural" and (
        flat_statuses["semantic"] != "not_run"
        or flat_statuses["index"] != "not_run"
        or any(
            manifest.get(key) is not None
            for key in (
                "index_version", "embedding_version", "jev_response_versions",
                "jev_selection_versions", "extraction_versions", "profile_provenance",
                "prompt_hashes",
            )
        )
    ):
        errors.append("structural manifest semantic metadata is not null/not_run")
    elif manifest.get("projection_mode") not in {"structural", "semantic"}:
        errors.append("manifest projection mode is invalid")
    node_ids = [row.get("id") for row in nodes]
    edge_ids = [row.get("id") for row in edges]
    if len(set(node_ids)) != len(node_ids) or node_ids != sorted(node_ids):
        errors.append("node IDs are duplicate or nondeterministically ordered")
    if len(set(edge_ids)) != len(edge_ids) or edge_ids != sorted(edge_ids):
        errors.append("edge IDs are duplicate or nondeterministically ordered")
    for edge in edges:
        if edge.get("source") not in set(node_ids) or edge.get("target") not in set(node_ids):
            errors.append(f"dangling edge endpoint: {edge.get('id')}")
    expected_scope = manifest.get("scope")
    if any(row.get("scope") != expected_scope for row in [*nodes, *edges]):
        errors.append("artifact scope differs from manifest")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        errors.append("manifest artifact hashes are missing")
    else:
        if set(hashes) != files - {"manifest.json"}:
            errors.append("manifest artifact hash membership differs")
        for relative, expected in hashes.items():
            path = version_dir / str(relative)
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"artifact hash differs: {relative}")
    hash_groups = manifest.get("hashes")
    if not isinstance(hash_groups, dict) or (
        hash_groups.get("sources") != manifest.get("source_hashes")
        or hash_groups.get("prompts") != manifest.get("prompt_hashes")
        or hash_groups.get("artifacts") != hashes
    ):
        errors.append("manifest hash groups are missing or inconsistent")
    counts = manifest.get("counts", {})
    if counts.get("nodes") != len(nodes) or counts.get("edges") != len(edges):
        errors.append("manifest graph counts differ")
    return ExportValidationReport(not errors, len(nodes), len(edges), tuple(errors))


def export_graph_version(engine: Engine, scope: Scope, output_root: Path, graph_version: str) -> ExportReport:
    """Stage and validate a version directory before an immutable history rename."""
    version = _validate_graph_version(graph_version)
    root = output_root.resolve()
    history = root / "history"
    target = history / version
    history.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"history destination already exists: {version}")
    staging = root / f".staging-{version}-{uuid4().hex}"
    staging.mkdir(parents=False)
    try:
        (staging / "graph").mkdir()
        with engine.begin() as connection:
            connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            active_graph_name(connection, scope)
            metadata = _metadata(connection, scope, version)
            nodes, edges = _export_rows(
                connection, scope, str(metadata["mode"]), bool(metadata["allow_partial"])
            )
            active_counts = graph_counts(connection, scope)
            if active_counts.as_tuple() != (len(nodes), len(edges)):
                raise ValueError("relational and AGE graph counts differ")
            state_counts = Counter(
                str(row.state)
                for row in connection.execute(
                    text(
                        """SELECT processing.state FROM passage_processing AS processing
                           JOIN source_head AS head
                             ON (head.tenant_id,head.project_id,head.source_id,head.source_version) =
                                (processing.tenant_id,processing.project_id,processing.source_id,processing.source_version)
                           WHERE processing.tenant_id=:tenant AND processing.project_id=:project"""
                    ),
                    {"tenant": scope.tenant_id, "project": scope.project_id},
                )
            )
        _write_jsonl(staging / "graph/nodes.jsonl", nodes)
        _write_jsonl(staging / "graph/edges.jsonl", edges)
        semantic_run = str(metadata["mode"]) == "semantic"
        _write_json(
            staging / "extraction-coverage.json",
            {"scope": _scope_dict(scope), "statuses": dict(sorted(state_counts.items())), "omitted": metadata["omitted"]},
        )
        _write_json(
            staging / "graph-validation.json",
            {"scope": _scope_dict(scope), "graph_version": version, **dict(metadata["graph_validation"])},
        )
        _write_json(
            staging / "import-report.json",
            {
                "corpus_snapshot": metadata["source_hashes"],
                "index_version": metadata["index_version"],
                "imported": sum(state_counts.values()),
                "unchanged": 0,
                "failed": state_counts.get("failed", 0),
                "omitted": metadata["omitted"],
                "errors": [],
                "scope": _scope_dict(scope),
                "per_source": metadata["active_heads"],
                "semantic_extraction_coverage": dict(sorted(state_counts.items())),
            },
        )
        artifact_hashes = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file())
        }
        extraction = metadata["extraction"]
        jev = metadata["jev"]
        jev_response_versions = (
            sorted({str(row["jev_response_version"]) for row in jev}) if semantic_run else None
        )
        jev_selection_versions = (
            sorted({str(row["jev_selection_version"]) for row in jev}) if semantic_run else None
        )
        extraction_versions = (
            sorted({str(row["extraction_version"]) for row in extraction}) if semantic_run else None
        )
        profile_provenance = (
            sorted(
                [
                    {
                        "name": row.get("profile_name"),
                        "version": row.get("profile_version"),
                        "source_path": trusted_descriptor_path_for_profile(
                            str(row.get("profile_name") or "")
                        ),
                        "source_sha256": row.get("profile_source_sha256"),
                    }
                    for row in extraction
                ],
                key=canonical_json,
            )
            if semantic_run
            else None
        )
        prompt_hashes = (
            sorted(
                {
                    str(value)
                    for row in [*extraction, *jev]
                    for key, value in row.items()
                    if key.endswith("hash") and value
                }
            )
            if semantic_run
            else None
        )
        manifest = {
            "contract_phase": "knowledge_graph",
            "scope": _scope_dict(scope),
            "graph_version": version,
            "graph_name": metadata["graph_name"],
            "projection_mode": metadata["mode"],
            "allow_partial": metadata["allow_partial"],
            "active_heads": metadata["active_heads"],
            "index_version": metadata["index_version"] if semantic_run else None,
            "embedding_version": metadata["embedding_version"] if semantic_run else None,
            "jev_response_versions": jev_response_versions,
            "jev_selection_versions": jev_selection_versions,
            "extraction_versions": extraction_versions,
            "profile_provenance": profile_provenance,
            "source_hashes": metadata["source_hashes"],
            "prompt_hashes": prompt_hashes,
            "import_status": "succeeded",
            "semantic_status": "succeeded" if semantic_run else "not_run",
            "index_status": "succeeded" if semantic_run and metadata["index_version"] else "not_run",
            "graph_status": "succeeded",
            "counts": {"nodes": len(nodes), "edges": len(edges)},
            "artifact_hashes": artifact_hashes,
            "later_phase_exclusions": [
                "answers.jsonl", "traces.jsonl", "evaluation/", "architecture/", "fixtures/"
            ],
        }
        manifest["hashes"] = {
            "sources": metadata["source_hashes"],
            "prompts": manifest["prompt_hashes"],
            "artifacts": artifact_hashes,
        }
        manifest["statuses"] = {
            "import": manifest["import_status"],
            "semantic": manifest["semantic_status"],
            "index": manifest["index_status"],
            "graph": manifest["graph_status"],
        }
        manifest["jev_versions"] = {
            "responses": manifest["jev_response_versions"],
            "selections": manifest["jev_selection_versions"],
        }
        _write_json(staging / "manifest.json", manifest)
        validation = validate_kg_artifacts(staging)
        if not validation.valid:
            raise ValueError("staged KG artifacts failed validation")
        _fsync_tree(staging)
        os.rename(staging, target)
        history_descriptor = os.open(history, os.O_RDONLY)
        try:
            os.fsync(history_descriptor)
        finally:
            os.close(history_descriptor)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    current = promote_current(root, version)
    return ExportReport(version, target, current, len(nodes), len(edges))


def promote_current(output_root: Path, graph_version: str) -> Path:
    """Atomically replace ``current`` with a relative link to validated history."""
    version = _validate_graph_version(graph_version)
    root = output_root.resolve()
    target = root / "history" / version
    validation = validate_kg_artifacts(target)
    if not target.is_dir():
        raise FileNotFoundError(f"history version is missing: {version}")
    if not validation.valid:
        raise ValueError("history version is incomplete or invalid")
    temporary = root / f".current-{uuid4().hex}"
    os.symlink(Path("history") / version, temporary)
    try:
        os.replace(temporary, root / "current")
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.is_symlink():
            temporary.unlink()
    return root / "current"
