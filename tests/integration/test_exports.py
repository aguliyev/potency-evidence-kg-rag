from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import pytest
from sqlalchemy import text

import potency_kg.export as export_module
from potency_kg.age_projection import refresh_graph
from potency_kg.cli import export_or_reuse_graph
from potency_kg.contracts import Scope, ScopedSourceVersion
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.embeddings import embed_active_passages
from potency_kg.export import (
    ExportValidationReport,
    export_graph_version,
    promote_current,
    validate_kg_artifacts,
)
from potency_kg.graph_store import write_structural_version


SCOPE = Scope("tenant", "export")


def _seed(connection, scope: Scope = SCOPE, version: str = "v1", body: str = "exact quote") -> None:
    params = {"tenant": scope.tenant_id, "project": scope.project_id, "version": version, "body": body}
    connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source',:version,'title',:source_sha,1,'final')"), {**params, "source_sha": f"source-sha-{version}"})
    connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source',:version,'p1',:body,:text_sha,CAST(:locator AS jsonb))"), {**params, "text_sha": f"text-sha-{version}", "locator": '{"page":7}'})
    connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source',:version,'p1','structural_only','none')"), params)
    connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source',:version) ON CONFLICT (tenant_id,project_id,source_id) DO UPDATE SET source_version=EXCLUDED.source_version"), params)
    write_structural_version(connection, ScopedSourceVersion(*scope.as_tuple(), "source", version))


@dataclass
class _EmbeddingClient:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity(
        "test-double", "export-model", 3072, "float", "provider_unit_length_v1",
        "https://invalid", "KEY",
    )

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        values = tuple(texts)
        return EmbeddingBatch(
            self.identity.model,
            tuple((1.0,) * 3072 for _ in values),
            3072,
            "float",
            self.identity.normalization,
            len(values),
            len(values),
        )


def test_export_is_immutable_validated_and_promotes_relative_current(db_engine, tmp_path) -> None:
    """Publishing before validation or overwriting history would expose incomplete artifacts."""
    with db_engine.begin() as connection:
        _seed(connection)
    graph = refresh_graph(db_engine, SCOPE, mode="structural")

    report = export_graph_version(db_engine, SCOPE, tmp_path, graph.graph_version)
    validation = validate_kg_artifacts(report.version_dir)
    current = promote_current(tmp_path, graph.graph_version)

    assert validation.valid and validation.node_count == 2 and validation.edge_count == 1
    assert current.is_symlink() and not os.readlink(current).startswith("/")
    assert current.resolve() == report.version_dir.resolve()
    assert sorted(path.relative_to(report.version_dir).as_posix() for path in report.version_dir.rglob("*") if path.is_file()) == [
        "extraction-coverage.json", "graph-validation.json", "graph/edges.jsonl",
        "graph/nodes.jsonl", "import-report.json", "manifest.json",
    ]
    manifest = json.loads((current / "manifest.json").read_text())
    assert manifest["scope"] == {"tenant_id": "tenant", "project_id": "export"}
    assert manifest["graph_version"] == graph.graph_version
    assert manifest["semantic_status"] == "not_run"
    assert manifest["index_version"] is None and manifest["embedding_version"] is None
    assert manifest["jev_response_versions"] is None
    assert manifest["jev_selection_versions"] is None
    assert manifest["extraction_versions"] is None
    assert manifest["profile_provenance"] is None
    assert manifest["statuses"] == {
        "graph": "succeeded",
        "import": "succeeded",
        "index": "not_run",
        "semantic": "not_run",
    }
    assert manifest["artifact_hashes"]
    with pytest.raises(FileExistsError):
        export_graph_version(db_engine, SCOPE, tmp_path, graph.graph_version)


def test_same_local_content_in_two_scopes_gets_distinct_shared_history_entries(db_engine, tmp_path) -> None:
    """One shared output root must never reuse or promote another project's immutable export."""
    other_scope = Scope("tenant", "export-other")
    with db_engine.begin() as connection:
        _seed(connection, SCOPE)
        _seed(connection, other_scope)
    first_graph = refresh_graph(db_engine, SCOPE, mode="structural")
    second_graph = refresh_graph(db_engine, other_scope, mode="structural")

    first = export_or_reuse_graph(db_engine, SCOPE, tmp_path, first_graph.graph_version)
    second = export_or_reuse_graph(db_engine, other_scope, tmp_path, second_graph.graph_version)

    assert first.graph_version != second.graph_version
    assert first.version_dir != second.version_dir
    second_manifest = json.loads((second.version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["scope"] == {"tenant_id": "tenant", "project_id": "export-other"}
    assert (tmp_path / "current").resolve() == second.version_dir.resolve()


def test_force_refresh_cannot_promote_history_for_dropped_physical_graph(db_engine, tmp_path) -> None:
    """A force attempt must fail closed instead of promoting a manifest naming the removed graph."""
    with db_engine.begin() as connection:
        _seed(connection)
    first = refresh_graph(db_engine, SCOPE, mode="structural")
    exported = export_or_reuse_graph(db_engine, SCOPE, tmp_path, first.graph_version)
    forced = refresh_graph(db_engine, SCOPE, mode="structural", force=True)

    with pytest.raises(ValueError, match="current active graph"):
        export_or_reuse_graph(db_engine, SCOPE, tmp_path, forced.graph_version)

    assert forced.graph_name != first.graph_name
    assert (tmp_path / "current").resolve() == exported.version_dir.resolve()


def test_semantic_export_uses_reactivated_exact_current_index_not_latest_created(db_engine, tmp_path) -> None:
    """Reactivating older membership must bind export metadata to the stage-selected index pointer."""
    client = _EmbeddingClient()
    with db_engine.begin() as connection:
        _seed(connection, version="v1", body="first evidence")
    first = embed_active_passages(db_engine, SCOPE, client)
    with db_engine.begin() as connection:
        _seed(connection, version="v2", body="second evidence")
    second = embed_active_passages(db_engine, SCOPE, client)
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE source_head SET source_version='v1' WHERE tenant_id=:tenant AND project_id=:project AND source_id='source'"),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        )
    reactivated = embed_active_passages(db_engine, SCOPE, client)
    graph = refresh_graph(db_engine, SCOPE, mode="semantic")

    report = export_graph_version(db_engine, SCOPE, tmp_path, graph.graph_version)
    manifest = json.loads((report.version_dir / "manifest.json").read_text(encoding="utf-8"))

    assert first.index_version != second.index_version
    assert reactivated.index_version == first.index_version
    assert manifest["index_version"] == first.index_version
    assert manifest["embedding_version"] == first.embedding_version


def test_export_root_guard_rejects_symlink_alias_into_protected_input(tmp_path, monkeypatch) -> None:
    """The library export boundary must reject protected aliases before creating history."""
    package = tmp_path / "package"
    for name in ("configs", "corpus", "fixtures", "schema"):
        (package / name).mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(package / "schema", target_is_directory=True)
    monkeypatch.setattr(export_module, "_package_root", lambda: package)

    with pytest.raises(ValueError, match="protected output"):
        export_graph_version(object(), SCOPE, alias, "graph-" + "0" * 64)  # type: ignore[arg-type]

    assert list((package / "schema").iterdir()) == []


def test_failed_promotion_never_changes_current(tmp_path) -> None:
    """A missing or invalid history target must not replace the last complete current symlink."""
    good = tmp_path / "history" / "good"
    good.mkdir(parents=True)
    (good / "manifest.json").write_text("{}")
    os.symlink("history/good", tmp_path / "current")

    with pytest.raises((FileNotFoundError, ValueError)):
        promote_current(tmp_path, "graph-" + "0" * 64)

    assert os.readlink(tmp_path / "current") == "history/good"


def test_failed_staging_validation_preserves_current_and_history(db_engine, tmp_path, monkeypatch) -> None:
    """Renaming an invalid staging tree would expose incomplete immutable history or current."""
    with db_engine.begin() as connection:
        _seed(connection)
    first = refresh_graph(db_engine, SCOPE, mode="structural")
    export_graph_version(db_engine, SCOPE, tmp_path, first.graph_version)

    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v2','title','source-sha-v2',1,'final')"), params)
        connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v2','p1','new exact quote','text-sha-v2',CAST(:locator AS jsonb))"), {**params, "locator": '{"page":8}'})
        connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v2','p1','structural_only','none')"), params)
        connection.execute(text("UPDATE source_head SET source_version='v2' WHERE tenant_id=:tenant AND project_id=:project AND source_id='source'"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v2"))
    second = refresh_graph(db_engine, SCOPE, mode="structural")

    monkeypatch.setattr(
        export_module,
        "validate_kg_artifacts",
        lambda _path: ExportValidationReport(False, 0, 0, ("injected invalid staging",)),
    )
    with pytest.raises(ValueError, match="staged KG artifacts failed validation"):
        export_graph_version(db_engine, SCOPE, tmp_path, second.graph_version)

    assert os.readlink(tmp_path / "current") == f"history/{first.graph_version}"
    assert not (tmp_path / "history" / second.graph_version).exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_validation_rejects_structural_manifest_with_semantic_metadata(db_engine, tmp_path) -> None:
    """Promotion must not accept a structurally impossible manifest even when artifacts hash."""
    with db_engine.begin() as connection:
        _seed(connection)
    graph = refresh_graph(db_engine, SCOPE, mode="structural")
    report = export_graph_version(db_engine, SCOPE, tmp_path, graph.graph_version)
    manifest_path = report.version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["semantic_status"] = "succeeded"
    manifest["statuses"]["semantic"] = "succeeded"
    manifest["embedding_version"] = "embedding-corrupt"
    manifest_path.write_text(json.dumps(manifest))

    validation = validate_kg_artifacts(report.version_dir)

    assert validation.valid is False
    assert "structural manifest semantic metadata is not null/not_run" in validation.errors
