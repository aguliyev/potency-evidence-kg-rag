from __future__ import annotations

import json
import os

import pytest
from sqlalchemy import text

import potency_kg.export as export_module
from potency_kg.age_projection import refresh_graph
from potency_kg.contracts import Scope, ScopedSourceVersion
from potency_kg.export import (
    ExportValidationReport,
    export_graph_version,
    promote_current,
    validate_kg_artifacts,
)
from potency_kg.graph_store import write_structural_version


SCOPE = Scope("tenant", "export")


def _seed(connection) -> None:
    params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
    connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v1','title','source-sha',1,'final')"), params)
    connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source','v1','p1','exact quote','text-sha',CAST(:locator AS jsonb))"), {**params, "locator": '{"page":7}'})
    connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v1','p1','structural_only','none')"), params)
    connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v1')"), params)
    write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))


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
