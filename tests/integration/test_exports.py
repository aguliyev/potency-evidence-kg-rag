from __future__ import annotations

import json
import os
from dataclasses import dataclass
from threading import Event, Thread
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
from potency_kg.ids import derive_scoped_id


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


def test_history_validation_holds_activation_boundary_through_current_promotion(
    db_engine, tmp_path, monkeypatch
) -> None:
    """A force refresh must not invalidate a history target between validation and promotion."""
    with db_engine.begin() as connection:
        _seed(connection)
    graph = refresh_graph(db_engine, SCOPE, mode="structural")
    export_or_reuse_graph(db_engine, SCOPE, tmp_path, graph.graph_version)

    promotion_entered = Event()
    release_promotion = Event()
    refresh_at_activation = Event()
    refresh_done = Event()
    failures: list[BaseException] = []
    real_replace = export_module.os.replace

    def pause_current_replace(source, destination) -> None:
        if os.fspath(destination).endswith("/current"):
            promotion_entered.set()
            if not release_promotion.wait(5):
                raise TimeoutError("test did not release current promotion")
        real_replace(source, destination)

    def promote_history() -> None:
        try:
            export_or_reuse_graph(db_engine, SCOPE, tmp_path, graph.graph_version)
        except BaseException as error:
            failures.append(error)

    def force_refresh() -> None:
        try:
            refresh_graph(
                db_engine,
                SCOPE,
                mode="structural",
                force=True,
                failure_injector=lambda point: (
                    refresh_at_activation.set() if point == "before_activation" else None
                ),
            )
        except BaseException as error:
            failures.append(error)
        finally:
            refresh_done.set()

    monkeypatch.setattr(export_module.os, "replace", pause_current_replace)
    promotion_thread = Thread(target=promote_history)
    refresh_thread = Thread(target=force_refresh)
    promotion_thread.start()
    assert promotion_entered.wait(5)
    refresh_thread.start()
    assert refresh_at_activation.wait(5)
    refresh_completed_while_promotion_paused = refresh_done.wait(0.5)
    release_promotion.set()
    promotion_thread.join(5)
    refresh_thread.join(5)

    assert not promotion_thread.is_alive() and not refresh_thread.is_alive()
    assert failures == []
    assert refresh_completed_while_promotion_paused is False


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


def test_export_guard_rejects_nested_history_symlink_before_write(tmp_path, monkeypatch) -> None:
    """A safe-looking root must not hide a protected destination behind its history child."""
    package = tmp_path / "package"
    for name in ("configs", "corpus", "fixtures", "schema"):
        (package / name).mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    (output / "history").symlink_to(package / "schema", target_is_directory=True)
    monkeypatch.setattr(export_module, "_package_root", lambda: package)

    with pytest.raises(ValueError, match="protected output"):
        export_graph_version(object(), SCOPE, output, "graph-" + "0" * 64)  # type: ignore[arg-type]

    assert list((package / "schema").iterdir()) == []


def test_semantic_correction_exports_only_active_exact_entity_mentions(db_engine, tmp_path) -> None:
    """Shared Product/Application nodes must expose only v2 mention quotes from the active extraction."""
    scope = Scope("tenant", "semantic-correction")
    product_id = derive_scoped_id("product", scope, "shared-product")
    application_id = derive_scoped_id("application", scope, "shared-application")
    run_ids: dict[str, object] = {}
    with db_engine.begin() as connection:
        for version in ("v1", "v2"):
            from uuid import uuid4

            run_id = uuid4()
            run_ids[version] = run_id
            params = {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "version": version,
                "run_id": run_id,
            }
            connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source',:version,'title',:version,1,'final')"), params)
            connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256,locator) VALUES (:tenant,:project,'source',:version,'p1',:body,:version,'{}')"), {**params, "body": f"full passage {version} must not substitute for mentions"})
            connection.execute(
                text(
                    """INSERT INTO llm_extraction_run
                       (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,
                        extraction_version,model,sanitized_origin,credential_env,instructor_mode,
                        selected_bundles,input_sha256,profile_name,profile_aliases,
                        profile_source_sha256,profile_version,status,response)
                       VALUES (:run_id,:tenant,:project,'source',:version,'p1',:extraction_version,
                        'model','https://invalid','KEY','openai','["product_context"]',:input_sha,
                        'empty-v1','{}','profile-sha','profile-v1','succeeded','{}')"""
                ),
                {**params, "extraction_version": f"extraction-{version}", "input_sha": f"input-{version}"},
            )
            connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason,active_extraction_run_id) VALUES (:tenant,:project,'source',:version,'p1','extracted','extracted_successfully',:run_id)"), params)
            write_structural_version(connection, ScopedSourceVersion(*scope.as_tuple(), "source", version))
            assertion_id = derive_scoped_id("assertion", scope, "source", version, "p1", str(run_id), "a1")
            connection.execute(text("INSERT INTO assertion (tenant_id,project_id,assertion_id,source_id,source_version,passage_id,extraction_run_id,claim,chronology_state,review_state,original_quote) VALUES (:tenant,:project,:assertion_id,'source',:version,'p1',:run_id,:claim,'reported','unreviewed',:quote)"), {**params, "assertion_id": assertion_id, "claim": f"claim {version}", "quote": f"assertion quote {version}"})
            connection.execute(text("INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes) VALUES (:tenant,:project,:product_id,'Product','{}'),(:tenant,:project,:application_id,'Application','{}'),(:tenant,:project,:assertion_id,'Assertion','{}') ON CONFLICT DO NOTHING"), {**params, "product_id": product_id, "application_id": application_id, "assertion_id": assertion_id})
            for node_id, mention_id, entity_type, quote in (
                (product_id, f"product-{version}", "Product", f"product mention {version}"),
                (application_id, f"application-{version}", "Application", f"application mention {version}"),
            ):
                connection.execute(text("INSERT INTO entity_mention (tenant_id,project_id,source_id,source_version,passage_id,mention_id,entity_type,text,extraction_run_id,exact_quote) VALUES (:tenant,:project,'source',:version,'p1',:mention_id,:entity_type,:quote,:run_id,:quote)"), {**params, "mention_id": mention_id, "entity_type": entity_type, "quote": quote})
                connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:node_id,'source',:version,'p1') ON CONFLICT DO NOTHING"), {**params, "node_id": node_id})
                connection.execute(text("INSERT INTO kg_node_mention_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id,mention_id,extraction_run_id) VALUES (:tenant,:project,:node_id,'source',:version,'p1',:mention_id,:run_id)"), {**params, "node_id": node_id, "mention_id": mention_id})
            connection.execute(text("INSERT INTO kg_node_source_ref (tenant_id,project_id,node_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:assertion_id,'source',:version,'p1')"), {**params, "assertion_id": assertion_id})
            for suffix, target in (("product", product_id), ("application", application_id)):
                edge_id = derive_scoped_id("evidence", scope, version, suffix)
                connection.execute(text("INSERT INTO kg_edge (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,assertion_id,is_evidence,attributes) VALUES (:tenant,:project,:edge_id,'applies_to',:assertion_id,:target,:assertion_id,true,'{}')"), {**params, "edge_id": edge_id, "assertion_id": assertion_id, "target": target})
                connection.execute(text("INSERT INTO kg_edge_source_ref (tenant_id,project_id,edge_id,source_id,source_version,passage_id) VALUES (:tenant,:project,:edge_id,'source',:version,'p1')"), {**params, "edge_id": edge_id})
        connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES (:tenant,:project,'source','v2')"), {"tenant": scope.tenant_id, "project": scope.project_id})

    embed_active_passages(db_engine, scope, _EmbeddingClient())
    graph = refresh_graph(db_engine, scope, mode="semantic")
    report = export_graph_version(db_engine, scope, tmp_path, graph.graph_version)
    nodes = [json.loads(line) for line in (report.version_dir / "graph" / "nodes.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {node["id"]: node for node in nodes}

    assert [ref["source_version"] for ref in by_id[product_id]["source_refs"]] == ["v2"]
    assert [ref["quote"] for ref in by_id[product_id]["source_refs"]] == ["product mention v2"]
    assert [ref["source_version"] for ref in by_id[application_id]["source_refs"]] == ["v2"]
    assert [ref["quote"] for ref in by_id[application_id]["source_refs"]] == ["application mention v2"]


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
