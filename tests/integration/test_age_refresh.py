from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Sequence

import pytest
from sqlalchemy import text

import potency_kg.age_projection as age_projection
from potency_kg.age_projection import (
    GraphRefreshError,
    active_graph_name,
    graph_counts,
    refresh_graph,
)
from potency_kg.contracts import Scope, ScopedSourceVersion
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.embeddings import embed_active_passages
from potency_kg.graph_store import load_current_projection, write_structural_version


SCOPE = Scope("tenant", "age-refresh")


@dataclass
class _EmbeddingDouble:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity(
        "test-double", "age-test", 3072, "float", "provider_unit_length_v1",
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


def _seed(connection, *, passage_state: str = "structural_only") -> None:
    connection.execute(
        text(
            """INSERT INTO source_document
               (tenant_id,project_id,source_id,source_version,title,source_sha256,
                expected_passage_count,status)
               VALUES (:tenant,:project,'source','v1','title','source-sha',1,'final')"""
        ),
        {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
    )
    connection.execute(
        text(
            """INSERT INTO source_passage
               (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
               VALUES (:tenant,:project,'source','v1','p1','private full passage','text-sha')"""
        ),
        {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
    )
    connection.execute(
        text(
            """INSERT INTO passage_processing
               (tenant_id,project_id,source_id,source_version,passage_id,state,reason)
               VALUES (:tenant,:project,'source','v1','p1',:state,'test')"""
        ),
        {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "state": passage_state},
    )
    connection.execute(
        text(
            """INSERT INTO source_head (tenant_id,project_id,source_id,source_version)
               VALUES (:tenant,:project,'source','v1')"""
        ),
        {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
    )
    write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v1"))


def test_shadow_refresh_reuses_validated_version_and_force_gets_unique_attempt(db_engine) -> None:
    """Dropping/rebuilding an unchanged active graph would weaken last-known-good activation."""
    with db_engine.begin() as connection:
        _seed(connection)

    first = refresh_graph(db_engine, SCOPE, mode="structural")
    reused = refresh_graph(db_engine, SCOPE, mode="structural")
    forced = refresh_graph(db_engine, SCOPE, mode="structural", force=True)

    assert reused.reused is True and reused.graph_name == first.graph_name
    assert reused.refresh_run_id is not None
    assert forced.graph_version == first.graph_version
    assert forced.graph_name != first.graph_name
    with db_engine.begin() as connection:
        assert active_graph_name(connection, SCOPE) == forced.graph_name
        assert graph_counts(connection, SCOPE).as_tuple() == (2, 1)
        graph_names = set(
            connection.execute(
                text("SELECT name FROM ag_catalog.ag_graph WHERE name IN (:first,:forced)"),
                {"first": first.graph_name, "forced": forced.graph_name},
            ).scalars()
        )
        statuses = connection.execute(
            text(
                """SELECT status FROM graph_refresh_run
                   WHERE tenant_id=:tenant AND project_id=:project ORDER BY created_at"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).scalars().all()
    assert graph_names == {forced.graph_name}
    assert statuses == ["succeeded", "succeeded", "succeeded"]


@pytest.mark.parametrize("failure_at", ["after_vertices", "after_edges", "after_validation", "before_activation", "after_activation", "after_cleanup"])
def test_refresh_failure_preserves_previous_graph_and_records_attempt(db_engine, failure_at: str) -> None:
    """Any build or activation failure must leave the complete former graph active."""
    with db_engine.begin() as connection:
        _seed(connection)
    previous = refresh_graph(db_engine, SCOPE, mode="structural")

    with pytest.raises(GraphRefreshError):
        refresh_graph(db_engine, SCOPE, mode="structural", force=True, failure_at=failure_at)

    with db_engine.begin() as connection:
        assert active_graph_name(connection, SCOPE) == previous.graph_name
        assert graph_counts(connection, SCOPE).as_tuple() == (2, 1)
        failed = connection.execute(
            text(
                """SELECT shadow_graph_name,status,error_detail FROM graph_refresh_run
                   WHERE tenant_id=:tenant AND project_id=:project AND status='failed'"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).one()
        graph_names = set(
            connection.execute(
                text("SELECT name FROM ag_catalog.ag_graph WHERE name IN (:previous,:failed)"),
                {"previous": previous.graph_name, "failed": failed.shadow_graph_name},
            ).scalars()
        )
    assert failed.status == "failed" and failed.shadow_graph_name != previous.graph_name
    assert "injected" in failed.error_detail
    assert graph_names == {previous.graph_name}


def test_validation_rejects_mismatched_type_properties(db_engine) -> None:
    """AGE label/type mirrors drive readers and must agree with the relational snapshot."""
    with db_engine.begin() as connection:
        _seed(connection)
    refreshed = refresh_graph(db_engine, SCOPE, mode="structural")

    with db_engine.begin() as connection:
        snapshot = load_current_projection(connection, SCOPE, "structural", False)
        age_projection._prepare_age(connection)
        age_projection._cypher(
            connection,
            refreshed.graph_name,
            "MATCH (n:SourcePassage) SET n.node_type=$node_type RETURN n.logical_id",
            "logical_id agtype",
            {"node_type": "Product"},
        ).all()
        age_projection._cypher(
            connection,
            refreshed.graph_name,
            "MATCH ()-[edge:contains_passage]->() SET edge.edge_type=$edge_type RETURN edge.logical_id",
            "logical_id agtype",
            {"edge_type": "applies_to"},
        ).all()
        validation = age_projection.validate_graph(
            connection, refreshed.graph_name, snapshot
        )

    assert validation.valid is False
    assert any("node type property differs" in error for error in validation.errors)
    assert any("edge type property differs" in error for error in validation.errors)


def test_semantic_refresh_refuses_or_exactly_records_unresolved_passages(db_engine) -> None:
    """Silently omitting selected semantic work would publish a misleading complete version."""
    with db_engine.begin() as connection:
        _seed(connection, passage_state="selected")
    embed_active_passages(db_engine, SCOPE, _EmbeddingDouble())

    with pytest.raises(GraphRefreshError, match="unresolved"):
        refresh_graph(db_engine, SCOPE)
    partial = refresh_graph(db_engine, SCOPE, allow_partial=True)
    complete_version = refresh_graph(db_engine, SCOPE, mode="structural").graph_version

    with db_engine.begin() as connection:
        semantic_attempts = connection.execute(
            text(
                """SELECT allow_partial,status,omitted FROM graph_refresh_run
                   WHERE tenant_id=:tenant AND project_id=:project AND mode='semantic'
                   ORDER BY created_at"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).all()

    assert partial.omitted == (("tenant", "age-refresh", "source", "v1", "p1", "selected"),)
    assert partial.graph_version != complete_version
    assert semantic_attempts == [
        (False, "failed", [["tenant", "age-refresh", "source", "v1", "p1", "selected"]]),
        (True, "succeeded", [["tenant", "age-refresh", "source", "v1", "p1", "selected"]]),
    ]


def test_concurrent_readers_observe_only_complete_old_or_new_graphs(db_engine) -> None:
    """Moving the pointer before validation would let a reader observe a partial shadow."""
    with db_engine.begin() as connection:
        _seed(connection)
    refresh_graph(db_engine, SCOPE, mode="structural")
    with db_engine.begin() as connection:
        params = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}
        connection.execute(text("INSERT INTO source_document (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status) VALUES (:tenant,:project,'source','v2','title','source-sha-v2',2,'final')"), params)
        for passage_id in ("p1", "p2"):
            connection.execute(text("INSERT INTO source_passage (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256) VALUES (:tenant,:project,'source','v2',:passage,:body,:sha)"), {**params, "passage": passage_id, "body": f"body {passage_id}", "sha": f"sha-{passage_id}"})
            connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason) VALUES (:tenant,:project,'source','v2',:passage,'structural_only','none')"), {**params, "passage": passage_id})
        connection.execute(text("UPDATE source_head SET source_version='v2' WHERE tenant_id=:tenant AND project_id=:project AND source_id='source'"), params)
        write_structural_version(connection, ScopedSourceVersion(*SCOPE.as_tuple(), "source", "v2"))

    stop = Event()
    observed: list[tuple[int, int]] = [(2, 1)]
    failures: list[Exception] = []

    def read_loop() -> None:
        try:
            while not stop.is_set():
                with db_engine.begin() as connection:
                    observed.append(graph_counts(connection, SCOPE).as_tuple())
        except Exception as error:  # pragma: no cover - asserted below
            failures.append(error)

    reader = Thread(target=read_loop)
    reader.start()
    try:
        refresh_graph(
            db_engine,
            SCOPE,
            mode="structural",
            failure_injector=lambda point: time.sleep(0.05) if point == "after_vertices" else None,
        )
    finally:
        with db_engine.begin() as connection:
            observed.append(graph_counts(connection, SCOPE).as_tuple())
        stop.set()
        reader.join(timeout=5)

    assert not reader.is_alive() and not failures
    assert set(observed) == {(2, 1), (3, 2)}
