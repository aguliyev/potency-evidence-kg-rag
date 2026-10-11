from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import text

from potency_kg.ingest import ImportRequest, import_source_version, preflight_official_inputs
from potency_kg.contracts import Scope
from potency_kg.graph_store import load_current_projection
from potency_kg.ids import derive_scoped_id


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "configs/casgevy-potency.json"


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _synthetic_descriptor(
    root: Path,
    *,
    source_version: str,
    passage_ids: tuple[str, ...] = ("synthetic-source:p0001",),
    expected_passage_count: int | None = None,
    replacement_from: str | None = None,
) -> Path:
    configs = root / "configs"
    corpus = root / "corpus"
    raw = corpus / "raw"
    configs.mkdir(parents=True)
    raw.mkdir(parents=True)
    raw_bytes = f"synthetic raw source {source_version}".encode()
    (raw / "synthetic.txt").write_bytes(raw_bytes)
    records = []
    for number, passage_id in enumerate(passage_ids, start=1):
        body = f"Synthetic passage {number} for import behavior tests."
        records.append(
            {
                "scope": {"tenant_id": "test-tenant", "project_id": "import-behavior"},
                "source_id": "synthetic-source",
                "source_version": source_version,
                "passage_id": passage_id,
                "text": body,
                "text_sha256": _sha256(body.encode()),
                "locator": {"page": number},
                "subject_scope": {},
                "status": "fictional",
                "authority": "synthetic_fixture",
            }
        )
    records_bytes = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    (corpus / "records.jsonl").write_bytes(records_bytes)
    manifest = {
        "snapshot": f"snapshot-{source_version}",
        "source_count": 1,
        "record_count": len(records),
        "record_sha256": _sha256(records_bytes),
        "sources": [
            {
                "source_id": "synthetic-source",
                "source_version": source_version,
                "title": "Synthetic source",
                "sha256": _sha256(raw_bytes),
                "passage_count": expected_passage_count if expected_passage_count is not None else len(records),
                "status": "fictional",
                "authority": "synthetic_fixture",
                "source_type": "fixture",
                "source_url": None,
                "filename": "corpus/raw/synthetic.txt",
            }
        ],
    }
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    descriptor: dict[str, object] = {
        "scope": {"tenant_id": "test-tenant", "project_id": "import-behavior"},
        "corpus_snapshot": manifest["snapshot"],
        "corpus_manifest": "../corpus/manifest.json",
        "source_records": "../corpus/records.jsonl",
    }
    if replacement_from is not None:
        descriptor["replacements"] = [
            {
                "source_id": "synthetic-source",
                "from_source_version": replacement_from,
                "to_source_version": source_version,
            }
        ]
    path = configs / "project.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def test_official_preflight_reconciles_manifest_records_and_raw_files() -> None:
    """Skipping a supplied source, record, or raw digest must break this test."""
    report = preflight_official_inputs(DESCRIPTOR)
    assert (report.document_count, report.passage_outcome_count) == (24, 791)
    assert sum(report.per_source_counts.values()) == 791
    manifest_path = ROOT / "corpus/manifest.json"
    records_path = ROOT / "corpus/records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report.descriptor_sha256 == _sha256(DESCRIPTOR.read_bytes())
    assert report.manifest_sha256 == _sha256(manifest_path.read_bytes())
    assert report.records_sha256 == _sha256(records_path.read_bytes())
    assert report.per_source_counts == {
        source["source_id"]: source["passage_count"] for source in manifest["sources"]
    }
    assert report.invalid_passage_outcomes == 0
    assert report.valid_passage_outcomes == 791
    assert len(report.raw_sha256) == 24


def test_import_is_idempotent_after_reconnect_and_writes_structural_rows(db_engine) -> None:
    """Dropping conflict handling or structural writing must break this test."""
    first = import_source_version(db_engine, ImportRequest(DESCRIPTOR))
    assert (first.source_count, first.passage_outcome_count) == (24, 791)
    assert (first.imported_passages, first.invalid_passage_outcomes) == (791, 0)
    assert (first.new_source_versions, first.unchanged_source_versions) == (24, 0)
    assert (first.structural_nodes, first.structural_edges) == (815, 791)
    with db_engine.connect() as connection:
        original_heads = connection.execute(
            text("SELECT tenant_id, project_id, source_id, source_version, activated_at FROM source_head ORDER BY tenant_id, project_id, source_id")
        ).all()
    db_engine.dispose()
    from potency_kg.db import create_engine_from_env

    second_engine = create_engine_from_env()
    second = import_source_version(second_engine, ImportRequest(DESCRIPTOR))
    with second_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM source_document")).scalar_one() == 24
        assert connection.execute(text("SELECT count(*) FROM source_passage")).scalar_one() == 791
        assert connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 815
        assert connection.execute(text("SELECT count(*) FROM kg_edge")).scalar_one() == 791
        assert connection.execute(text("SELECT count(*) FROM source_head")).scalar_one() == 24
        reimported_heads = connection.execute(
            text("SELECT tenant_id, project_id, source_id, source_version, activated_at FROM source_head ORDER BY tenant_id, project_id, source_id")
        ).all()
        assert reimported_heads == original_heads
    assert second.new_source_versions == 0
    assert second.unchanged_source_versions == 24
    assert second.imported_passages == 0
    assert (second.structural_nodes, second.structural_edges) == (0, 0)
    second_engine.dispose()


@pytest.mark.parametrize("failure_at", ["after_passages", "after_structural", "before_activation"])
def test_injected_source_failure_rolls_back_all_source_rows(db_engine, failure_at: str) -> None:
    """Moving any injected failure outside the source transaction must break this test."""
    with pytest.raises(RuntimeError):
        import_source_version(db_engine, ImportRequest(DESCRIPTOR, source_id="casgevy-125785-approval-letter", failure_at=failure_at))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM source_document")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM source_passage")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM source_head")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 0


def test_invalid_passage_keeps_document_and_valid_rows_but_never_activates(db_engine, tmp_path: Path) -> None:
    """Rolling back a source document merely because one passage is invalid must break this test."""
    descriptor = _synthetic_descriptor(
        tmp_path / "incomplete",
        source_version="v1",
        passage_ids=("synthetic-source:p0001", ""),
    )
    report = import_source_version(db_engine, ImportRequest(descriptor))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM source_document")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM source_passage")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM import_error")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM source_head")).scalar_one() == 0
        assert connection.execute(text("SELECT reason FROM passage_processing")).scalar_one() == "incomplete_source_version"
        assert connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 2
        assert connection.execute(text("SELECT count(*) FROM kg_edge")).scalar_one() == 1
    assert report.passage_outcome_count == 2
    assert report.imported_passages == 1
    assert report.invalid_passage_outcomes == 1


def test_head_replacement_requires_explicit_descriptor_metadata(db_engine, tmp_path: Path) -> None:
    """Replacing by arrival order or caller convention instead of descriptor metadata must break this test."""
    v1 = _synthetic_descriptor(tmp_path / "v1", source_version="v1")
    v2 = _synthetic_descriptor(tmp_path / "v2", source_version="v2")
    v3 = _synthetic_descriptor(tmp_path / "v3", source_version="v3", replacement_from="v1")
    import_source_version(db_engine, ImportRequest(v1))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT source_version FROM source_head")).scalar_one() == "v1"
        assert connection.execute(text("SELECT reason FROM passage_processing WHERE source_version='v1'")).scalar_one() == "eligible_for_processing"
    import_source_version(db_engine, ImportRequest(v2))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT source_version FROM source_head")).scalar_one() == "v1"
        assert connection.execute(text("SELECT reason FROM passage_processing WHERE source_version='v2'")).scalar_one() == "inactive_source_version"
    import_source_version(db_engine, ImportRequest(v3))
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT source_version FROM source_head")).scalar_one() == "v3"
        reasons = {row.source_version: row.reason for row in connection.execute(text("SELECT source_version, reason FROM passage_processing"))}
        assert reasons == {"v1": "inactive_source_version", "v2": "inactive_source_version", "v3": "eligible_for_processing"}
        scope = Scope("test-tenant", "import-behavior")
        projection = load_current_projection(connection, scope, "structural", allow_partial=False)
        assert {node["node_id"] for node in projection.nodes} == {
            derive_scoped_id("document", scope, "synthetic-source", "v3"),
            derive_scoped_id("passage", scope, "synthetic-source", "v3", "synthetic-source:p0001"),
        }
        assert {edge["edge_id"] for edge in projection.edges} == {
            derive_scoped_id("contains", scope, "synthetic-source", "v3", "synthetic-source:p0001")
        }


def test_explicit_replacement_can_activate_an_already_imported_version(db_engine, tmp_path: Path) -> None:
    """Skipping head reconciliation for unchanged rows must break this test."""
    v1 = _synthetic_descriptor(tmp_path / "v1", source_version="v1")
    v2_without_replacement = _synthetic_descriptor(tmp_path / "v2-inactive", source_version="v2")
    v2_with_replacement = _synthetic_descriptor(
        tmp_path / "v2-declared",
        source_version="v2",
        replacement_from="v1",
    )

    import_source_version(db_engine, ImportRequest(v1))
    import_source_version(db_engine, ImportRequest(v2_without_replacement))
    report = import_source_version(db_engine, ImportRequest(v2_with_replacement))

    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT source_version FROM source_head")).scalar_one() == "v2"
        reasons = {
            row.source_version: row.reason
            for row in connection.execute(
                text("SELECT source_version, reason FROM passage_processing ORDER BY source_version")
            )
        }
    assert reasons == {"v1": "inactive_source_version", "v2": "eligible_for_processing"}
    assert (report.new_source_versions, report.unchanged_source_versions, report.imported_passages) == (0, 1, 0)


def test_replacement_does_not_relabel_incomplete_versions_as_inactive(db_engine, tmp_path: Path) -> None:
    """A broad supersession update that hides incompleteness must break this test."""
    v1 = _synthetic_descriptor(tmp_path / "v1", source_version="v1")
    incomplete_v2 = _synthetic_descriptor(
        tmp_path / "v2-incomplete",
        source_version="v2",
        passage_ids=("synthetic-source:p0001", ""),
    )
    v3 = _synthetic_descriptor(tmp_path / "v3", source_version="v3", replacement_from="v1")

    import_source_version(db_engine, ImportRequest(v1))
    import_source_version(db_engine, ImportRequest(incomplete_v2))
    import_source_version(db_engine, ImportRequest(v3))

    with db_engine.connect() as connection:
        reasons = {
            row.source_version: row.reason
            for row in connection.execute(
                text("SELECT source_version, reason FROM passage_processing ORDER BY source_version")
            )
        }
    assert reasons == {
        "v1": "inactive_source_version",
        "v2": "incomplete_source_version",
        "v3": "eligible_for_processing",
    }


def test_reimport_records_invalid_outcomes_for_the_current_import_run(db_engine, tmp_path: Path) -> None:
    """Returning an invalid outcome without persisting it for that run must break this test."""
    descriptor = _synthetic_descriptor(
        tmp_path / "incomplete",
        source_version="v1",
        passage_ids=("synthetic-source:p0001", ""),
    )

    import_source_version(db_engine, ImportRequest(descriptor))
    report = import_source_version(db_engine, ImportRequest(descriptor))

    with db_engine.connect() as connection:
        run_outcomes = connection.execute(
            text(
                """
                SELECT run_id, count(import_error.error_id) AS error_count
                FROM import_run
                LEFT JOIN import_error USING (tenant_id, project_id, run_id)
                GROUP BY run_id
                ORDER BY run_id
                """
            )
        ).mappings().all()
    assert report.invalid_passage_outcomes == 1
    assert len(run_outcomes) == 2
    assert {row["error_count"] for row in run_outcomes} == {1}


def test_filtered_import_report_contains_only_the_requested_source(db_engine) -> None:
    """Returning all preflight counts for a single-source import must break this test."""
    source_id = "casgevy-125785-approval-letter"
    report = import_source_version(db_engine, ImportRequest(DESCRIPTOR, source_id=source_id))

    assert report.source_count == 1
    assert report.per_source_counts == {source_id: report.passage_outcome_count}


def test_reimport_rejects_changed_immutable_passage_metadata(db_engine, tmp_path: Path) -> None:
    """Comparing only passage IDs and text hashes must break this test."""
    descriptor = _synthetic_descriptor(tmp_path / "immutable", source_version="v1")
    import_source_version(db_engine, ImportRequest(descriptor))

    records_path = descriptor.parent.parent / "corpus/records.jsonl"
    manifest_path = descriptor.parent.parent / "corpus/manifest.json"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["locator"] = {"page": 999}
    records_bytes = (json.dumps(record, sort_keys=True) + "\n").encode()
    records_path.write_bytes(records_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_sha256"] = _sha256(records_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable passage rows conflict"):
        import_source_version(db_engine, ImportRequest(descriptor))
