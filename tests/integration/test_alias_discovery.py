from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from sqlalchemy import text

import potency_kg.alias_discovery as alias_discovery
from potency_kg.alias_discovery import discover_aliases_deterministic
from potency_kg.aliases import AliasDecision, decide_alias
from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.ingest import ImportRequest, import_source_version


SCOPE = Scope("discovery-tenant", "discovery-project")


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _descriptor(
    root: Path,
    versions: dict[str, str],
    *,
    replacements: dict[str, str] | None = None,
) -> Path:
    """Write a complete, two-document corpus fixture with independently derived digests."""
    configs = root / "configs"
    corpus = root / "corpus"
    raw = corpus / "raw"
    configs.mkdir(parents=True)
    raw.mkdir(parents=True)
    records: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for source_id, version in sorted(versions.items()):
        raw_bytes = f"raw {source_id} {version}".encode()
        (raw / f"{source_id}-{version}.txt").write_bytes(raw_bytes)
        body = "Exagamglogene autotemcel (exa-cel) was evaluated."
        passage_id = f"{source_id}:p1"
        records.append(
            {
                "scope": {"tenant_id": SCOPE.tenant_id, "project_id": SCOPE.project_id},
                "source_id": source_id,
                "source_version": version,
                "passage_id": passage_id,
                "text": body,
                "text_sha256": _digest(body.encode()),
                "locator": {"page": 1},
                "subject_scope": {},
                "status": "fictional",
                "authority": "test",
            }
        )
        sources.append(
            {
                "source_id": source_id,
                "source_version": version,
                "title": f"{source_id} title",
                "sha256": _digest(raw_bytes),
                "passage_count": 1,
                "status": "fictional",
                "authority": "test",
                "source_type": "fixture",
                "source_url": None,
                "filename": f"corpus/raw/{source_id}-{version}.txt",
            }
        )
    records_bytes = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    (corpus / "records.jsonl").write_bytes(records_bytes)
    manifest = {
        "snapshot": "alias-discovery-snapshot",
        "source_count": len(sources),
        "record_count": len(records),
        "record_sha256": _digest(records_bytes),
        "sources": sources,
    }
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    descriptor: dict[str, object] = {
        "scope": {"tenant_id": SCOPE.tenant_id, "project_id": SCOPE.project_id},
        "corpus_snapshot": manifest["snapshot"],
        "corpus_manifest": "../corpus/manifest.json",
        "source_records": "../corpus/records.jsonl",
    }
    if replacements:
        descriptor["replacements"] = [
            {"source_id": source_id, "from_source_version": old, "to_source_version": versions[source_id]}
            for source_id, old in sorted(replacements.items())
        ]
    path = configs / "project.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return path


def test_discovery_aggregates_two_documents_reuses_successes_and_reprocesses_only_a_replacement(db_engine, tmp_path: Path) -> None:
    """Duplicating aliases/evidence, re-running successes, or missing a replacement document must break this test."""
    first_descriptor = _descriptor(tmp_path / "v1", {"source-a": "v1", "source-b": "v1"})
    import_source_version(db_engine, ImportRequest(first_descriptor))

    first = discover_aliases_deterministic(db_engine, SCOPE)
    with db_engine.connect() as connection:
        first_alias_id = connection.execute(text("SELECT alias_id FROM aliases")).scalar_one()
        first_evidence = connection.execute(text(
            "SELECT source_id,source_version,passage_id,exact_quote,original_start,original_end "
            "FROM alias_evidence ORDER BY source_id"
        )).all()
        evidence_with_run = connection.execute(text(
            "SELECT count(*) FROM alias_evidence WHERE discovery_run_id IS NOT NULL"
        )).scalar_one()
    assert (first.method, first.documents, first.succeeded, first.failed, first.reused, first.proposals, first.evidence_inserted) == (
        "deterministic", 2, 2, 0, 0, 2, 2
    )
    assert first_evidence == [
        ("source-a", "v1", "source-a:p1", "Exagamglogene autotemcel (exa-cel)", 0, 34),
        ("source-b", "v1", "source-b:p1", "Exagamglogene autotemcel (exa-cel)", 0, 34),
    ]
    assert evidence_with_run == 2

    repeated = discover_aliases_deterministic(db_engine, SCOPE)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM aliases")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM alias_evidence")).scalar_one() == 2
        assert connection.execute(text("SELECT count(*) FROM alias_discovery_run WHERE status='succeeded'")).scalar_one() == 2
    assert (repeated.documents, repeated.succeeded, repeated.failed, repeated.reused, repeated.proposals, repeated.evidence_inserted) == (2, 0, 0, 2, 0, 0)
    assert repeated.discovery_run_ids == first.discovery_run_ids

    replacement_descriptor = _descriptor(
        tmp_path / "v2",
        {"source-a": "v2"},
        replacements={"source-a": "v1"},
    )
    import_source_version(db_engine, ImportRequest(replacement_descriptor))
    replaced = discover_aliases_deterministic(db_engine, SCOPE)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT alias_id FROM aliases")).scalar_one() == first_alias_id
        assert connection.execute(text("SELECT count(*) FROM alias_evidence")).scalar_one() == 3
        assert connection.execute(text("SELECT count(*) FROM alias_discovery_run WHERE status='succeeded'")).scalar_one() == 3
        provenance = connection.execute(text(
            "SELECT source_id,source_version,passage_id,exact_quote,original_start,original_end "
            "FROM alias_evidence WHERE source_id='source-a' ORDER BY source_version"
        )).all()
    assert (replaced.documents, replaced.succeeded, replaced.failed, replaced.reused, replaced.proposals, replaced.evidence_inserted) == (2, 1, 0, 1, 1, 1)
    assert provenance == [
        ("source-a", "v1", "source-a:p1", "Exagamglogene autotemcel (exa-cel)", 0, 34),
        ("source-a", "v2", "source-a:p1", "Exagamglogene autotemcel (exa-cel)", 0, 34),
    ]


def test_discovery_attaches_new_evidence_without_reopening_a_rejected_alias(db_engine, tmp_path: Path) -> None:
    """Replacing a source must not silently erase a reviewer rejection while adding new evidence."""
    v1 = _descriptor(tmp_path / "v1", {"source-a": "v1"})
    import_source_version(db_engine, ImportRequest(v1))
    discover_aliases_deterministic(db_engine, SCOPE)
    with db_engine.connect() as connection:
        alias_id = connection.execute(text("SELECT alias_id FROM aliases")).scalar_one()
    decide_alias(db_engine, SCOPE, AliasDecision(alias_id, "rejected", "reviewer", note="not accepted"))

    v2 = _descriptor(tmp_path / "v2", {"source-a": "v2"}, replacements={"source-a": "v1"})
    import_source_version(db_engine, ImportRequest(v2))
    report = discover_aliases_deterministic(db_engine, SCOPE)
    with db_engine.connect() as connection:
        status = connection.execute(text("SELECT status,decided_by,decision_note FROM aliases WHERE alias_id=:alias_id"), {"alias_id": alias_id}).one()
        evidence_versions = connection.execute(text(
            "SELECT source_version FROM alias_evidence WHERE alias_id=:alias_id ORDER BY source_version"
        ), {"alias_id": alias_id}).scalars().all()
    assert (report.succeeded, report.reused, report.evidence_inserted) == (1, 0, 1)
    assert status == ("rejected", "reviewer", "not accepted")
    assert evidence_versions == ["v1", "v2"]


def test_pending_proposals_with_conflicting_targets_remain_separate_review_items(db_engine, tmp_path: Path) -> None:
    """Collapsing distinct proposed targets would hide a conflict that a reviewer must resolve."""
    descriptor = _descriptor(tmp_path / "document", {"source-a": "v1"})
    import_source_version(db_engine, ImportRequest(descriptor))
    source = ScopedPassageIdentity(SCOPE.tenant_id, SCOPE.project_id, "source-a", "v1", "source-a:p1")
    proposal = alias_discovery.discover_alias_patterns(
        source, "Exagamglogene autotemcel (exa-cel) was evaluated."
    )[0]

    with db_engine.begin() as connection:
        alias_discovery.record_alias_proposal(connection, SCOPE, proposal)
        alias_discovery.record_alias_proposal(
            connection, SCOPE, replace(proposal, canonical_name="Exagamglogene alternative")
        )

    with db_engine.connect() as connection:
        pending_targets = connection.execute(text(
            "SELECT canonical_name FROM aliases WHERE normalized_alias='exa-cel' ORDER BY canonical_name"
        )).scalars().all()
    assert pending_targets == ["Exagamglogene alternative", "Exagamglogene autotemcel"]


def test_discovery_rolls_back_a_failed_document_but_continues_with_later_documents(db_engine, tmp_path: Path, monkeypatch) -> None:
    """Persisting partial evidence from one failed document, or stopping later documents, breaks atomicity."""
    descriptor = _descriptor(tmp_path / "documents", {"source-a": "v1", "source-b": "v1"})
    import_source_version(db_engine, ImportRequest(descriptor))
    real_record = alias_discovery.record_alias_proposal

    def fail_source_a(connection, scope, proposal, *, discovery_run_id=None):
        if proposal.source.source_id == "source-a":
            raise ValueError("unsafe diagnostic containing source-a details")
        return real_record(connection, scope, proposal, discovery_run_id=discovery_run_id)

    monkeypatch.setattr(alias_discovery, "record_alias_proposal", fail_source_a)
    report = discover_aliases_deterministic(db_engine, SCOPE)

    with db_engine.connect() as connection:
        runs = connection.execute(text(
            "SELECT source_id,status FROM alias_discovery_run ORDER BY source_id"
        )).all()
        evidence_sources = connection.execute(text(
            "SELECT source_id FROM alias_evidence ORDER BY source_id"
        )).scalars().all()
    assert (report.documents, report.succeeded, report.failed, report.reused, report.proposals, report.evidence_inserted) == (2, 1, 1, 0, 1, 1)
    assert runs == [("source-a", "failed"), ("source-b", "succeeded")]
    assert evidence_sources == ["source-b"]
