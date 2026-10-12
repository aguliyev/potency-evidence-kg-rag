from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from potency_kg.aliases import (
    AliasDecision,
    alias_catalog_status,
    approved_alias_catalog,
    decide_alias,
    list_aliases,
    load_alias_seeds,
)
from potency_kg.contracts import Scope


SCOPE = Scope("public-fda", "casgevy-potency")
OTHER_SCOPE = Scope("other-tenant", "other-project")


def _seed(entries: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": "scoped-alias-seeds-v1", "entries": entries}


def _entry(*, aliases: list[str], canonical_key: str = "casgevy", canonical_name: str = "CASGEVY") -> dict[str, object]:
    return {
        "tenant_id": SCOPE.tenant_id,
        "project_id": SCOPE.project_id,
        "entity_type": "Product",
        "canonical_key": canonical_key,
        "canonical_name": canonical_name,
        "aliases": aliases,
    }


def _write_seed(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(_seed(entries)), encoding="utf-8")
    return path


def _insert_review_alias(connection, scope: Scope, alias_id: str, value: str, *, status: str = "pending") -> None:
    values = {
        "tenant": scope.tenant_id,
        "project": scope.project_id,
        "alias_id": alias_id,
        "value": value,
        "normalized": value.casefold(),
        "status": status,
    }
    if status == "pending":
        connection.execute(text(
            """INSERT INTO aliases
               (tenant_id,project_id,alias_id,alias_value,normalized_alias,status,managed_by,normalization_version)
               VALUES (:tenant,:project,:alias_id,:value,:normalized,:status,'review','nfkc-casefold-whitespace-v1')"""
        ), values)
    else:
        connection.execute(text(
            """INSERT INTO aliases
               (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,
                status,managed_by,normalization_version,decided_at,decided_by)
               VALUES (:tenant,:project,:alias_id,:value,:normalized,'Product','casgevy','CASGEVY',
                :status,'review','nfkc-casefold-whitespace-v1',now(),'reviewer')"""
        ), values)


def _insert_semantic_history(connection, catalog_version: str) -> None:
    response_id = uuid4()
    extraction_id = uuid4()
    connection.execute(text(
        """INSERT INTO source_document
           (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status)
           VALUES (:tenant,:project,'source','v1','title','source-sha',1,'final')"""
    ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id})
    connection.execute(text(
        """INSERT INTO source_passage
           (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
           VALUES (:tenant,:project,'source','v1','p1','body','passage-sha')"""
    ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id})
    values = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "response": response_id,
              "extraction": extraction_id, "catalog": catalog_version}
    connection.execute(text(
        """INSERT INTO jev_response
           (response_id,tenant_id,project_id,source_id,source_version,passage_id,jev_response_version,model,
            questions,probabilities,input_sha256,profile_version,alias_catalog_version,status)
           VALUES (:response,:tenant,:project,'source','v1','p1','jev-v1','model','{}','{}','input','profile',:catalog,'succeeded')"""
    ), values)
    connection.execute(text(
        """INSERT INTO llm_extraction_run
           (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,extraction_version,model,
            sanitized_origin,credential_env,selected_bundles,input_sha256,profile_version,alias_catalog_version,status)
           VALUES (:extraction,:tenant,:project,'source','v1','p1','extract-v1','model',
            'https://example.test/v1','LLM_API_KEY','[]','input','profile',:catalog,'succeeded')"""
    ), values)


def test_seed_load_reconciles_idempotently_and_catalog_is_canonical(db_engine, tmp_path: Path) -> None:
    """Unstable ordering, duplicate seed writes, or a catalog containing unapproved rows breaks reuse identity."""
    seed_path = _write_seed(tmp_path / "aliases.json", [_entry(aliases=["exa-cel", "CASGEVY", "exagamglogene autotemcel"])])

    initial = load_alias_seeds(db_engine, SCOPE, seed_path)
    repeat = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        catalog = approved_alias_catalog(connection, SCOPE)
        aliases = list_aliases(connection, SCOPE)

    assert (initial.inserted, initial.unchanged, initial.retired, initial.replaced, initial.evidence_inserted) == (3, 0, 0, 0, 3)
    assert (repeat.inserted, repeat.unchanged, repeat.retired, repeat.replaced, repeat.evidence_inserted) == (0, 3, 0, 0, 0)
    assert [item.normalized_alias for item in catalog.entries] == ["casgevy", "exa-cel", "exagamglogene autotemcel"]
    assert [item.alias_value for item in aliases] == ["CASGEVY", "exa-cel", "exagamglogene autotemcel"]
    assert catalog.version == initial.catalog_version == repeat.catalog_version
    assert tuple(catalog.entries) == catalog.entries


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("alias_id", "approved-2"),
        ("alias_value", "CASGEVY branded"),
        ("normalized_alias", "casgevy branded"),
        ("entity_type", "Assay"),
        ("canonical_key", "casgevy-next"),
        ("canonical_name", "CASGEVY Next"),
        ("normalization_version", "nfkc-casefold-whitespace-v2"),
    ],
)
def test_catalog_version_changes_when_any_approved_mapping_field_changes(db_engine, column: str, replacement: str) -> None:
    """Leaving an approved mapping field out of the catalog digest permits incorrect semantic reuse."""
    with db_engine.begin() as connection:
        connection.execute(text(
            """INSERT INTO aliases
               (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,
                status,managed_by,normalization_version)
               VALUES (:tenant,:project,'approved-1','CASGEVY','casgevy','Product','casgevy','CASGEVY',
                'approved','review','nfkc-casefold-whitespace-v1')"""
        ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id})
    with db_engine.connect() as connection:
        original = approved_alias_catalog(connection, SCOPE).version
    with db_engine.begin() as connection:
        connection.execute(text(f"UPDATE aliases SET {column}=:replacement WHERE tenant_id=:tenant AND project_id=:project"), {
            "replacement": replacement, "tenant": SCOPE.tenant_id, "project": SCOPE.project_id,
        })
    with db_engine.connect() as connection:
        changed = approved_alias_catalog(connection, SCOPE).version

    assert changed != original


def test_invalid_seed_is_rejected_before_any_database_write(db_engine, tmp_path: Path) -> None:
    """Opening the reconciliation transaction before whole-document validation could leave partial static aliases."""
    seed_path = _write_seed(tmp_path / "invalid.json", [_entry(aliases=["CASGEVY"]), {"tenant_id": SCOPE.tenant_id}])

    with pytest.raises(ValueError):
        load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM aliases")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM alias_evidence")).scalar_one() == 0


def test_seed_validation_rejects_invalid_entries_outside_the_requested_scope(db_engine, tmp_path: Path) -> None:
    """A multi-scope document must be fully validated before any scoped reconciliation starts."""
    invalid_other_scope = {
        "tenant_id": OTHER_SCOPE.tenant_id,
        "project_id": OTHER_SCOPE.project_id,
        "entity_type": "Unsupported",
        "canonical_key": "other",
        "canonical_name": "Other",
        "aliases": ["OTHER"],
    }
    seed_path = _write_seed(tmp_path / "invalid-other-scope.json", [_entry(aliases=["CASGEVY"]), invalid_other_scope])

    with pytest.raises(ValueError):
        load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM aliases")).scalar_one() == 0


def test_seed_removal_retires_static_mapping_without_removing_its_evidence(db_engine, tmp_path: Path) -> None:
    """Deleting removed configuration rows would lose the auditable static observation."""
    seed_path = _write_seed(tmp_path / "aliases.json", [_entry(aliases=["CASGEVY", "exa-cel"])])
    load_alias_seeds(db_engine, SCOPE, seed_path)
    _write_seed(seed_path, [_entry(aliases=["CASGEVY"])])

    report = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        retired = connection.execute(text(
            "SELECT status,managed_by,decided_by,decision_note FROM aliases WHERE normalized_alias='exa-cel'"
        )).one()
        evidence = connection.execute(text(
            "SELECT count(*) FROM alias_evidence AS evidence JOIN aliases AS alias "
            "ON (alias.tenant_id,alias.project_id,alias.alias_id)=(evidence.tenant_id,evidence.project_id,evidence.alias_id) "
            "WHERE alias.normalized_alias='exa-cel'"
        )).scalar_one()

    assert (report.inserted, report.unchanged, report.retired, report.replaced) == (0, 1, 1, 0)
    assert retired[0:3] == ("rejected", "static_config", "static_config")
    assert "removed" in retired[3].casefold()
    assert evidence == 1


def test_seed_target_change_retires_old_mapping_and_adds_new_mapping_atomically(db_engine, tmp_path: Path) -> None:
    """Updating a static row in place would rewrite an immutable historical mapping."""
    seed_path = _write_seed(tmp_path / "aliases.json", [_entry(aliases=["CASGEVY"])])
    load_alias_seeds(db_engine, SCOPE, seed_path)
    _write_seed(seed_path, [_entry(aliases=["CASGEVY"], canonical_key="casgevy-next", canonical_name="CASGEVY Next")])

    report = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        rows = connection.execute(text(
            "SELECT canonical_key,canonical_name,status FROM aliases WHERE normalized_alias='casgevy' ORDER BY status,canonical_key"
        )).all()

    assert (report.inserted, report.unchanged, report.retired, report.replaced) == (1, 0, 1, 1)
    assert rows == [("casgevy-next", "CASGEVY Next", "approved"), ("casgevy", "CASGEVY", "rejected")]


def test_seed_reconciliation_keeps_same_text_for_different_entity_types(db_engine, tmp_path: Path) -> None:
    """The approved-uniqueness key is typed, so Product and Assay seeds must not retire each other."""
    assay_entry = {
        **_entry(aliases=["shared short name"]),
        "entity_type": "Assay",
        "canonical_key": "shared-assay",
        "canonical_name": "Shared assay",
    }
    seed_path = _write_seed(tmp_path / "aliases.json", [
        _entry(aliases=["shared short name"]),
        assay_entry,
    ])

    first = load_alias_seeds(db_engine, SCOPE, seed_path)
    second = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        rows = connection.execute(text(
            "SELECT entity_type,status FROM aliases WHERE normalized_alias='shared short name' ORDER BY entity_type"
        )).all()

    assert (first.inserted, first.retired) == (2, 0)
    assert (second.inserted, second.unchanged, second.retired) == (0, 2, 0)
    assert rows == [("Assay", "approved"), ("Product", "approved")]


def test_review_decisions_are_scoped_idempotent_and_preserve_existing_evidence(db_engine) -> None:
    """Allowing static edits, contradictory repeats, or later evidence to overwrite review state breaks auditability."""
    with db_engine.begin() as connection:
        _insert_review_alias(connection, SCOPE, "review-cgv", "CGV-X")
        _insert_review_alias(connection, OTHER_SCOPE, "review-cgv", "CGV-X")
        _insert_review_alias(connection, SCOPE, "review-reject", "EXA-OLD")
        _insert_review_alias(connection, SCOPE, "approved-conflict", "CGV-X")
        connection.execute(text(
            """INSERT INTO alias_evidence
               (tenant_id,project_id,evidence_id,alias_id,source_type,detector_name,detector_version,input_sha256)
               VALUES (:tenant,:project,'first-evidence','review-cgv','deterministic_rule','rule','v1','first')"""
        ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id})

    approved = decide_alias(db_engine, SCOPE, AliasDecision("review-cgv", "approved", "reviewer", "Product", "casgevy", "CASGEVY"))
    repeated = decide_alias(db_engine, SCOPE, AliasDecision("review-cgv", "approved", "reviewer", "Product", "casgevy", "CASGEVY"))
    rejected = decide_alias(db_engine, SCOPE, AliasDecision("review-reject", "rejected", "reviewer", note="not a product"))
    with db_engine.begin() as connection:
        connection.execute(text(
            """INSERT INTO alias_evidence
               (tenant_id,project_id,evidence_id,alias_id,source_type,detector_name,detector_version,input_sha256)
               VALUES (:tenant,:project,'later-evidence','review-cgv','extraction','extractor','v1','later')"""
        ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id})

    assert approved == repeated
    assert (approved.status, approved.entity_type, approved.canonical_key) == ("approved", "Product", "casgevy")
    assert rejected.status == "rejected"
    with db_engine.begin() as connection:
        _insert_review_alias(connection, SCOPE, "review-incomplete", "NO-TARGET")
    with pytest.raises(ValueError):
        decide_alias(db_engine, SCOPE, AliasDecision("review-incomplete", "approved", "reviewer", "Product", "casgevy", None))
    with pytest.raises(ValueError):
        decide_alias(db_engine, SCOPE, AliasDecision("review-cgv", "rejected", "reviewer"))
    with pytest.raises(ValueError):
        decide_alias(db_engine, SCOPE, AliasDecision("review-cgv", "approved", "reviewer", "Product", "other", "Other"))
    with pytest.raises(IntegrityError):
        decide_alias(db_engine, SCOPE, AliasDecision("approved-conflict", "approved", "reviewer", "Product", "casgevy", "CASGEVY"))
    with db_engine.connect() as connection:
        assert [row.status for row in list_aliases(connection, OTHER_SCOPE)] == ["pending"]
        assert connection.execute(text("SELECT count(*) FROM alias_evidence WHERE alias_id='review-cgv' AND tenant_id=:tenant AND project_id=:project"), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id}).scalar_one() == 2


def test_static_aliases_cannot_be_changed_by_review_and_catalog_status_marks_old_semantic_runs_stale(db_engine, tmp_path: Path) -> None:
    """Database review of source-controlled aliases or reuse after a catalog change must be refused."""
    seed_path = _write_seed(tmp_path / "aliases.json", [_entry(aliases=["CASGEVY"])])
    initial = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.begin() as connection:
        _insert_semantic_history(connection, initial.catalog_version)
    with db_engine.connect() as connection:
        current = alias_catalog_status(connection, SCOPE)
        static_id = list_aliases(connection, SCOPE)[0].alias_id

    assert current.catalog_version == initial.catalog_version
    assert current.latest_jev_catalog_versions == (initial.catalog_version,)
    assert current.latest_extraction_catalog_versions == (initial.catalog_version,)
    assert current.semantic_stale is False
    with pytest.raises(ValueError):
        decide_alias(db_engine, SCOPE, AliasDecision(static_id, "rejected", "reviewer"))

    _write_seed(seed_path, [_entry(aliases=["CASGEVY", "exa-cel"])])
    changed = load_alias_seeds(db_engine, SCOPE, seed_path)
    with db_engine.connect() as connection:
        stale = alias_catalog_status(connection, SCOPE)
        catalog = approved_alias_catalog(connection, SCOPE)

    assert changed.catalog_version != initial.catalog_version
    assert stale.semantic_stale is True
    assert [entry.normalized_alias for entry in catalog.entries] == ["casgevy", "exa-cel"]
