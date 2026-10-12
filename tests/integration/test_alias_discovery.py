from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable

import pytest
from pydantic import BaseModel
from sqlalchemy import text

import potency_kg.alias_discovery as alias_discovery
from potency_kg.alias_discovery import (
    DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY,
    MAX_ALIAS_DISCOVERY_CHARACTERS,
    discover_aliases_deterministic,
    discover_aliases_llm,
)
from potency_kg.aliases import AliasDecision, decide_alias
from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.ingest import ImportRequest, import_source_version
from potency_kg.llm_client import GenerationClientIdentity, GenerationTransientError


SCOPE = Scope("discovery-tenant", "discovery-project")


@dataclass
class _AliasDiscoveryDouble:
    """Complete in-process provider boundary; all database behavior remains real."""

    responder: Callable[[str], dict[str, object]]
    attempts: int = 1
    identity: GenerationClientIdentity = GenerationClientIdentity(
        provider="deterministic",
        model="alias-discovery-test",
        sanitized_origin="https://models.example.test/v1",
        credential_env="LLM_API_KEY",
        instructor_mode="openai",
        max_attempts=3,
    )

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, type[BaseModel], str]] = []

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
        self.calls.append((model_text, response_model, prompt))
        return response_model.model_validate(self.responder(model_text))

    def usage_for_current_thread(self) -> dict[str, int]:
        return {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    def attempts_for_current_thread(self) -> int:
        return self.attempts


def _insert_llm_document(connection, source_id: str, passages: list[tuple[str, str]]) -> None:
    connection.execute(text(
        """INSERT INTO source_document
           (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status)
           VALUES (:tenant,:project,:source,'v1',:source,:sha,:count,'final')"""
    ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "source": source_id,
        "sha": f"sha-{source_id}", "count": len(passages)})
    connection.execute(text(
        """INSERT INTO source_head (tenant_id,project_id,source_id,source_version)
           VALUES (:tenant,:project,:source,'v1')"""
    ), {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "source": source_id})
    for passage_id, body in passages:
        values = {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "source": source_id,
                  "passage": passage_id, "body": body, "sha": _digest(body.encode())}
        connection.execute(text(
            """INSERT INTO source_passage
               (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
               VALUES (:tenant,:project,:source,'v1',:passage,:body,:sha)"""
        ), values)
        connection.execute(text(
            """INSERT INTO passage_processing
               (tenant_id,project_id,source_id,source_version,passage_id,state,reason)
               VALUES (:tenant,:project,:source,'v1',:passage,'structural_only','eligible_for_processing')"""
        ), values)


def _valid_alias(passage_id: str, quote: str, **changes: object) -> dict[str, object]:
    item: dict[str, object] = {
        "alias_value": "exa-cel",
        "canonical_name": "Exagamglogene autotemcel",
        "entity_type": "Product",
        "passage_id": passage_id,
        "supporting_quote": quote,
        "occurrence_index": 0,
        "confidence": 0.91,
    }
    item.update(changes)
    return item


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


def test_llm_discovery_disabled_boundary_skips_database_and_provider(db_engine) -> None:
    """Opening a run or calling the provider while disabled would violate the explicit cost and credential gate."""
    client = _AliasDiscoveryDouble(lambda _text: {"aliases": []})

    report = discover_aliases_llm(db_engine, SCOPE, client, enabled=False)

    assert DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY is False
    assert report.documents == 0 and report.discovery_run_ids == ()
    assert client.calls == []
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM alias_discovery_run")).scalar_one() == 0


def test_llm_discovery_persists_pending_grounded_aliases_and_reuses_successes(db_engine) -> None:
    """Persisting an approved alias, losing offsets, or repaying an identical successful run must break this test."""
    quote = "Exagamglogene autotemcel (exa-cel)"
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", [("p1", f"{quote} was evaluated.")])
    client = _AliasDiscoveryDouble(lambda _text: {"aliases": [_valid_alias("p1", quote)]})

    first = discover_aliases_llm(db_engine, SCOPE, client, enabled=True)
    repeated = discover_aliases_llm(db_engine, SCOPE, _AliasDiscoveryDouble(lambda _text: {"aliases": []}), enabled=True)

    assert (first.method, first.documents, first.succeeded, first.failed, first.reused, first.proposals, first.evidence_inserted) == (
        "llm", 1, 1, 0, 0, 1, 1
    )
    assert len(client.calls) == 1
    assert (repeated.succeeded, repeated.failed, repeated.reused, repeated.proposals, repeated.evidence_inserted) == (0, 0, 1, 0, 0)
    assert repeated.discovery_run_ids == first.discovery_run_ids
    with db_engine.connect() as connection:
        alias = connection.execute(text("SELECT status,managed_by,entity_type,canonical_name FROM aliases")).one()
        evidence = connection.execute(text(
            "SELECT source_type,passage_id,exact_quote,original_start,original_end FROM alias_evidence"
        )).one()
    assert alias == ("pending", "review", "Product", "Exagamglogene autotemcel")
    assert evidence == ("llm_mined", "p1", quote, 0, len(quote))


def test_llm_discovery_splits_large_documents_into_deterministic_overlapping_passage_windows(db_engine, monkeypatch) -> None:
    """Skipping the overlap or changing window order can lose aliases that sit at a passage boundary."""
    monkeypatch.setattr(alias_discovery, "MAX_ALIAS_DISCOVERY_CHARACTERS", 45)
    passages = [("p1", "first text is short."), ("p2", "second text is short."), ("p3", "third text is short.")]
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", passages)
    client = _AliasDiscoveryDouble(lambda _text: {"aliases": []})

    report = discover_aliases_llm(db_engine, SCOPE, client, enabled=True)

    assert MAX_ALIAS_DISCOVERY_CHARACTERS > 40
    assert (report.succeeded, report.failed) == (1, 0)
    rendered_windows = [model_text for model_text, _model, _prompt in client.calls]
    assert len(rendered_windows) == 2
    assert "first text is short." in rendered_windows[0]
    assert "second text is short." in rendered_windows[0]
    assert "second text is short." in rendered_windows[1]
    assert "third text is short." in rendered_windows[1]


def test_llm_discovery_drops_an_unfit_overlap_and_aggregates_each_window_usage(db_engine, monkeypatch) -> None:
    """Overlap must not make a bounded multi-passage request oversized, and costs cover every request."""
    monkeypatch.setattr(alias_discovery, "MAX_ALIAS_DISCOVERY_CHARACTERS", 30)
    passages = [("p1", "a" * 20), ("p2", "b" * 20)]
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", passages)
    client = _AliasDiscoveryDouble(lambda _text: {"aliases": []}, attempts=2)

    report = discover_aliases_llm(db_engine, SCOPE, client, enabled=True)

    assert (report.succeeded, report.failed) == (1, 0)
    assert len(client.calls) == 2
    assert "b" * 20 not in client.calls[0][0]
    assert "a" * 20 not in client.calls[1][0]
    with db_engine.connect() as connection:
        usage, attempts = connection.execute(text(
            "SELECT usage,attempts FROM alias_discovery_run WHERE source_id='source-a'"
        )).one()
    assert usage == {"completion_tokens": 4, "prompt_tokens": 10, "total_tokens": 14}
    assert attempts == 4


def test_llm_discovery_rejects_a_quote_grounded_to_another_window_before_persisting(db_engine, monkeypatch) -> None:
    """A provider cannot use a document-level passage ID to smuggle evidence into a different request window."""
    monkeypatch.setattr(alias_discovery, "MAX_ALIAS_DISCOVERY_CHARACTERS", 30)
    quote = "Exagamglogene autotemcel (exa-cel)"
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", [("p1", quote), ("p2", "b" * 20)])

    def respond(model_text: str) -> dict[str, object]:
        return {"aliases": []} if quote in model_text else {"aliases": [_valid_alias("p1", quote)]}

    report = discover_aliases_llm(db_engine, SCOPE, _AliasDiscoveryDouble(respond), enabled=True)

    assert (report.succeeded, report.failed, report.proposals, report.evidence_inserted) == (0, 1, 0, 0)
    with db_engine.connect() as connection:
        counts = tuple(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() for table in ("aliases", "alias_evidence"))
        run = connection.execute(text("SELECT status,error_detail FROM alias_discovery_run")).one()
    assert counts == (0, 0)
    assert run == ("failed", "alias discovery failed")


@pytest.mark.parametrize(
    "item",
    [
        _valid_alias("p1", "Exagamglogene autotemcel (exa-cel)", entity_type="Application"),
        _valid_alias("missing", "Exagamglogene autotemcel (exa-cel)"),
        _valid_alias("p1", "not an original quote"),
        _valid_alias("p1", "exa-cel", occurrence_index=None),
    ],
    ids=("invalid-entity-type", "unknown-passage", "quote-mismatch", "ambiguous-quote"),
)
def test_llm_discovery_rejects_invalid_or_ungrounded_complete_responses_atomically(db_engine, item: dict[str, object]) -> None:
    """Persisting any proposal before complete response grounding would leave review state partly trusted."""
    body = "Exagamglogene autotemcel (exa-cel); exa-cel remains under review."
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", [("p1", body)])
    client = _AliasDiscoveryDouble(lambda _text: {"aliases": [item]})

    report = discover_aliases_llm(db_engine, SCOPE, client, enabled=True)

    assert (report.succeeded, report.failed, report.proposals, report.evidence_inserted) == (0, 1, 0, 0)
    with db_engine.connect() as connection:
        counts = tuple(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() for table in ("aliases", "alias_evidence"))
        run = connection.execute(text("SELECT status,error_detail FROM alias_discovery_run")).one()
    assert counts == (0, 0)
    assert run == ("failed", "alias discovery failed")


def test_llm_discovery_records_sanitized_transient_exhaustion_without_losing_other_documents(db_engine, capsys) -> None:
    """Leaking provider detail, rolling back a prior success, or aborting later documents would make failures unsafe."""
    quote = "Exagamglogene autotemcel (exa-cel)"
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", [("p1", f"{quote} was evaluated.")])
        _insert_llm_document(connection, "source-b", [("p1", f"{quote} was evaluated.")])

    calls = 0

    def respond(model_text: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"aliases": [_valid_alias("p1", quote)]}
        raise GenerationTransientError("provider failed with TEST_SECRET_SENTINEL")

    report = discover_aliases_llm(db_engine, SCOPE, _AliasDiscoveryDouble(respond, attempts=3), enabled=True)

    captured = capsys.readouterr()
    assert (report.documents, report.succeeded, report.failed, report.proposals, report.evidence_inserted) == (2, 1, 1, 1, 1)
    assert "TEST_SECRET_SENTINEL" not in repr(report)
    assert "TEST_SECRET_SENTINEL" not in captured.out + captured.err
    with db_engine.connect() as connection:
        runs = connection.execute(text("SELECT source_id,status,error_detail,attempts FROM alias_discovery_run ORDER BY source_id")).all()
        evidence_sources = connection.execute(text("SELECT source_id FROM alias_evidence ORDER BY source_id")).scalars().all()
    assert runs == [("source-a", "succeeded", None, 3), ("source-b", "failed", "alias discovery failed", 3)]
    assert evidence_sources == ["source-a"]


def test_llm_discovery_sanitizes_a_provider_exception_without_partial_commit(db_engine, capsys) -> None:
    """Raw provider diagnostics must not become persisted history or leave a partially proposed alias behind."""
    quote = "Exagamglogene autotemcel (exa-cel)"
    with db_engine.begin() as connection:
        _insert_llm_document(connection, "source-a", [("p1", f"{quote} was evaluated.")])

    def fail(_model_text: str) -> dict[str, object]:
        raise RuntimeError("provider exception carrying TEST_SECRET_SENTINEL")

    report = discover_aliases_llm(db_engine, SCOPE, _AliasDiscoveryDouble(fail), enabled=True)

    captured = capsys.readouterr()
    assert (report.succeeded, report.failed, report.proposals, report.evidence_inserted) == (0, 1, 0, 0)
    assert "TEST_SECRET_SENTINEL" not in repr(report)
    assert "TEST_SECRET_SENTINEL" not in captured.out + captured.err
    with db_engine.connect() as connection:
        counts = tuple(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() for table in ("aliases", "alias_evidence"))
        run = connection.execute(text("SELECT status,error_detail FROM alias_discovery_run")).one()
    assert counts == (0, 0)
    assert run == ("failed", "alias discovery failed")
