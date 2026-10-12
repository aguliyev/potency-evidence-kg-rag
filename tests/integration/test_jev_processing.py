from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import pytest
from potency_kg.aliases import approved_alias_catalog
from potency_kg.contracts import Scope
from potency_kg.ids import canonical_json
from potency_kg.jev_client import (
    JevClientIdentity,
    JevResponse,
    JevTransientError,
    NoulQuestion,
)
from potency_kg.jev_processing import classify_active_passages
from potency_kg.processing_policy import BUNDLE_ORDER, JevSelectionPolicy
from sqlalchemy import text

SCOPE = Scope("tenant", "project")


@dataclass
class _JevDouble:
    probabilities: dict[str, float]
    fail: bool = False
    identity: JevClientIdentity = JevClientIdentity(
        provider="deterministic",
        model="jev-test",
        credential_env="TYPESAFE_API_KEY",
        max_attempts=3,
        max_concurrency=2,
    )

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, tuple[NoulQuestion, ...]]] = []

    def classify(self, value: str, questions: Sequence[NoulQuestion]) -> JevResponse:
        self.calls.append((value, tuple(questions)))
        if self.fail:
            raise JevTransientError("secret provider detail", attempts=3)
        return JevResponse(
            model=self.identity.model,
            probabilities=self.probabilities,
            request_id="request-id",
            usage={"input_tokens": 4, "output_tokens": 7},
            attempts=1,
        )


def _insert_embedded_passages(connection, passages: list[tuple[str, str]]) -> None:
    connection.execute(text("INSERT INTO source_document (tenant_id, project_id, source_id, source_version, title, source_sha256, expected_passage_count, status) VALUES ('tenant', 'project', 'source', 'v1', 'title', 'sha', :count, 'final')"), {"count": len(passages)})
    connection.execute(text("INSERT INTO source_head (tenant_id, project_id, source_id, source_version) VALUES ('tenant', 'project', 'source', 'v1')"))
    for index, (passage_id, body) in enumerate(passages):
        run_id = f"00000000-0000-0000-0000-{index + 1:012d}"
        params = {"passage": passage_id, "body": body, "run_id": run_id, "sha": f"sha-{passage_id}"}
        connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES ('tenant', 'project', 'source', 'v1', :passage, :body, :sha)"), params)
        connection.execute(text("INSERT INTO passage_processing (tenant_id, project_id, source_id, source_version, passage_id, state, reason) VALUES ('tenant', 'project', 'source', 'v1', :passage, 'not_run', 'imported')"), params)
        connection.execute(text("INSERT INTO embedding_run (embedding_run_id, tenant_id, project_id, source_id, source_version, passage_id, embedding_version, model, sanitized_origin, credential_env, dimensions, normalization, input_sha256, status) VALUES (:run_id, 'tenant', 'project', 'source', 'v1', :passage, 'embedding-v', 'model', 'https://example.test/v1', 'EMBEDDING_LLM_API_KEY', 3072, 'provider_unit_length_v1', :sha, 'succeeded')"), params)
        connection.execute(text("INSERT INTO passage_embedding (tenant_id, project_id, source_id, source_version, passage_id, embedding_run_id) VALUES ('tenant', 'project', 'source', 'v1', :passage, :run_id)"), params)


def _approve_alias(connection, alias_id: str, alias_value: str) -> None:
    connection.execute(text("""INSERT INTO aliases
        (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,normalization_version,decided_at,decided_by)
        VALUES ('tenant','project',:alias_id,:alias_value,lower(:alias_value),'Product','casgevy','CASGEVY','approved','review','nfkc-casefold-whitespace-v1',now(),'reviewer')"""), {"alias_id": alias_id, "alias_value": alias_value})


def test_classification_redacts_once_per_eligible_passage_and_persists_complete_decision(db_engine) -> None:
    """Calling per question, skipping redaction, or omitting response metadata would break auditability."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "CASGEVY uses (b)(4) testing."), ("p2", "2")])
    client = _JevDouble({name: (0.5 if name == "product_context" else 0.0) for name in BUNDLE_ORDER})

    report = classify_active_passages(db_engine, SCOPE, client, JevSelectionPolicy())

    assert len(client.calls) == 1
    assert client.calls[0][0] == "CASGEVY uses [REDACTED:p1:1] testing."
    assert tuple(question.name for question in client.calls[0][1]) == BUNDLE_ORDER
    assert (report.classified, report.structural_only, report.failed) == (1, 1, 0)
    with db_engine.connect() as connection:
        response = connection.execute(text("SELECT model, request_id, questions, criteria, probabilities, usage, input_sha256, prompt_hash, jev_response_version FROM jev_response")).mappings().one()
        selection = connection.execute(text("SELECT thresholds, selected_bundles, jev_selection_version FROM jev_selection")).mappings().one()
        states = connection.execute(text("SELECT passage_id, state, reason FROM passage_processing ORDER BY passage_id")).all()
    assert response["model"] == "jev-test" and response["request_id"] == "request-id"
    assert set(response["questions"]) == set(BUNDLE_ORDER)
    assert set(response["criteria"]) == set(BUNDLE_ORDER)
    assert set(response["probabilities"]) == set(BUNDLE_ORDER)
    assert response["usage"] == {"input_tokens": 4, "output_tokens": 7}
    expected_prompt_hash = sha256(
        canonical_json(
            {
                name: {
                    "question": response["questions"][name],
                    "criteria": response["criteria"][name],
                }
                for name in BUNDLE_ORDER
            }
        ).encode()
    ).hexdigest()
    assert response["prompt_hash"] == expected_prompt_hash
    assert selection["selected_bundles"] == ["product_context"]
    assert response["jev_response_version"].startswith("jev-response-")
    assert selection["jev_selection_version"].startswith("jev-selection-")
    assert states == [("p1", "selected", "jev_bundle_selected"), ("p2", "structural_only", "page_number_only")]


def test_threshold_only_change_reuses_response_and_recomputes_selection(db_engine) -> None:
    """Binding probabilities to thresholds would spend another Jev request for a policy-only change."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "A substantive passage")])
    client = _JevDouble({name: (0.5 if name == "product_context" else 0.0) for name in BUNDLE_ORDER})

    first = classify_active_passages(db_engine, SCOPE, client, JevSelectionPolicy())
    second = classify_active_passages(db_engine, SCOPE, client, JevSelectionPolicy({"product_context": 0.6}))

    assert len(client.calls) == 1
    assert first.jev_response_version == second.jev_response_version
    assert first.jev_selection_version != second.jev_selection_version
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM jev_response")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM jev_selection")).scalar_one() == 2
        state = connection.execute(
            text(
                """SELECT pp.state,pp.reason,s.jev_selection_version
                   FROM passage_processing AS pp
                   JOIN jev_selection AS s ON s.selection_id=pp.active_jev_selection_id
                    AND s.tenant_id=pp.tenant_id AND s.project_id=pp.project_id
                    AND s.source_id=pp.source_id AND s.source_version=pp.source_version
                    AND s.passage_id=pp.passage_id"""
            )
        ).one()
    assert state == ("structural_only", "jev_no_bundle_above_threshold", second.jev_selection_version)


def test_jev_reuse_is_rejected_after_an_approved_catalog_change(db_engine) -> None:
    """Reusing a response from catalog A after approval creates catalog B would retain stale semantics."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "CASGEVY uses testing.")])
        _approve_alias(connection, "casgevy", "CASGEVY")
        catalog_a = approved_alias_catalog(connection, SCOPE)
    first_client = _JevDouble({name: 0.5 for name in BUNDLE_ORDER})
    classify_active_passages(db_engine, SCOPE, first_client, JevSelectionPolicy())
    with db_engine.begin() as connection:
        _approve_alias(connection, "exa-cel", "exa-cel")
        catalog_b = approved_alias_catalog(connection, SCOPE)
    second_client = _JevDouble({name: 0.5 for name in BUNDLE_ORDER})

    classify_active_passages(db_engine, SCOPE, second_client, JevSelectionPolicy())

    assert catalog_a.version != catalog_b.version
    assert len(first_client.calls) == 1 and len(second_client.calls) == 1
    with db_engine.connect() as connection:
        versions = connection.execute(text("SELECT alias_catalog_version FROM jev_response ORDER BY created_at")).scalars().all()
    assert versions == [catalog_a.version, catalog_b.version]


def test_jev_reuse_requires_the_exact_persisted_catalog_snapshot(db_engine) -> None:
    """A matching digest alone cannot make a corrupted catalog snapshot reusable."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "CASGEVY uses testing.")])
        _approve_alias(connection, "casgevy", "CASGEVY")
    first_client = _JevDouble({name: 0.5 for name in BUNDLE_ORDER})
    classify_active_passages(db_engine, SCOPE, first_client, JevSelectionPolicy())
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE jev_response SET profile_aliases=CAST(:aliases AS jsonb)"),
            {"aliases": json.dumps([])},
        )
    replacement_client = _JevDouble({name: 0.5 for name in BUNDLE_ORDER})

    classify_active_passages(db_engine, SCOPE, replacement_client, JevSelectionPolicy())

    assert len(first_client.calls) == 1
    assert len(replacement_client.calls) == 1


def test_jev_failure_has_no_selection_or_unrestricted_fallback(db_engine) -> None:
    """A failed gate must not silently create extraction eligibility."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "A substantive passage")])
    client = _JevDouble({name: 1.0 for name in BUNDLE_ORDER}, fail=True)

    report = classify_active_passages(db_engine, SCOPE, client, JevSelectionPolicy())

    assert report.failed == 1
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM jev_selection")).scalar_one() == 0
        response = connection.execute(
            text(
                """SELECT status,error_detail,prompt_hash,bundle_hash,usage,attempts
                   FROM jev_response"""
            )
        ).one()
        state = connection.execute(text("SELECT state, reason, active_jev_selection_id FROM passage_processing")).one()
    assert response[0] == "failed" and "secret provider detail" not in (response[1] or "")
    assert all(isinstance(value, str) and value for value in response[2:4])
    assert response[4:] == ({}, 3)
    assert state == ("failed", "jev_failed", None)


def test_process_control_interrupt_is_not_persisted_as_a_jev_failure(db_engine) -> None:
    """Catching process-control signals would falsely report an interrupted run as a model failure."""
    with db_engine.begin() as connection:
        _insert_embedded_passages(connection, [("p1", "A substantive passage")])

    class _InterruptJev(_JevDouble):
        def classify(self, value: str, questions: Sequence[NoulQuestion]) -> JevResponse:
            raise KeyboardInterrupt

    client = _InterruptJev({name: 1.0 for name in BUNDLE_ORDER})

    with pytest.raises(KeyboardInterrupt):
        classify_active_passages(db_engine, SCOPE, client, JevSelectionPolicy())

    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM jev_response")).scalar_one() == 0
        assert connection.execute(text("SELECT state,reason FROM passage_processing")).one() == (
            "not_run",
            "imported",
        )
