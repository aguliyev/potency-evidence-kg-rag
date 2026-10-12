from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
from pydantic import BaseModel
from sqlalchemy import text

from potency_kg.contracts import Scope
from potency_kg.extraction import extract_selected_passages
from potency_kg.llm_client import GenerationClientIdentity


SCOPE = Scope("tenant", "project")


@dataclass
class _ExtractionDouble:
    payloads: list[dict[str, object]]
    identity: GenerationClientIdentity = GenerationClientIdentity(
        provider="deterministic",
        model="generation-test",
        sanitized_origin="https://models.example.test/v1",
        credential_env="LLM_API_KEY",
        instructor_mode="openai",
        max_attempts=3,
    )

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
        self.calls.append((model_text, prompt))
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return response_model.model_validate(payload)

    def usage_for_current_thread(self) -> dict[str, int]:
        return {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def _insert_selected(connection, body: str, bundles: list[str]) -> None:
    response_id, selection_id = uuid4(), uuid4()
    connection.execute(text("INSERT INTO source_document (tenant_id, project_id, source_id, source_version, title, source_sha256, expected_passage_count, status) VALUES ('tenant','project','source','v1','title','sha',1,'final')"))
    connection.execute(text("INSERT INTO source_passage (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256) VALUES ('tenant','project','source','v1','p1',:body,'text-sha')"), {"body": body})
    connection.execute(text("INSERT INTO source_head (tenant_id,project_id,source_id,source_version) VALUES ('tenant','project','source','v1')"))
    connection.execute(text("INSERT INTO jev_response (response_id,tenant_id,project_id,source_id,source_version,passage_id,jev_response_version,model,questions,probabilities,input_sha256,profile_version,status) VALUES (:response,'tenant','project','source','v1','p1','jv','jev','{}','{}','hash','profile','succeeded')"), {"response": response_id})
    connection.execute(text("INSERT INTO jev_selection (selection_id,response_id,tenant_id,project_id,source_id,source_version,passage_id,thresholds,selected_bundles,jev_selection_version) VALUES (:selection,:response,'tenant','project','source','v1','p1','{}',CAST(:bundles AS jsonb),'sv')"), {"selection": selection_id, "response": response_id, "bundles": __import__('json').dumps(bundles)})
    connection.execute(text("INSERT INTO passage_processing (tenant_id,project_id,source_id,source_version,passage_id,state,reason,active_jev_selection_id) VALUES ('tenant','project','source','v1','p1','selected','jev_bundle_selected',:selection)"), {"selection": selection_id})


def _valid_payload() -> dict[str, object]:
    return {
        "entities": [{"type": "Product", "local_id": "e1", "name": "CASGEVY", "supporting_quote": "CASGEVY"}],
        "assertions": [{"local_id": "a1", "claim": "CASGEVY was reviewed.", "supporting_quote": "CASGEVY was reviewed"}],
        "relationships": [{"local_id": "r1", "relation_type": "applies_to", "source_local_id": "a1", "target_local_id": "e1", "assertion_local_id": "a1"}],
        "source_refs": [],
    }


def test_extraction_persists_one_complete_grounded_run_with_sanitized_identity(db_engine) -> None:
    """Partial or ungrounded extraction persistence would make later materialization unsafe."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed after (b)(4).", ["product_context"])
    client = _ExtractionDouble([_valid_payload()])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=2)

    assert (report.succeeded, report.failed) == (1, 0)
    assert client.calls[0][0] == "CASGEVY was reviewed after [REDACTED:p1:1]."
    with db_engine.connect() as connection:
        row = connection.execute(text("SELECT model,sanitized_origin,credential_env,instructor_mode,selected_bundles,prompt_hash,schema_hash,profile_version,input_sha256,usage,status,response,extraction_version FROM llm_extraction_run")).mappings().one()
        state = connection.execute(text("SELECT state,reason,active_extraction_run_id FROM passage_processing")).one()
    assert (row["model"], row["sanitized_origin"], row["credential_env"], row["instructor_mode"]) == ("generation-test", "https://models.example.test/v1", "LLM_API_KEY", "openai")
    assert row["selected_bundles"] == ["product_context"]
    assert row["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert row["response"]["assertions"][0]["original_quote"] == "CASGEVY was reviewed"
    assert row["status"] == "succeeded" and row["extraction_version"].startswith("extraction-")
    assert state == ("selected", "extraction_validated", None)


def test_validated_extraction_is_reused_after_pre_materialization_restart(db_engine) -> None:
    """Restarting after validation must not repeat a paid request for identical extraction inputs."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])
    first_client = _ExtractionDouble([_valid_payload()])
    first = extract_selected_passages(db_engine, SCOPE, first_client, max_concurrency=1)
    second_client = _ExtractionDouble([_valid_payload()])

    second = extract_selected_passages(db_engine, SCOPE, second_client, max_concurrency=1)

    assert len(first_client.calls) == 1
    assert second_client.calls == []
    assert second.extraction_run_ids == first.extraction_run_ids
    assert (second.succeeded, second.failed) == (1, 0)
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM llm_extraction_run")).scalar_one() == 1


def test_extraction_persists_adapter_level_attempt_count(db_engine) -> None:
    """The immutable run must include transport attempts hidden inside the adapter boundary."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])

    class _AttemptReportingDouble(_ExtractionDouble):
        def attempts_for_current_thread(self) -> int:
            return 3

    client = _AttemptReportingDouble([_valid_payload()])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.succeeded == 1
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT attempts FROM llm_extraction_run")).scalar_one() == 3


def test_unselected_type_gets_one_emphasized_retry_then_complete_replacement(db_engine) -> None:
    """Accepting an unselected type or retrying more than once would bypass the scoped schema contract."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])
    invalid = {**_valid_payload(), "entities": [{"type": "Assay", "local_id": "e1", "name": "assay", "supporting_quote": "CASGEVY"}]}
    client = _ExtractionDouble([invalid, _valid_payload()])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.succeeded == 1 and len(client.calls) == 2
    assert "only" in client.calls[1][1].lower() and "Product" in client.calls[1][1]
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM llm_extraction_run")).scalar_one() == 1


def test_second_unselected_response_records_one_atomic_failure_without_drafts(db_engine) -> None:
    """A second invalid complete response must leave no partial mention, assertion, or relation rows."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])
    invalid = {**_valid_payload(), "entities": [{"type": "Assay", "local_id": "e1", "name": "assay", "supporting_quote": "CASGEVY"}]}
    client = _ExtractionDouble([invalid, invalid])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.failed == 1 and len(client.calls) == 2
    with db_engine.connect() as connection:
        run = connection.execute(text("SELECT status,error_detail,response FROM llm_extraction_run")).one()
        counts = tuple(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() for table in ("entity_mention", "assertion", "alias_candidate"))
        failure_count = connection.execute(text("SELECT count(*) FROM validation_failure")).scalar_one()
        state = connection.execute(text("SELECT state,reason FROM passage_processing")).one()
    assert run[0] == "failed" and run[2] is None
    assert counts == (0, 0, 0) and failure_count == 1
    assert state == ("failed", "extraction_failed")


def test_post_response_validation_failure_counts_request_usage_once(db_engine) -> None:
    """A local grounding failure must not double-count the one provider response already received."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])
    invalid_quote = {
        **_valid_payload(),
        "entities": [
            {
                "type": "Product",
                "local_id": "e1",
                "name": "CASGEVY",
                "supporting_quote": "not in the passage",
            }
        ],
    }
    client = _ExtractionDouble([invalid_quote])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.failed == 1 and len(client.calls) == 1
    with db_engine.connect() as connection:
        usage = connection.execute(text("SELECT usage FROM llm_extraction_run")).scalar_one()
    assert usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_unselected_type_retry_requires_the_typed_parseable_response_signal(db_engine) -> None:
    """Provider failures merely mentioning an unselected type must not receive the schema retry."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])

    class _RuntimeFailureDouble:
        identity = _ExtractionDouble.identity

        def __init__(self) -> None:
            self.calls = 0

        def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
            self.calls += 1
            raise RuntimeError("provider failed near an unselected entity type")

    client = _RuntimeFailureDouble()

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.failed == 1
    assert client.calls == 1
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT attempts FROM llm_extraction_run")).scalar_one() == 1


def test_process_control_interrupt_is_not_persisted_as_an_extraction_failure(db_engine) -> None:
    """An interrupted worker must roll control back to the caller instead of creating failed model history."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "CASGEVY was reviewed.", ["product_context"])

    class _InterruptExtraction:
        identity = _ExtractionDouble.identity

        def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        extract_selected_passages(db_engine, SCOPE, _InterruptExtraction(), max_concurrency=1)

    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM llm_extraction_run")).scalar_one() == 0
        assert connection.execute(text("SELECT state,reason FROM passage_processing")).one() == (
            "selected",
            "jev_bundle_selected",
        )


def test_no_selected_bundle_never_schedules_instructor(db_engine) -> None:
    """A Jev no-selection result must remain a hard gate rather than unrestricted extraction fallback."""
    with db_engine.begin() as connection:
        _insert_selected(connection, "A structural passage.", [])
        connection.execute(
            text(
                """UPDATE passage_processing
                   SET state='structural_only',reason='jev_no_bundle_above_threshold',
                       active_jev_selection_id=NULL"""
            )
        )
    client = _ExtractionDouble([_valid_payload()])

    report = extract_selected_passages(db_engine, SCOPE, client, max_concurrency=1)

    assert report.eligible_passages == 0
    assert report.extraction_run_ids == ()
    assert client.calls == []
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM llm_extraction_run")).scalar_one() == 0
