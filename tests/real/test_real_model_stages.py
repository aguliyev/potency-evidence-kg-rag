from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from potency_kg.cli import MODEL_GATE, real_model_readiness
from potency_kg.embedding_client import OpenAIEmbeddingClient
from potency_kg.jev_client import TypeSafeJevClient, fixed_noul_questions
from potency_kg.llm_client import InstructorExtractionClient
from potency_kg.processing_policy import BUNDLE_ORDER
from potency_kg.settings import load_embedding_settings, load_generation_settings
from potency_kg.workflow_stages import discover_aliases_llm_stage


pytestmark = pytest.mark.real_models


def _require_real_models() -> dict[str, bool]:
    readiness = real_model_readiness(os.environ)
    if os.environ.get(MODEL_GATE) != "true":
        pytest.skip("not_run: KG_RUN_REAL_MODELS is not true")
    if not readiness["ready"]:
        pytest.skip("not_run: one or more required model variables are missing or invalid")
    return readiness


def test_real_model_gate_requires_explicit_opt_in_and_all_stage_credentials() -> None:
    """Treating partial credentials or an implicit run as production-ready must break this test."""
    readiness = _require_real_models()
    assert readiness == {
        "enabled": True,
        "embedding_present": True,
        "generation_present": True,
        "jev_present": True,
        "ready": True,
    }


def test_real_embedding_adapter_returns_official_full_dimension_float_vector() -> None:
    """A credential-presence check without one validated provider response must not claim access works."""
    _require_real_models()
    settings = load_embedding_settings(os.environ)
    client = OpenAIEmbeddingClient(settings, SecretStr(os.environ[settings.credential_env]))

    batch = client.embed(("Access check: a fictional assay measures a fictional attribute.",))

    assert client.identity.credential_env == "EMBEDDING_LLM_API_KEY"
    assert client.identity.sanitized_origin == "https://api.openai.com/v1"
    assert batch.model == "text-embedding-3-large"
    assert batch.dimensions == 3072 and batch.encoding_format == "float"
    assert len(batch.vectors) == 1 and len(batch.vectors[0]) == 3072
    assert batch.total_tokens >= batch.prompt_tokens > 0


def test_real_jev_adapter_returns_all_seven_probabilities() -> None:
    """A partial or malformed Jev response must not satisfy the real-adapter smoke gate."""
    _require_real_models()
    client = TypeSafeJevClient.from_environment(os.environ)

    response = client.classify(
        "SYNTHETIC ACCESS CHECK. A fictional assay measures a fictional potency attribute.",
        fixed_noul_questions(),
    )

    assert tuple(response.probabilities) == BUNDLE_ORDER
    assert all(0.0 <= value <= 1.0 for value in response.probabilities.values())
    assert response.model and client.identity.credential_env == "TYPESAFE_API_KEY"
    assert 1 <= response.attempts <= client.identity.max_attempts


class _AccessAcknowledgement(BaseModel):
    acknowledged: bool


def test_real_generation_adapter_returns_schema_valid_acknowledgement() -> None:
    """Authentication alone must not pass when structured generation is unusable."""
    _require_real_models()
    settings = load_generation_settings(os.environ)
    client = InstructorExtractionClient.from_environment(os.environ)

    response = client.extract(
        "Access check: a fictional assay measures a fictional attribute.",
        _AccessAcknowledgement,
        "Return acknowledged=true for this synthetic access check.",
    )

    assert response.acknowledged is True
    assert client.identity.model == settings.model
    assert client.identity.credential_env == "LLM_API_KEY"
    assert client.identity.sanitized_origin == settings.sanitized_origin
    assert 1 <= client.attempts_for_current_thread() <= client.identity.max_attempts


def test_real_llm_alias_discovery_is_explicitly_gated_and_reports_provider_evidence() -> None:
    """The optional discovery provider check is skipped as not_run unless the existing real-model gate is complete."""
    _require_real_models()

    config = Path(__file__).resolve().parents[2] / "configs" / "casgevy-potency.json"
    report = discover_aliases_llm_stage(config, enabled=True, env=os.environ)

    # An empty structural database must be reported as not_run.  Otherwise this
    # real-provider test could pass despite making no provider request at all.
    if report.documents == 0:
        pytest.skip("not_run: no active source documents have been imported for LLM alias discovery")
    assert report.method == "llm"
    assert report.documents == report.succeeded + report.failed + report.reused
    assert report.succeeded > 0
    assert report.failed == 0
    assert all(str(run_id) for run_id in report.discovery_run_ids)
