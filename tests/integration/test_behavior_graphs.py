from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from sqlalchemy import text

from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.fixture_catalog import load_fixture_catalog
from potency_kg.fixture_runs import ModelClients, run_kg_fixtures
from potency_kg.jev_client import JevClientIdentity, JevResponse, NoulQuestion
from potency_kg.llm_client import GenerationClientIdentity
from potency_kg.processing_policy import BUNDLE_ORDER


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = 3072


@dataclass
class _EmbeddingDouble:
    identity: EmbeddingClientIdentity = EmbeddingClientIdentity(
        provider="test-double", model="fixture-embedding", dimensions=DIMENSIONS,
        encoding_format="float", normalization="provider_unit_length_v1",
        sanitized_origin="https://test.invalid/embedding", credential_env="TEST_EMBEDDING_KEY",
    )

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch(self.identity.model, tuple((1.0,) * DIMENSIONS for _ in texts), DIMENSIONS, "float", "provider_unit_length_v1", len(texts), len(texts))


@dataclass
class _JevDouble:
    identity: JevClientIdentity = JevClientIdentity(
        provider="test-double", model="fixture-jev", credential_env="TEST_JEV_KEY",
        max_attempts=1, max_concurrency=1,
    )

    def classify(self, text: str, questions: Sequence[NoulQuestion]) -> JevResponse:
        return JevResponse(self.identity.model, {name: 1.0 for name in BUNDLE_ORDER})


@dataclass
class _ExtractionDouble:
    identity: GenerationClientIdentity = GenerationClientIdentity(
        provider="test-double", model="fixture-extraction", sanitized_origin="https://test.invalid/extraction",
        credential_env="TEST_EXTRACTION_KEY", instructor_mode="test-double", max_attempts=1,
    )

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
        sentences = tuple(part.strip().rstrip(".!?") for part in model_text.split(". ") if part.strip())
        claim = max(sentences, key=len)
        return response_model.model_validate({
            "entities": [],
            "assertions": [{"local_id": "assertion", "claim": claim, "supporting_quote": claim}],
            "relationships": [],
            "source_refs": [],
        })


def test_fixture_orchestration_uses_one_database_and_preserves_declared_history(db_engine, tmp_path) -> None:
    """Fixture scenarios must use the ordinary pipeline without cross-scope leakage or inferred replacement."""
    catalog = load_fixture_catalog(PACKAGE_ROOT)
    clients = ModelClients(_EmbeddingDouble(), _JevDouble(), _ExtractionDouble())
    before = catalog.protected_hashes

    report = run_kg_fixtures(db_engine, catalog, tmp_path / "results" / "kg" / "fixtures", clients)

    assert catalog.protected_hashes == before
    assert len(report.runs) == 5
    assert all(item.output_dir.is_relative_to(tmp_path / "results" / "kg" / "fixtures") for item in report.runs)
    assert all(item.export.version_dir.is_relative_to(item.output_dir) for item in report.runs)
    assert report.reconnect_verified is True
    by_run = {(item.configuration.scenario_id, item.configuration.stage): item for item in report.runs}
    assert {ref.text for ref in by_run[("isolation-project-one", "combined")].retrieval.source_refs} == {
        "PROJECT_ONE_ONLY: The fictional assay measures attribute alpha."
    }
    assert {ref.text for ref in by_run[("isolation-project-two", "combined")].retrieval.source_refs} == {
        "PROJECT_TWO_ONLY: The fictional assay measures attribute beta."
    }
    with db_engine.connect() as connection:
        isolation = connection.execute(text("SELECT project_id, text FROM source_passage WHERE tenant_id='fixture-tenant' AND source_id='shared-document' ORDER BY project_id")).all()
        correction = connection.execute(text("SELECT source_version FROM source_head WHERE tenant_id='fixture-tenant' AND project_id='source-correction' AND source_id='assay-attribute-note'")).scalar_one()
        correction_versions = connection.execute(text("SELECT source_version FROM source_document WHERE tenant_id='fixture-tenant' AND project_id='source-correction' AND source_id='assay-attribute-note' ORDER BY source_version")).scalars().all()
        correction_current = connection.execute(text("SELECT source_version FROM passage_processing WHERE tenant_id='fixture-tenant' AND project_id='source-correction' AND source_id='assay-attribute-note' AND state='extracted' ORDER BY source_version")).scalars().all()
        conflict = connection.execute(text("SELECT source_id FROM source_head WHERE tenant_id='fixture-tenant' AND project_id='unresolved-conflict' ORDER BY source_id")).scalars().all()
        conflict_assertions = connection.execute(text("SELECT count(*) FROM assertion WHERE tenant_id='fixture-tenant' AND project_id='unresolved-conflict'")).scalar_one()
        conflict_claims = connection.execute(text("SELECT claim FROM assertion WHERE tenant_id='fixture-tenant' AND project_id='unresolved-conflict' ORDER BY claim")).scalars().all()
        contradictions = connection.execute(text("SELECT count(*) FROM kg_edge WHERE tenant_id='fixture-tenant' AND project_id='unresolved-conflict' AND edge_type='contradicts'")).scalar_one()

    assert isolation == [("project-one", "PROJECT_ONE_ONLY: The fictional assay measures attribute alpha."), ("project-two", "PROJECT_TWO_ONLY: The fictional assay measures attribute beta.")]
    assert correction == "fixture-v2" and correction_versions == ["fixture-v1", "fixture-v2"]
    assert correction_current == ["fixture-v2"]
    assert by_run[("source-correction", "before")].export.version_dir.is_dir()
    assert conflict == ["plan-k-note-a", "plan-k-note-b"] and conflict_assertions == 2 and contradictions == 0
    assert any("assay Vega is used as a release assay" in claim for claim in conflict_claims)
    assert any("assay Vega is not used as a release assay" in claim for claim in conflict_claims)
