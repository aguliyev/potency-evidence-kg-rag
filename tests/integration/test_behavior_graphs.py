from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from sqlalchemy import text

import potency_kg.fixture_runs as fixture_runs
from potency_kg.age_projection import refresh_graph
from potency_kg.alias_discovery import AliasDiscoveryReport, discover_aliases_deterministic
from potency_kg.alias_discovery import AliasProposal, record_alias_proposal
from potency_kg.aliases import AliasDecision, AliasSeedReport, approved_alias_catalog, decide_alias, load_alias_seeds
from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.embedding_client import EmbeddingBatch, EmbeddingClientIdentity
from potency_kg.fixture_catalog import load_fixture_catalog
from potency_kg.fixture_runs import ModelClients, run_kg_fixtures
from potency_kg.jev_client import JevClientIdentity, JevResponse, NoulQuestion
from potency_kg.llm_client import GenerationClientIdentity
from potency_kg.processing_policy import BUNDLE_ORDER
from potency_kg.ingest import ImportRequest, import_source_version, preflight_official_inputs
from potency_kg.project_profiles import assert_protected_inputs_unchanged, protected_input_hashes


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = 3072
OFFICIAL_SCOPE = Scope("public-fda", "casgevy-potency")
OFFICIAL_DESCRIPTOR = PACKAGE_ROOT / "configs" / "casgevy-potency.json"


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


def test_official_structural_alias_workflow_is_idempotent_on_a_fresh_database(db_engine) -> None:
    """Exercise the complete credential-free workflow twice without mutating supplied inputs."""
    before = dict(protected_input_hashes(PACKAGE_ROOT))
    try:
        preflight = preflight_official_inputs(OFFICIAL_DESCRIPTOR)
        first_import = import_source_version(db_engine, ImportRequest(OFFICIAL_DESCRIPTOR))
        first_seed = load_alias_seeds(db_engine, OFFICIAL_SCOPE)
        first_discovery = discover_aliases_deterministic(db_engine, OFFICIAL_SCOPE)
        first_graph = refresh_graph(db_engine, OFFICIAL_SCOPE, mode="structural")

        second_import = import_source_version(db_engine, ImportRequest(OFFICIAL_DESCRIPTOR))
        second_seed = load_alias_seeds(db_engine, OFFICIAL_SCOPE)
        second_discovery = discover_aliases_deterministic(db_engine, OFFICIAL_SCOPE)
        second_graph = refresh_graph(db_engine, OFFICIAL_SCOPE, mode="structural")

        with db_engine.connect() as connection:
            approved_products = connection.execute(text(
                """SELECT count(*) FROM aliases
                   WHERE tenant_id=:tenant AND project_id=:project
                     AND status='approved' AND entity_type='Product'"""
            ), {"tenant": OFFICIAL_SCOPE.tenant_id, "project": OFFICIAL_SCOPE.project_id}).scalar_one()

        assert (preflight.document_count, preflight.passage_outcome_count) == (24, 791)
        assert (first_import.source_count, first_import.imported_passages) == (24, 791)
        assert (first_import.new_source_versions, first_import.unchanged_source_versions) == (24, 0)
        assert approved_products == 3
        assert (first_seed.inserted, first_seed.unchanged, first_seed.retired, first_seed.replaced) == (3, 0, 0, 0)
        assert (first_discovery.documents, first_discovery.succeeded, first_discovery.failed) == (24, 24, 0)
        assert first_graph.validation.valid is True

        assert (second_import.new_source_versions, second_import.imported_passages) == (0, 0)
        assert (second_seed.inserted, second_seed.retired, second_seed.replaced, second_seed.evidence_inserted) == (0, 0, 0, 0)
        assert second_seed.unchanged == 3
        assert (second_discovery.succeeded, second_discovery.failed, second_discovery.evidence_inserted) == (0, 0, 0)
        assert second_discovery.reused == second_discovery.documents == 24
        assert second_graph.reused is True
    finally:
        assert_protected_inputs_unchanged(PACKAGE_ROOT, before)


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


def test_fixture_orchestration_prepares_aliases_idempotently_without_semantic_identity_drift(
    db_engine, tmp_path, monkeypatch
) -> None:
    """Alias preparation is a structural, scoped prerequisite and pending discoveries cannot rewrite fixture identity."""
    catalog = load_fixture_catalog(PACKAGE_ROOT)
    clients = ModelClients(_EmbeddingDouble(), _JevDouble(), _ExtractionDouble())
    real_load = fixture_runs.load_alias_seeds
    real_discover = fixture_runs.discover_aliases_deterministic
    seed_reports: list[AliasSeedReport] = []
    discovery_reports: list[AliasDiscoveryReport] = []
    injected = False
    controlled_pending: tuple[Scope, str] | None = None

    def load_seeds(engine, scope):
        report = real_load(engine, scope)
        seed_reports.append(report)
        return report

    def discover(engine, scope):
        nonlocal controlled_pending, injected
        report = real_discover(engine, scope)
        # The supplied scenario prose intentionally need not contain an alias
        # pattern.  Attach one grounded deterministic-detector observation to the
        # ordinary discovery boundary so the fixture can exercise human approval
        # and catalog invalidation without changing protected corpus inputs.
        if not injected and scope.project_id == "project-one":
            with engine.begin() as connection:
                source = connection.execute(text(
                    """SELECT source_id,source_version,passage_id,text
                       FROM source_passage
                       WHERE tenant_id=:tenant AND project_id=:project
                       ORDER BY source_id,source_version,passage_id LIMIT 1"""
                ), {"tenant": scope.tenant_id, "project": scope.project_id}).mappings().one()
                quote = str(source["text"])
                persisted = record_alias_proposal(
                    connection,
                    scope,
                    AliasProposal(
                        alias_value="fixture-controlled-alias", canonical_name=None,
                        entity_type=None, canonical_key=None, confidence=0.5,
                        source=ScopedPassageIdentity(
                            scope.tenant_id, scope.project_id, str(source["source_id"]),
                            str(source["source_version"]), str(source["passage_id"]),
                        ),
                        exact_quote=quote, original_start=0, original_end=len(quote),
                        source_type="deterministic_rule", detector_name="fixture-deterministic-detector",
                        detector_version="fixture-deterministic-detector-v1",
                    ),
                )
                stored = connection.execute(text(
                    """SELECT status,managed_by FROM aliases
                       WHERE tenant_id=:tenant AND project_id=:project AND alias_id=:alias_id"""
                ), {
                    "tenant": scope.tenant_id, "project": scope.project_id,
                    "alias_id": persisted.alias_id,
                }).one()
                assert stored == ("pending", "review")
                controlled_pending = (scope, persisted.alias_id)
            injected = True
        discovery_reports.append(report)
        return report

    monkeypatch.setattr(fixture_runs, "load_alias_seeds", load_seeds)
    monkeypatch.setattr(fixture_runs, "discover_aliases_deterministic", discover)

    first = run_kg_fixtures(db_engine, catalog, tmp_path / "results" / "first", clients)
    with db_engine.connect() as connection:
        first_identity = connection.execute(text(
            "SELECT count(*) FROM kg_node WHERE tenant_id='fixture-tenant'"
        )).scalar_one()
        pending_before = connection.execute(text(
            "SELECT count(*) FROM aliases WHERE tenant_id='fixture-tenant' AND status='pending'"
        )).scalar_one()

    second = run_kg_fixtures(db_engine, catalog, tmp_path / "results" / "second", clients)
    with db_engine.connect() as connection:
        second_identity = connection.execute(text(
            "SELECT count(*) FROM kg_node WHERE tenant_id='fixture-tenant'"
        )).scalar_one()
        pending_after = connection.execute(text(
            "SELECT count(*) FROM aliases WHERE tenant_id='fixture-tenant' AND status='pending'"
        )).scalar_one()

    assert len(seed_reports) == len(first.runs) + len(second.runs)
    assert len(discovery_reports) == len(seed_reports)
    assert all(report.inserted == 0 for report in seed_reports)
    assert any(report.succeeded > 0 for report in discovery_reports[: len(first.runs)])
    assert all(report.failed == 0 for report in discovery_reports[: len(first.runs)])
    assert all(report.succeeded == 0 and report.reused == report.documents for report in discovery_reports[len(first.runs) :])
    assert first_identity == second_identity
    assert pending_before == pending_after
    assert all(item.export.version_dir.is_dir() for item in second.runs)

    # A review decision must affect a later semantic run, never the graph already
    # materialized above.  Select the grounded fixture-controlled deterministic
    # proposal attached at the public discovery boundary above.
    assert controlled_pending is not None
    scope, controlled_alias_id = controlled_pending
    with db_engine.connect() as connection:
        persisted = connection.execute(text(
            """SELECT status,managed_by,normalized_alias FROM aliases
               WHERE tenant_id=:tenant AND project_id=:project AND alias_id=:alias_id"""
        ), {
            "tenant": scope.tenant_id, "project": scope.project_id,
            "alias_id": controlled_alias_id,
        }).one()
        assert persisted == ("pending", "review", "fixture-controlled-alias")
        before_approval = approved_alias_catalog(connection, scope).version

    decide_alias(
        db_engine,
        scope,
        AliasDecision(
            controlled_alias_id, "approved", "fixture-reviewer",
            entity_type="Assay", canonical_key="fixture-assay", canonical_name="Fixture Assay",
        ),
    )
    with db_engine.connect() as connection:
        after_approval = approved_alias_catalog(connection, scope).version
    assert after_approval != before_approval

    third = run_kg_fixtures(db_engine, catalog, tmp_path / "results" / "third", clients)
    with db_engine.connect() as connection:
        jev_versions = connection.execute(text(
            """SELECT DISTINCT alias_catalog_version FROM jev_response
               WHERE tenant_id=:tenant AND project_id=:project AND status='succeeded'"""
        ), {"tenant": scope.tenant_id, "project": scope.project_id}).scalars().all()
        extraction_versions = connection.execute(text(
            """SELECT DISTINCT alias_catalog_version FROM llm_extraction_run
               WHERE tenant_id=:tenant AND project_id=:project AND status='succeeded'"""
        ), {"tenant": scope.tenant_id, "project": scope.project_id}).scalars().all()

    assert before_approval in jev_versions and after_approval in jev_versions
    assert before_approval in extraction_versions and after_approval in extraction_versions
    assert all(item.export.version_dir.is_dir() for item in third.runs)
