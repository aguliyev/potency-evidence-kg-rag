from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

import potency_kg.resolution as resolution_module
from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.ids import derive_scoped_id
from potency_kg.project_profiles import ProjectProfile
from potency_kg.resolution import (
    EntityMentionDraft,
    bootstrap_profile_aliases,
    materialize_extraction,
    resolve_mention,
)


SCOPE = Scope("public-fda", "casgevy-potency")
PASSAGE = ScopedPassageIdentity(*SCOPE.as_tuple(), "source", "v1", "p1")
ALIASES = {
    "CASGEVY": "casgevy",
    "exagamglogene autotemcel": "casgevy",
    "exa-cel": "casgevy",
}
PROFILE = ProjectProfile(
    "casgevy-v1",
    ALIASES,
    "configs/casgevy-potency.json",
    "descriptor-sha",
    "profile-version",
    SCOPE,
    {"125785": "casgevy-tdt", "125787": "casgevy-scd"},
)


def _response() -> dict[str, object]:
    return {
        "entities": [
            {
                "type": "Product",
                "local_id": "product-1",
                "name": "CASGEVY",
                "supporting_quote": "CASGEVY",
                "original_quote": "CASGEVY",
                "original_start": 0,
                "original_end": 7,
                "model_start": 0,
                "model_end": 7,
            },
            {
                "type": "AliasCandidate",
                "local_id": "alias-1",
                "candidate": "CGV-X",
                "product_local_id": "product-1",
                "supporting_quote": "CGV-X",
                "original_quote": "CGV-X",
                "original_start": 16,
                "original_end": 21,
                "model_start": 16,
                "model_end": 21,
            },
            {
                "type": "Assay",
                "local_id": "assay-1",
                "name": "flow assay",
                "role": "release",
                "supporting_quote": "flow assay",
                "original_quote": "flow assay",
                "original_start": 34,
                "original_end": 44,
                "model_start": 34,
                "model_end": 44,
            },
            {
                "type": "MeasuredAttribute",
                "local_id": "attribute-1",
                "name": "editing rate",
                "supporting_quote": "editing rate",
                "original_quote": "editing rate",
                "original_start": 54,
                "original_end": 66,
                "model_start": 54,
                "model_end": 66,
            },
        ],
        "assertions": [
            {
                "local_id": "assertion-1",
                "claim": "CASGEVY uses a flow assay for editing rate.",
                "supporting_quote": "CASGEVY uses a flow assay for editing rate",
                "original_quote": "CASGEVY uses a flow assay for editing rate",
                "original_start": 0,
                "original_end": 43,
                "model_start": 0,
                "model_end": 43,
                "chronology_state": "reported",
                "review_state": "unreviewed",
                "authority": "FDA",
                "status": "accepted",
            }
        ],
        "relationships": [
            {
                "local_id": "relation-1",
                "relation_type": "applies_to",
                "source_local_id": "assertion-1",
                "target_local_id": "product-1",
                "assertion_local_id": "assertion-1",
                "source_type": "Assertion",
                "target_type": "Product",
                "attributes": {},
            },
            {
                "local_id": "relation-2",
                "relation_type": "measures",
                "source_local_id": "assay-1",
                "target_local_id": "attribute-1",
                "assertion_local_id": "assertion-1",
                "source_type": "Assay",
                "target_type": "MeasuredAttribute",
                "attributes": {},
            },
        ],
        "source_refs": [
            {
                "local_id": "source-ref-1",
                "record_local_id": "assertion-1",
                "supporting_quote": "CASGEVY uses a flow assay for editing rate",
                "original_quote": "CASGEVY uses a flow assay for editing rate",
                "original_start": 0,
                "original_end": 43,
                "model_start": 0,
                "model_end": 43,
            }
        ],
    }


def _insert_run(connection, extraction_run_id: UUID, response: dict[str, object]) -> None:
    connection.execute(
        text(
            """INSERT INTO llm_extraction_run
               (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,
                extraction_version,model,sanitized_origin,credential_env,instructor_mode,
                selected_bundles,input_sha256,profile_name,profile_aliases,
                profile_source_sha256,profile_version,status,response)
               VALUES (:run_id,:tenant,:project,:source_id,:source_version,:passage_id,
                :version,'generation-test','https://models.example.test/v1','LLM_API_KEY','openai',
                '["product_context","alias_identity","assay_evidence"]',
                'input-sha','casgevy-v1',CAST(:aliases AS jsonb),'descriptor-sha','profile-version',
                'succeeded',CAST(:response AS jsonb))"""
        ),
        {
            "run_id": extraction_run_id,
            "tenant": PASSAGE.tenant_id,
            "project": PASSAGE.project_id,
            "source_id": PASSAGE.source_id,
            "source_version": PASSAGE.source_version,
            "passage_id": PASSAGE.passage_id,
            "version": f"extraction-{extraction_run_id.hex}",
            "aliases": json.dumps(ALIASES),
            "response": json.dumps(response),
        },
    )


def _seed_passage(db_engine) -> tuple[UUID, UUID]:
    previous, candidate = uuid4(), uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text(
                """INSERT INTO source_document
                   (tenant_id,project_id,source_id,source_version,title,source_sha256,expected_passage_count,status)
                   VALUES (:tenant,:project,'source','v1','title','source-sha',1,'final')"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        )
        connection.execute(
            text(
                """INSERT INTO source_passage
                   (tenant_id,project_id,source_id,source_version,passage_id,text,text_sha256)
                   VALUES (:tenant,:project,'source','v1','p1',
                    'CASGEVY uses CGV-X and a flow assay for editing rate.','text-sha')"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        )
        passage_node = derive_scoped_id("passage", SCOPE, "source", "v1", "p1")
        connection.execute(
            text(
                """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
                   VALUES (:tenant,:project,:node_id,'SourcePassage',CAST(:attributes AS jsonb))"""
            ),
            {
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "node_id": passage_node,
                "attributes": json.dumps({"system_generated": True}),
            },
        )
        _insert_run(connection, previous, {"entities": [], "assertions": [], "relationships": [], "source_refs": []})
        _insert_run(connection, candidate, _response())
        connection.execute(
            text(
                """INSERT INTO passage_processing
                   (tenant_id,project_id,source_id,source_version,passage_id,state,reason,active_extraction_run_id)
                   VALUES (:tenant,:project,'source','v1','p1','extracted','extracted_successfully',:previous)"""
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id, "previous": previous},
        )
    return previous, candidate


def test_profile_aliases_are_the_only_identity_merges_and_have_profile_provenance(db_engine) -> None:
    """Similarity and extracted candidate state must never become an implicit identity merge."""
    with db_engine.begin() as connection:
        report = bootstrap_profile_aliases(connection, PROFILE)
        resolved = [
            resolve_mention(
                connection,
                SCOPE,
                EntityMentionDraft(
                    local_id=f"m-{index}",
                    entity_type="Product",
                    text=alias,
                    source_ref=PASSAGE,
                    original_quote=alias,
                ),
                PROFILE,
            )
            for index, alias in enumerate(ALIASES)
        ]
        similar = resolve_mention(
            connection,
            SCOPE,
            EntityMentionDraft(
                local_id="m-similar",
                entity_type="Product",
                text="CASGEVY therapy",
                source_ref=PASSAGE,
                original_quote="CASGEVY therapy",
            ),
            PROFILE,
        )

    assert report.alias_count == 3 and report.product_count == 1 and report.application_count == 2
    assert {item.node_id for item in resolved} == {derive_scoped_id("product", SCOPE, "casgevy")}
    assert all(item.matched_by == "trusted_profile_alias" for item in resolved)
    assert similar.node_id != resolved[0].node_id and similar.matched_by == "new_entity"
    with db_engine.connect() as connection:
        product = connection.execute(
            text(
                """SELECT attributes FROM kg_node
                   WHERE tenant_id=:tenant AND project_id=:project AND node_id=:node_id"""
            ),
            {
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "node_id": resolved[0].node_id,
            },
        ).scalar_one()
        assert connection.execute(text("SELECT count(*) FROM alias_candidate")).scalar_one() == 0
        assert connection.execute(
            text("SELECT count(*) FROM kg_node WHERE node_type='Application'")
        ).scalar_one() == 2
        application_profiles = connection.execute(
            text("SELECT attributes FROM kg_node WHERE node_type='Application' ORDER BY node_id")
        ).scalars().all()
    assert product["profile_version"] == "profile-version"
    assert product["profile_name"] == "casgevy-v1"
    assert product["profile_source_path"] == "configs/casgevy-potency.json"
    assert product["profile_source_sha256"] == "descriptor-sha"
    assert product["trusted_aliases"] == list(ALIASES)
    assert "quote" not in product
    assert all(
        item["profile_source_path"] == "configs/casgevy-potency.json"
        and item["profile_source_sha256"] == "descriptor-sha"
        for item in application_profiles
    )
    assert not any(
        hasattr(resolution_module, name)
        for name in ("approve_alias", "reject_alias", "redirect_entity", "approve_candidate")
    )


def test_materialization_commits_resolution_provenance_and_current_pointer_together(db_engine) -> None:
    """A successful extraction becomes current only with all semantic rows and provenance present."""
    previous, candidate = _seed_passage(db_engine)

    report = materialize_extraction(db_engine, PASSAGE, candidate)

    assert report.extraction_run_id == candidate
    assert report.previous_active_extraction_run_id == previous
    assert report.assertions == 1 and report.evidence_edges == 2
    with db_engine.connect() as connection:
        state = connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one()
        run = connection.execute(
            text("SELECT status,previous_active_extraction_run_id,counts FROM materialization_run")
        ).mappings().one()
        candidate_row = connection.execute(
            text(
                """SELECT candidate,extraction_run_id,proposer,extraction_version,original_quote,status,
                          tenant_id,project_id,source_id,source_version,passage_id
                   FROM alias_candidate"""
            )
        ).one()
        assertions = connection.execute(
            text("SELECT extraction_run_id,claim,original_quote FROM assertion")
        ).all()
        evidence_edges = connection.execute(
            text("SELECT assertion_id,is_evidence FROM kg_edge WHERE is_evidence ORDER BY edge_id")
        ).all()
        node_refs = connection.execute(
            text("SELECT count(*) FROM kg_node_source_ref WHERE tenant_id=:tenant AND project_id=:project"),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).scalar_one()
        edge_refs = connection.execute(
            text("SELECT count(*) FROM kg_edge_source_ref WHERE tenant_id=:tenant AND project_id=:project"),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).scalar_one()
        trusted_product = connection.execute(
            text(
                """SELECT attributes FROM kg_node
                   WHERE node_id=:node_id AND node_type='Product'"""
            ),
            {"node_id": derive_scoped_id("product", SCOPE, "casgevy")},
        ).scalar_one()

    assert state == ("extracted", "extracted_successfully", candidate)
    assert run["status"] == "succeeded" and run["previous_active_extraction_run_id"] == previous
    assert run["counts"]["assertions"] == 1
    assert candidate_row == (
        "CGV-X",
        candidate,
        "instructor",
        f"extraction-{candidate.hex}",
        "CGV-X",
        "pending",
        *PASSAGE.as_tuple(),
    )
    assert assertions == [(candidate, "CASGEVY uses a flow assay for editing rate.", "CASGEVY uses a flow assay for editing rate")]
    assert len(evidence_edges) == 2 and all(row[0] is not None and row[1] for row in evidence_edges)
    assert node_refs >= 4 and edge_refs == 2
    assert trusted_product["profile_source_path"] == "configs/casgevy-potency.json"
    with db_engine.begin() as connection:
        proposed_alias = resolve_mention(
            connection,
            SCOPE,
            EntityMentionDraft(
                local_id="candidate-reuse-check",
                entity_type="Product",
                text="CGV-X",
                source_ref=PASSAGE,
                original_quote="CGV-X",
            ),
            PROFILE,
        )
    assert proposed_alias.matched_by == "new_entity"
    assert proposed_alias.node_id != derive_scoped_id("product", SCOPE, "casgevy")


@pytest.mark.parametrize(
    "failure_at",
    [
        "after_resolution",
        "after_assertions",
        "after_relationships",
        "before_activation",
        "after_completion",
    ],
)
def test_injected_materialization_failure_preserves_previous_pointer_and_no_partial_semantics(
    db_engine,
    failure_at: str,
) -> None:
    """Every semantic write group must roll back before a candidate extraction can become active."""
    previous, candidate = _seed_passage(db_engine)

    with pytest.raises(RuntimeError, match="injected materialization failure"):
        materialize_extraction(db_engine, PASSAGE, candidate, failure_at=failure_at)

    with db_engine.connect() as connection:
        state = connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one()
        assert connection.execute(
            text("SELECT count(*) FROM entity_mention WHERE extraction_run_id=:run_id"),
            {"run_id": candidate},
        ).scalar_one() == 0
        assert connection.execute(
            text("SELECT count(*) FROM assertion WHERE extraction_run_id=:run_id"),
            {"run_id": candidate},
        ).scalar_one() == 0
        assert connection.execute(
            text("SELECT count(*) FROM alias_candidate WHERE extraction_run_id=:run_id"),
            {"run_id": candidate},
        ).scalar_one() == 0
        materialization = connection.execute(
            text(
                """SELECT status,previous_active_extraction_run_id,error_detail
                   FROM materialization_run WHERE extraction_run_id=:run_id"""
            ),
            {"run_id": candidate},
        ).one()
        assert connection.execute(text("SELECT count(*) FROM kg_edge")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 1

    assert state == ("extracted", "extracted_successfully", previous)
    assert materialization[0] == "failed" and materialization[1] == previous
    assert "injected" not in (materialization[2] or "")


def test_materialization_rejects_cross_project_extraction_id_even_with_colliding_local_ids(db_engine) -> None:
    """A globally colliding UUID or local IDs must not bypass full tenant/project/source identity checks."""
    previous, candidate = _seed_passage(db_engine)
    wrong_scope = ScopedPassageIdentity(PASSAGE.tenant_id, "other-project", *PASSAGE.as_tuple()[2:])

    with pytest.raises(ValueError, match="full passage identity"):
        materialize_extraction(db_engine, wrong_scope, candidate)

    with db_engine.connect() as connection:
        state = connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one()
        assert connection.execute(text("SELECT count(*) FROM materialization_run")).scalar_one() == 0
    assert state == ("extracted", "extracted_successfully", previous)


def test_materialization_requires_the_exact_current_processing_row(db_engine) -> None:
    """A matching immutable run without its exact current-state row must not report false activation."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        connection.execute(text("DELETE FROM passage_processing"))

    with pytest.raises(ValueError, match="processing state"):
        materialize_extraction(db_engine, PASSAGE, candidate)

    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM materialization_run")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM assertion")).scalar_one() == 0


def test_process_control_interrupt_rolls_back_without_becoming_a_materialization_failure(db_engine) -> None:
    """A process-control signal must roll back the transaction without being rewritten as an application error."""
    previous, candidate = _seed_passage(db_engine)

    def interrupt(point: str) -> None:
        if point == "after_resolution":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        materialize_extraction(db_engine, PASSAGE, candidate, failure_injector=interrupt)

    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM materialization_run")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM entity_mention")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 1
        assert connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one() == ("extracted", "extracted_successfully", previous)
