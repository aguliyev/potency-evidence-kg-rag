from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from uuid import UUID, uuid4

import potency_kg.cli as cli_module
import pytest
from potency_kg.aliases import ApprovedAlias, AliasCatalog, approved_alias_catalog, normalize_alias
from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.extraction import build_response_model
from potency_kg.ids import canonical_json, derive_extraction_version, derive_scoped_id
from potency_kg.prompt_resources.extraction import assemble_prompt
from potency_kg.resolution import (
    EntityMentionDraft,
    materialize_extraction,
    resolve_mention,
)
from sqlalchemy import text

SCOPE = Scope("public-fda", "casgevy-potency")
PASSAGE = ScopedPassageIdentity(*SCOPE.as_tuple(), "source", "v1", "p1")
APPROVED_ALIASES = (
    ("catalog-casgevy", "CASGEVY", "Product", "casgevy", "CASGEVY"),
    (
        "catalog-exagamglogene",
        "exagamglogene autotemcel",
        "Product",
        "casgevy",
        "CASGEVY",
    ),
    ("catalog-exa-cel", "exa-cel", "Product", "casgevy", "CASGEVY"),
)
SELECTED_BUNDLES = ("product_context", "alias_identity", "assay_evidence")
CURRENT_PROMPT_HASH = sha256(assemble_prompt(SELECTED_BUNDLES).encode()).hexdigest()
CURRENT_SCHEMA_HASH = sha256(
    canonical_json(build_response_model(SELECTED_BUNDLES).model_json_schema()).encode()
).hexdigest()


def _insert_approved_aliases(connection) -> AliasCatalog:
    for alias_id, value, entity_type, key, name in APPROVED_ALIASES:
        connection.execute(
            text("""INSERT INTO aliases
                   (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,
                    canonical_key,canonical_name,status,managed_by,normalization_version)
                   VALUES (:tenant,:project,:alias_id,:value,:normalized,:entity_type,
                    :canonical_key,:canonical_name,'approved','static_config',
                    'nfkc-casefold-whitespace-v1')"""),
            {
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "alias_id": alias_id,
                "value": value,
                "normalized": normalize_alias(value),
                "entity_type": entity_type,
                "canonical_key": key,
                "canonical_name": name,
            },
        )
    return approved_alias_catalog(connection, SCOPE)


def _profile_payload(catalog: AliasCatalog) -> dict[str, object]:
    return {
        "name": "casgevy-v1",
        "aliases": [asdict(entry) for entry in catalog.entries],
        "alias_catalog_version": catalog.version,
        "descriptor_sha256": "descriptor-sha",
        "profile_version": "profile-version",
    }


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
                "type": "Application",
                "local_id": "application-1",
                "application_id": "125785",
                "supporting_quote": "125785",
                "original_quote": "125785",
                "original_start": 8,
                "original_end": 14,
                "model_start": 8,
                "model_end": 14,
            },
            {
                "type": "AliasCandidate",
                "local_id": "alias-1",
                "candidate": "CGV-X",
                "target_local_id": "product-1",
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


def _insert_run(
    connection,
    extraction_run_id: UUID,
    response: dict[str, object],
    *,
    prompt_hash: str = CURRENT_PROMPT_HASH,
    schema_hash: str = CURRENT_SCHEMA_HASH,
    normalization: str = "redaction-v1",
    model: str = "generation-test",
    instructor_mode: str = "openai",
) -> None:
    catalog = approved_alias_catalog(connection, SCOPE)
    profile = _profile_payload(catalog)
    version = derive_extraction_version(
        model=model,
        instructor_mode=instructor_mode,
        selected_bundles=SELECTED_BUNDLES,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        normalization=normalization,
        profile=profile,
    )
    connection.execute(
        text("""INSERT INTO llm_extraction_run
               (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,
                extraction_version,model,sanitized_origin,credential_env,instructor_mode,
                selected_bundles,prompt_hash,schema_hash,normalization,input_sha256,profile_name,profile_aliases,
                profile_source_sha256,profile_version,alias_catalog_version,status,response)
               VALUES (:run_id,:tenant,:project,:source_id,:source_version,:passage_id,
                :version,:model,'https://models.example.test/v1','LLM_API_KEY',:instructor_mode,
                '["product_context","alias_identity","assay_evidence"]',:prompt_hash,:schema_hash,
                :normalization,
                'input-sha','casgevy-v1',CAST(:aliases AS jsonb),'descriptor-sha','profile-version',
                :catalog_version,
                'succeeded',CAST(:response AS jsonb))"""),
        {
            "run_id": extraction_run_id,
            "tenant": PASSAGE.tenant_id,
            "project": PASSAGE.project_id,
            "source_id": PASSAGE.source_id,
            "source_version": PASSAGE.source_version,
            "passage_id": PASSAGE.passage_id,
            "version": version,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "normalization": normalization,
            "model": model,
            "instructor_mode": instructor_mode,
            "aliases": json.dumps(profile["aliases"]),
            "catalog_version": catalog.version,
            "response": json.dumps(response),
        },
    )


def _seed_passage(db_engine) -> tuple[UUID, UUID]:
    previous, candidate, response_id, selection_id = uuid4(), uuid4(), uuid4(), uuid4()
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
            text("""INSERT INTO source_head
                   (tenant_id,project_id,source_id,source_version)
                   VALUES (:tenant,:project,'source','v1')"""),
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
        catalog = _insert_approved_aliases(connection)
        profile = _profile_payload(catalog)
        _insert_run(
            connection,
            previous,
            {"entities": [], "assertions": [], "relationships": [], "source_refs": []},
        )
        _insert_run(connection, candidate, _response())
        connection.execute(
            text(
                """INSERT INTO jev_response
                   (response_id,tenant_id,project_id,source_id,source_version,passage_id,
                    jev_response_version,model,questions,probabilities,input_sha256,
                    profile_name,profile_aliases,profile_source_sha256,profile_version,
                    alias_catalog_version,status)
                   VALUES (:response_id,:tenant,:project,'source','v1','p1','jev-response-test',
                    'jev-test','{}','{}','input-sha','casgevy-v1',CAST(:aliases AS jsonb),
                    'descriptor-sha','profile-version',:catalog_version,'succeeded')"""
            ),
            {
                "response_id": response_id,
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "aliases": json.dumps(profile["aliases"]),
                "catalog_version": catalog.version,
            },
        )
        connection.execute(
            text(
                """INSERT INTO jev_selection
                   (selection_id,response_id,tenant_id,project_id,source_id,source_version,
                    passage_id,thresholds,selected_bundles,jev_selection_version)
                   VALUES (:selection_id,:response_id,:tenant,:project,'source','v1','p1','{}',
                    '["product_context","alias_identity","assay_evidence"]','selection-test')"""
            ),
            {
                "selection_id": selection_id,
                "response_id": response_id,
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
            },
        )
        connection.execute(
            text("""INSERT INTO passage_processing
                   (tenant_id,project_id,source_id,source_version,passage_id,state,reason,
                    active_jev_selection_id,active_extraction_run_id,validated_extraction_run_id)
                   VALUES (:tenant,:project,'source','v1','p1','extracted','extracted_successfully',
                    :selection_id,:previous,:candidate)"""),
            {
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "selection_id": selection_id,
                "previous": previous,
                "candidate": candidate,
            },
        )
    return previous, candidate


def test_materialization_rejects_run_from_prior_active_jev_selection(db_engine) -> None:
    """A newly active bundle selection must not reactivate a successful older extraction."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        response_id, selection_id = uuid4(), uuid4()
        catalog = approved_alias_catalog(connection, SCOPE)
        profile = _profile_payload(catalog)
        connection.execute(
            text(
                """INSERT INTO jev_response
                   (response_id,tenant_id,project_id,source_id,source_version,passage_id,
                    jev_response_version,model,questions,probabilities,input_sha256,
                    profile_name,profile_aliases,profile_source_sha256,profile_version,
                    alias_catalog_version,status)
                   VALUES (:response_id,:tenant,:project,'source','v1','p1','jev-response-new',
                    'jev-test','{}','{}','input-sha','casgevy-v1',CAST(:aliases AS jsonb),
                    'descriptor-sha','profile-version',:catalog_version,'succeeded')"""
            ),
            {
                "response_id": response_id,
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
                "aliases": json.dumps(profile["aliases"]),
                "catalog_version": catalog.version,
            },
        )
        connection.execute(
            text("""INSERT INTO jev_selection
                   (selection_id,response_id,tenant_id,project_id,source_id,source_version,
                    passage_id,thresholds,selected_bundles,jev_selection_version)
                   VALUES (:selection_id,:response_id,:tenant,:project,'source','v1','p1','{}',
                    '["regulatory_history"]','selection-new')"""),
            {
                "selection_id": selection_id,
                "response_id": response_id,
                "tenant": SCOPE.tenant_id,
                "project": SCOPE.project_id,
            },
        )
        connection.execute(
            text(
                """UPDATE passage_processing
                   SET state='selected',reason='jev_bundle_selected',
                       active_jev_selection_id=:selection_id,active_extraction_run_id=NULL"""
            ),
            {"selection_id": selection_id},
        )

    assert cli_module._pending_materializations(db_engine, SCOPE) == []
    with pytest.raises(ValueError, match="active Jev selection"):
        materialize_extraction(db_engine, PASSAGE, candidate)


def test_materialization_rejects_candidate_with_stale_normalization(db_engine) -> None:
    """The candidate normalization must equal the active Jev response normalization."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        connection.execute(text("UPDATE jev_response SET normalization='redaction-v2'"))
        connection.execute(
            text(
                "UPDATE passage_processing SET state='selected',reason='extraction_validated'"
            )
        )

    assert cli_module._pending_materializations(db_engine, SCOPE) == []
    with pytest.raises(ValueError, match="active Jev selection identity"):
        materialize_extraction(db_engine, PASSAGE, candidate)


def test_materialization_replays_with_new_alias_catalog_without_reextract(
    db_engine,
) -> None:
    """Approved aliases may change resolution without invalidating the extraction response."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE passage_processing SET state='selected',reason='extraction_validated'"
            )
        )
    assert cli_module._pending_materializations(db_engine, SCOPE) == [
        (PASSAGE, candidate)
    ]

    with db_engine.begin() as connection:
        old_version = connection.execute(
            text(
                "SELECT alias_catalog_version FROM llm_extraction_run WHERE extraction_run_id=:run_id"
            ),
            {"run_id": candidate},
        ).scalar_one()
        connection.execute(
            text("""INSERT INTO aliases
                   (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,
                    canonical_key,canonical_name,status,managed_by,normalization_version,
                    decided_at,decided_by)
                   VALUES (:tenant,:project,'approved-after-extraction','CGV-X','cgv-x',
                    'Product','casgevy','CASGEVY','approved','review',
                   'nfkc-casefold-whitespace-v1',now(),'reviewer')"""),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        )
        connection.execute(
            text("""INSERT INTO aliases
                   (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,
                    canonical_key,canonical_name,status,managed_by,normalization_version,
                    decided_at,decided_by)
                   VALUES (:tenant,:project,'approved-assay-after-extraction','flow assay','flow assay',
                    'Assay','flow','Flow assay','approved','review',
                    'nfkc-casefold-whitespace-v1',now(),'reviewer')"""),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        )
        new_version = approved_alias_catalog(connection, SCOPE).version

    assert new_version != old_version
    assert cli_module._pending_materializations(db_engine, SCOPE) == [
        (PASSAGE, candidate)
    ]
    report = materialize_extraction(db_engine, PASSAGE, candidate)
    assert report.extraction_run_id == candidate
    with db_engine.connect() as connection:
        assert connection.execute(
            text("SELECT count(*) FROM kg_node WHERE node_id=:node_id"),
            {"node_id": derive_scoped_id("entity", SCOPE, "Assay", "flow")},
        ).scalar_one() == 1
        assert connection.execute(
            text("SELECT resolution_alias_catalog_version FROM materialization_run")
        ).scalar_one() == new_version


def test_materialization_replays_against_current_catalog_not_historical_snapshot(
    db_engine,
) -> None:
    """A historical extraction snapshot does not block current-catalog materialization."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        catalog = approved_alias_catalog(connection, SCOPE)
        profile = _profile_payload(catalog)
        profile["aliases"] = list(profile["aliases"])[1:]
        mismatched_version = derive_extraction_version(
            model="generation-test",
            instructor_mode="openai",
            selected_bundles=SELECTED_BUNDLES,
            prompt_hash=CURRENT_PROMPT_HASH,
            schema_hash=CURRENT_SCHEMA_HASH,
            normalization="redaction-v1",
            profile=profile,
        )
        aliases = json.dumps(profile["aliases"])
        connection.execute(
            text("UPDATE jev_response SET profile_aliases=CAST(:aliases AS jsonb)"),
            {"aliases": aliases},
        )
        connection.execute(
            text("""UPDATE llm_extraction_run
                   SET profile_aliases=CAST(:aliases AS jsonb),extraction_version=:version
                   WHERE extraction_run_id=:run_id"""),
            {"aliases": aliases, "version": mismatched_version, "run_id": candidate},
        )
        connection.execute(
            text(
                "UPDATE passage_processing SET state='selected',reason='extraction_validated'"
            )
        )

    assert cli_module._pending_materializations(db_engine, SCOPE) == [(PASSAGE, candidate)]
    report = materialize_extraction(db_engine, PASSAGE, candidate)
    assert report.extraction_run_id == candidate


def test_materialization_rejects_self_consistent_stale_prompt_and_schema(
    db_engine,
) -> None:
    """Candidate hashes cannot define their own current extraction contract."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.connect() as connection:
        profile = _profile_payload(approved_alias_catalog(connection, SCOPE))
    stale_version = derive_extraction_version(
        model="generation-test",
        instructor_mode="openai",
        selected_bundles=SELECTED_BUNDLES,
        prompt_hash="stale-prompt",
        schema_hash="stale-schema",
        normalization="redaction-v1",
        profile=profile,
    )
    with db_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE llm_extraction_run SET prompt_hash='stale-prompt', "
                "schema_hash='stale-schema',extraction_version=:version "
                "WHERE extraction_run_id=:run_id"
            ),
            {"version": stale_version, "run_id": candidate},
        )
        connection.execute(
            text(
                "UPDATE passage_processing SET state='selected',reason='extraction_validated'"
            )
        )

    assert cli_module._pending_materializations(db_engine, SCOPE) == []
    with pytest.raises(ValueError, match="active Jev selection identity"):
        materialize_extraction(db_engine, PASSAGE, candidate)


def test_materialization_rejects_a_different_self_consistent_adapter_run(
    db_engine,
) -> None:
    """Only the exact run activated by extraction may be materialized, not a compatible historical run."""
    _previous, candidate = _seed_passage(db_engine)
    historical = uuid4()
    with db_engine.begin() as connection:
        _insert_run(
            connection,
            historical,
            _response(),
            model="historical-generation",
            instructor_mode="legacy",
        )

    with pytest.raises(ValueError, match="active extraction identity"):
        materialize_extraction(db_engine, PASSAGE, historical)

    report = materialize_extraction(db_engine, PASSAGE, candidate)
    assert report.extraction_run_id == candidate


def test_resolution_uses_only_exact_approved_typed_catalog_mappings(db_engine) -> None:
    """Pending/rejected aliases or cross-type matches must never merge a Product or Assay identity."""
    with db_engine.begin() as connection:
        for alias_id, value, entity_type, key, name, status in (
            (
                "product-approved",
                "exa-cel",
                "Product",
                "casgevy",
                "CASGEVY",
                "approved",
            ),
            (
                "assay-approved",
                "flow cytometry",
                "Assay",
                "flow",
                "Flow cytometry",
                "approved",
            ),
            ("product-pending", "cgv-x", None, None, None, "pending"),
            (
                "product-rejected",
                "casgevy therapy",
                "Product",
                "casgevy",
                "CASGEVY",
                "rejected",
            ),
        ):
            connection.execute(
                text(
                    """INSERT INTO aliases
                (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,normalization_version,decided_at,decided_by)
                VALUES (:tenant,:project,:id,:value,lower(:value),:type,:key,:name,:status,'review','nfkc-casefold-whitespace-v1',CASE WHEN :status='pending' THEN NULL ELSE now() END,CASE WHEN :status='pending' THEN NULL ELSE 'reviewer' END)"""
                ),
                {
                    "tenant": SCOPE.tenant_id,
                    "project": SCOPE.project_id,
                    "id": alias_id,
                    "value": value,
                    "type": entity_type,
                    "key": key,
                    "name": name,
                    "status": status,
                },
            )
        catalog = approved_alias_catalog(connection, SCOPE)

        def resolve(entity_type: str, value: str):
            return resolve_mention(
                connection,
                SCOPE,
                EntityMentionDraft(
                    "local-" + value, entity_type, value, PASSAGE, value
                ),
                catalog,
            )

        product = resolve("Product", " EXA-CEL ")
        assay = resolve("Assay", "flow cytometry")
        pending = resolve("Product", "cgv-x")
        rejected = resolve("Product", "casgevy therapy")
        similar = resolve("Product", "exa-cel therapy")
        wrong_type = resolve("Product", "flow cytometry")

    assert (product.node_type, product.canonical_name, product.matched_by) == (
        "Product",
        "CASGEVY",
        "approved_alias",
    )
    assert (assay.node_type, assay.canonical_name, assay.matched_by) == (
        "Assay",
        "Flow cytometry",
        "approved_alias",
    )
    assert product.node_id == derive_scoped_id("entity", SCOPE, "Product", "casgevy")
    assert assay.node_id == derive_scoped_id("entity", SCOPE, "Assay", "flow")
    assert (
        len({pending.node_id, rejected.node_id, similar.node_id, wrong_type.node_id})
        == 4
    )
    assert all(
        item.matched_by == "new_entity"
        for item in (pending, rejected, similar, wrong_type)
    )


def test_resolution_merges_equal_normalized_names_across_mentions(db_engine) -> None:
    """Unmapped mentions with the same typed normalized name share one entity node."""
    catalog = AliasCatalog(SCOPE, "empty-catalog", ())
    with db_engine.begin() as connection:
        first = resolve_mention(
            connection,
            SCOPE,
            EntityMentionDraft("first", "Product", "New Product", PASSAGE, "New Product"),
            catalog,
        )
        second = resolve_mention(
            connection,
            SCOPE,
            EntityMentionDraft("second", "Product", " new   product ", PASSAGE, " new   product "),
            catalog,
        )
        node_count = connection.execute(
            text("SELECT count(*) FROM kg_node WHERE node_type='Product'")
        ).scalar_one()

    assert first.node_id == second.node_id
    assert first.canonical_key == second.canonical_key == "new product"
    assert node_count == 1


def test_resolution_merges_an_approved_canonical_name_to_its_alias_target(db_engine) -> None:
    """An approved canonical name resolves to the same target as its approved aliases."""
    catalog = AliasCatalog(
        SCOPE,
        "catalog-v1",
        (
            ApprovedAlias(
                "alias-product",
                "product shorthand",
                "product shorthand",
                "Product",
                "canonical-product",
                "Canonical Product",
                "nfkc-casefold-whitespace-v1",
            ),
        ),
    )
    with db_engine.begin() as connection:
        resolved = resolve_mention(
            connection,
            SCOPE,
            EntityMentionDraft("canonical", "Product", " canonical   product ", PASSAGE, "canonical product"),
            catalog,
        )

    assert resolved.node_id == derive_scoped_id(
        "entity", SCOPE, "Product", "canonical-product"
    )
    assert (resolved.canonical_key, resolved.canonical_name, resolved.matched_by) == (
        "canonical-product",
        "Canonical Product",
        "approved_alias",
    )


def test_materialization_commits_resolution_provenance_and_current_pointer_together(
    db_engine,
) -> None:
    """A successful extraction becomes current only with all semantic rows and provenance present."""
    previous, candidate = _seed_passage(db_engine)

    report = materialize_extraction(db_engine, PASSAGE, candidate)

    assert report.extraction_run_id == candidate
    assert report.previous_active_extraction_run_id == previous
    assert (
        report.assertions == 1
        and report.evidence_edges == 2
        and report.alias_proposals == 1
    )
    with db_engine.connect() as connection:
        state = connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one()
        run = (
            connection.execute(
                text(
                    "SELECT status,previous_active_extraction_run_id,counts FROM materialization_run"
                )
            )
            .mappings()
            .one()
        )
        proposal_row = connection.execute(
            text(
                """SELECT evidence.source_type,evidence.exact_quote,evidence.original_start,evidence.original_end,
                          evidence.extraction_run_id,evidence.prompt_hash,evidence.schema_hash,evidence.input_sha256,
                          alias.entity_type,alias.canonical_key,alias.canonical_name
                   FROM alias_evidence AS evidence
                   JOIN aliases AS alias
                     ON (alias.tenant_id,alias.project_id,alias.alias_id)=
                        (evidence.tenant_id,evidence.project_id,evidence.alias_id)
                   WHERE evidence.source_type='extraction'"""
            )
        ).one()
        assertions = connection.execute(
            text("SELECT extraction_run_id,claim,original_quote FROM assertion")
        ).all()
        evidence_edges = connection.execute(
            text(
                "SELECT assertion_id,is_evidence FROM kg_edge WHERE is_evidence ORDER BY edge_id"
            )
        ).all()
        node_refs = connection.execute(
            text(
                "SELECT count(*) FROM kg_node_source_ref WHERE tenant_id=:tenant AND project_id=:project"
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).scalar_one()
        edge_refs = connection.execute(
            text(
                "SELECT count(*) FROM kg_edge_source_ref WHERE tenant_id=:tenant AND project_id=:project"
            ),
            {"tenant": SCOPE.tenant_id, "project": SCOPE.project_id},
        ).scalar_one()
        trusted_product = connection.execute(
            text("""SELECT attributes FROM kg_node
                   WHERE node_id=:node_id AND node_type='Product'"""),
            {"node_id": derive_scoped_id("entity", SCOPE, "Product", "casgevy")},
        ).scalar_one()
        applications = connection.execute(
            text(
                "SELECT node_id,attributes FROM kg_node WHERE node_type='Application' ORDER BY node_id"
            )
        ).all()
        application_mention_node = connection.execute(text("""SELECT ref.node_id
                   FROM kg_node_mention_ref AS ref
                   JOIN entity_mention AS mention
                     ON (mention.tenant_id,mention.project_id,mention.source_id,
                         mention.source_version,mention.passage_id,mention.mention_id,
                         mention.extraction_run_id)=
                        (ref.tenant_id,ref.project_id,ref.source_id,ref.source_version,
                         ref.passage_id,ref.mention_id,ref.extraction_run_id)
                   WHERE mention.entity_type='Application'""")).scalar_one()

    assert state == ("extracted", "extracted_successfully", candidate)
    assert (
        run["status"] == "succeeded"
        and run["previous_active_extraction_run_id"] == previous
    )
    assert run["counts"]["assertions"] == 1
    assert proposal_row[0:5] == ("extraction", "CGV-X", 16, 21, candidate)
    assert all(proposal_row[index] for index in (5, 6, 7))
    assert proposal_row[8:11] == ("Product", "casgevy", "CASGEVY")
    assert assertions == [
        (
            candidate,
            "CASGEVY uses a flow assay for editing rate.",
            "CASGEVY uses a flow assay for editing rate",
        )
    ]
    assert len(evidence_edges) == 2 and all(
        row[0] is not None and row[1] for row in evidence_edges
    )
    assert node_refs >= 4 and edge_refs == 2
    assert trusted_product["canonical_key"] == "casgevy"
    assert trusted_product["alias_catalog_version"]
    assert len(applications) == 2
    assert derive_scoped_id("application", SCOPE, "casgevy-tdt") in {
        row.node_id for row in applications
    }
    assert application_mention_node == derive_scoped_id(
        "application", SCOPE, "casgevy-tdt"
    )
    assert all(
        row.attributes["profile_source_path"] == "configs/casgevy-potency.json"
        for row in applications
    )
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
            approved_alias_catalog(connection, SCOPE),
        )
    assert proposed_alias.matched_by == "new_entity"
    assert proposed_alias.node_id != derive_scoped_id(
        "entity", SCOPE, "Product", "casgevy"
    )


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
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM entity_mention WHERE extraction_run_id=:run_id"
                ),
                {"run_id": candidate},
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM assertion WHERE extraction_run_id=:run_id"),
                {"run_id": candidate},
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM alias_evidence WHERE extraction_run_id=:run_id"
                ),
                {"run_id": candidate},
            ).scalar_one()
            == 0
        )
        materialization = connection.execute(
            text("""SELECT status,previous_active_extraction_run_id,error_detail
                   FROM materialization_run WHERE extraction_run_id=:run_id"""),
            {"run_id": candidate},
        ).one()
        assert (
            connection.execute(text("SELECT count(*) FROM kg_edge")).scalar_one() == 0
        )
        assert (
            connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 1
        )

    assert state == ("extracted", "extracted_successfully", previous)
    assert materialization[0] == "failed" and materialization[1] == previous
    assert "injected" not in (materialization[2] or "")


def test_materialization_rejects_cross_project_extraction_id_even_with_colliding_local_ids(
    db_engine,
) -> None:
    """A globally colliding UUID or local IDs must not bypass full tenant/project/source identity checks."""
    previous, candidate = _seed_passage(db_engine)
    wrong_scope = ScopedPassageIdentity(
        PASSAGE.tenant_id, "other-project", *PASSAGE.as_tuple()[2:]
    )

    with pytest.raises(ValueError, match="full passage identity"):
        materialize_extraction(db_engine, wrong_scope, candidate)

    with db_engine.connect() as connection:
        state = connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one()
        assert (
            connection.execute(
                text("SELECT count(*) FROM materialization_run")
            ).scalar_one()
            == 0
        )
    assert state == ("extracted", "extracted_successfully", previous)


def test_materialization_requires_the_exact_current_processing_row(db_engine) -> None:
    """A matching immutable run without its exact current-state row must not report false activation."""
    _previous, candidate = _seed_passage(db_engine)
    with db_engine.begin() as connection:
        connection.execute(text("DELETE FROM passage_processing"))

    with pytest.raises(ValueError, match="processing state"):
        materialize_extraction(db_engine, PASSAGE, candidate)

    with db_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM materialization_run")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(text("SELECT count(*) FROM assertion")).scalar_one() == 0
        )


def test_process_control_interrupt_rolls_back_without_becoming_a_materialization_failure(
    db_engine,
) -> None:
    """A process-control signal must roll back the transaction without being rewritten as an application error."""
    previous, candidate = _seed_passage(db_engine)

    def interrupt(point: str) -> None:
        if point == "after_resolution":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        materialize_extraction(
            db_engine, PASSAGE, candidate, failure_injector=interrupt
        )

    with db_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM materialization_run")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(text("SELECT count(*) FROM entity_mention")).scalar_one()
            == 0
        )
        assert (
            connection.execute(text("SELECT count(*) FROM kg_node")).scalar_one() == 1
        )
        assert connection.execute(
            text("SELECT state,reason,active_extraction_run_id FROM passage_processing")
        ).one() == ("extracted", "extracted_successfully", previous)
