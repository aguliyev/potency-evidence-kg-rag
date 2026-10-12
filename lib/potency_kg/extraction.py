"""Bundle-scoped dynamic extraction and complete-response grounding."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .contracts import ALLOWED_RELATION_ENDPOINTS, Scope, ScopedPassageIdentity
from .ids import canonical_json, derive_extraction_version
from .llm_client import (
    GenerationUnselectedTypeError,
    StructuredExtractionClient,
)
from .processing_policy import BUNDLE_ORDER
from .prompt_resources.extraction import RETRY_INSTRUCTION, assemble_prompt
from .prompt_resources.jev_bundles import BUNDLE_TO_TYPES
from .redactions import (
    RedactionSubstitution,
    restore_model_quote,
    substitute_redactions,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssertionRecord(_StrictModel):
    local_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    supporting_quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)
    chronology_state: str = "reported"
    review_state: str = "unreviewed"
    authority: str | None = None
    status: str | None = None


class EvidenceRelationshipRecord(_StrictModel):
    local_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    source_local_id: str = Field(min_length=1)
    target_local_id: str = Field(min_length=1)
    assertion_local_id: str = Field(min_length=1)
    attributes: dict[str, object] = Field(default_factory=dict)


class SourceReferenceRecord(_StrictModel):
    local_id: str = Field(min_length=1)
    record_local_id: str = Field(min_length=1)
    supporting_quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)


_ENTITY_FIELDS: dict[str, dict[str, tuple[object, object]]] = {
    "Product": {"name": (str, ...)},
    "Application": {"application_id": (str, ...)},
    "Indication": {"name": (str, ...)},
    "AliasCandidate": {"candidate": (str, ...), "target_local_id": (str, ...)},
    "PotencyStrategy": {"description": (str, ...)},
    "Assay": {"name": (str, ...), "role": (str | None, None)},
    "MeasuredAttribute": {"name": (str, ...)},
    "RegulatoryInteraction": {"chronology_state": (str, ...), "description": (str | None, None)},
    "ReviewConclusion": {"conclusion": (str, ...)},
    "RedactionMention": {"model_start": (int, ...), "model_end": (int, ...)},
}


@dataclass(frozen=True)
class ExtractionReport:
    eligible_passages: int
    succeeded: int
    failed: int
    extraction_run_ids: tuple[UUID, ...]


def selected_entity_types(selected_bundles: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(selected_bundles)
    unknown = set(requested) - set(BUNDLE_ORDER)
    if unknown:
        raise ValueError(f"unknown extraction bundles: {sorted(unknown)}")
    requested_set = set(requested)
    merged: list[str] = []
    for bundle in BUNDLE_ORDER:
        if bundle not in requested_set:
            continue
        for entity_type in BUNDLE_TO_TYPES[bundle]:
            if entity_type not in merged:
                merged.append(entity_type)
    return tuple(merged)


def _entity_model(entity_type: str) -> type[BaseModel]:
    base_fields: dict[str, tuple[object, object]] = {
        "type": (Literal[entity_type], entity_type),
        "local_id": (str, ...),
    }
    if entity_type != "RedactionMention":
        base_fields.update({"supporting_quote": (str, ...), "occurrence_index": (int | None, None)})
    base_fields.update(_ENTITY_FIELDS[entity_type])
    return create_model(f"{entity_type}Extraction", __base__=_StrictModel, **base_fields)


def build_response_model(selected_bundles: Iterable[str]) -> type[BaseModel]:
    requested = tuple(selected_bundles)
    unknown = set(requested) - set(BUNDLE_ORDER)
    if unknown:
        raise ValueError(f"unknown extraction bundles: {sorted(unknown)}")
    requested_set = set(requested)
    bundles = tuple(bundle for bundle in BUNDLE_ORDER if bundle in requested_set)
    entity_types = selected_entity_types(bundles)
    if not entity_types:
        raise ValueError("an extraction response needs at least one selected bundle")
    models = tuple(_entity_model(entity_type) for entity_type in entity_types)
    if len(models) == 1:
        entity_annotation: object = models[0]
    else:
        entity_annotation = Annotated[Union[models], Field(discriminator="type")]  # type: ignore[valid-type]
    digest = sha256(canonical_json({"bundles": bundles, "types": entity_types}).encode()).hexdigest()[:12]
    response_model = create_model(
        f"BundleExtraction_{digest}",
        __base__=_StrictModel,
        entities=(list[entity_annotation], Field(default_factory=list)),  # type: ignore[valid-type]
        assertions=(list[AssertionRecord], Field(default_factory=list)),
        relationships=(list[EvidenceRelationshipRecord], Field(default_factory=list)),
        source_refs=(list[SourceReferenceRecord], Field(default_factory=list)),
    )
    setattr(response_model, "__selected_bundles__", bundles)
    setattr(response_model, "__allowed_entity_types__", entity_types)
    return response_model


def _ground_quote(record: Mapping[str, object], substitution: RedactionSubstitution) -> dict[str, object]:
    grounded = dict(record)
    span = restore_model_quote(
        substitution,
        str(record["supporting_quote"]),
        record.get("occurrence_index") if isinstance(record.get("occurrence_index"), int) else None,
    )
    grounded.update(
        {
            "model_start": substitution.model_text.find(str(record["supporting_quote"]))
            if record.get("occurrence_index") in (None, 0)
            else [i for i in range(len(substitution.model_text)) if substitution.model_text.startswith(str(record["supporting_quote"]), i)][int(record["occurrence_index"])],
            "model_end": 0,
            "original_start": span.start,
            "original_end": span.end,
            "original_quote": span.text,
        }
    )
    grounded["model_end"] = int(grounded["model_start"]) + len(str(record["supporting_quote"]))
    return grounded


def validate_complete_response(
    response: BaseModel,
    substitution: RedactionSubstitution,
    selected_bundles: Iterable[str],
) -> dict[str, object]:
    """Validate the complete passage response before any draft row can be persisted."""
    allowed = set(selected_entity_types(selected_bundles))
    payload = response.model_dump(mode="json")
    entities = payload.get("entities", [])
    assertions = payload.get("assertions", [])
    relationships = payload.get("relationships", [])
    source_refs = payload.get("source_refs", [])
    if not all(isinstance(group, list) for group in (entities, assertions, relationships, source_refs)):
        raise ValueError("complete extraction response lists are required")
    all_records = [*entities, *assertions, *relationships, *source_refs]
    local_ids = [record.get("local_id") for record in all_records]
    if any(not isinstance(local_id, str) or not local_id.strip() for local_id in local_ids) or len(set(local_ids)) != len(local_ids):
        raise ValueError("passage-local IDs must be nonempty and unique")

    record_types: dict[str, str] = {}
    grounded_entities: list[dict[str, object]] = []
    for entity in entities:
        entity_type = str(entity.get("type"))
        if entity_type not in allowed:
            raise ValueError(f"unselected entity type: {entity_type}")
        local_id = str(entity["local_id"])
        record_types[local_id] = entity_type
        if entity_type == "RedactionMention":
            start, end = int(entity["model_start"]), int(entity["model_end"])
            matching = next((span for span in substitution.spans if (span.model_start, span.model_end) == (start, end)), None)
            if matching is None:
                raise ValueError("redaction mention must match an exact substituted token")
            grounded = dict(entity)
            grounded.update(
                original_start=matching.original_start,
                original_end=matching.original_end,
                original_quote=substitution.original_text[matching.original_start:matching.original_end],
            )
        else:
            grounded = _ground_quote(entity, substitution)
        grounded_entities.append(grounded)

    grounded_assertions: list[dict[str, object]] = []
    assertion_ids: set[str] = set()
    for assertion in assertions:
        local_id = str(assertion["local_id"])
        assertion_ids.add(local_id)
        record_types[local_id] = "Assertion"
        claim = str(assertion["claim"])
        if not claim.strip() or "\n" in claim:
            raise ValueError("assertions must be atomic single records")
        grounded_assertions.append(_ground_quote(assertion, substitution))

    for entity in entities:
        if (
            entity.get("type") == "AliasCandidate"
            and record_types.get(str(entity.get("target_local_id"))) not in {"Product", "Assay"}
        ):
            raise ValueError("alias candidate contains an unresolved passage-local Product or Assay reference")

    grounded_relationships: list[dict[str, object]] = []
    for relationship in relationships:
        assertion_id = relationship.get("assertion_local_id")
        if assertion_id not in assertion_ids:
            raise ValueError("evidence relationship contains an unresolved passage-local assertion reference")
        source_id, target_id = relationship.get("source_local_id"), relationship.get("target_local_id")
        source_type = "SourcePassage" if source_id == "source_passage" else record_types.get(str(source_id))
        target_type = "SourcePassage" if target_id == "source_passage" else record_types.get(str(target_id))
        if source_type is None or target_type is None:
            raise ValueError("evidence relationship contains an unresolved passage-local endpoint")
        relation_type = str(relationship.get("relation_type"))
        if (source_type, target_type) not in ALLOWED_RELATION_ENDPOINTS.get(relation_type, frozenset()):
            raise ValueError("evidence relationship endpoints are not allowlisted")
        enriched = dict(relationship)
        enriched.update(source_type=source_type, target_type=target_type)
        grounded_relationships.append(enriched)

    grounded_refs: list[dict[str, object]] = []
    resolvable = set(record_types) | {
        str(relationship["local_id"]) for relationship in grounded_relationships
    }
    for source_ref in source_refs:
        if source_ref.get("record_local_id") not in resolvable:
            raise ValueError("source reference contains an unresolved passage-local record reference")
        grounded_refs.append(_ground_quote(source_ref, substitution))
    return {
        "entities": grounded_entities,
        "assertions": grounded_assertions,
        "relationships": grounded_relationships,
        "source_refs": grounded_refs,
    }


def _unselected_error(error: BaseException, allowed: set[str]) -> bool:
    if isinstance(error, GenerationUnselectedTypeError):
        return True
    if isinstance(error, ValidationError):
        for item in error.errors(include_url=False):
            candidate = item.get("input")
            if isinstance(candidate, dict) and isinstance(candidate.get("type"), str) and candidate["type"] not in allowed:
                return True
            context = item.get("ctx")
            if isinstance(context, dict) and isinstance(context.get("tag"), str) and context["tag"] not in allowed:
                return True
    return False


def _selected_rows(engine: Engine, scope: Scope) -> list[dict[str, object]]:
    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT p.source_id,p.source_version,p.passage_id,p.text,
                           pp.active_jev_selection_id,s.selected_bundles,
                           r.profile_name,r.profile_aliases,r.profile_source_sha256,r.profile_version,r.alias_catalog_version
                    FROM passage_processing AS pp
                    JOIN source_passage AS p
                      ON (p.tenant_id,p.project_id,p.source_id,p.source_version,p.passage_id)=
                         (pp.tenant_id,pp.project_id,pp.source_id,pp.source_version,pp.passage_id)
                    JOIN source_head AS h
                      ON (h.tenant_id,h.project_id,h.source_id,h.source_version)=
                         (p.tenant_id,p.project_id,p.source_id,p.source_version)
                    JOIN jev_selection AS s
                      ON (s.selection_id,s.tenant_id,s.project_id,s.source_id,s.source_version,s.passage_id)=
                         (pp.active_jev_selection_id,pp.tenant_id,pp.project_id,pp.source_id,pp.source_version,pp.passage_id)
                    JOIN jev_response AS r
                      ON (r.response_id,r.tenant_id,r.project_id,r.source_id,r.source_version,r.passage_id)=
                         (s.response_id,s.tenant_id,s.project_id,s.source_id,s.source_version,s.passage_id)
                    WHERE pp.tenant_id=:tenant AND pp.project_id=:project AND pp.state='selected'
                      AND jsonb_array_length(s.selected_bundles) > 0
                    ORDER BY p.source_id,p.source_version,p.passage_id
                    """
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).mappings()
        ]


@dataclass(frozen=True)
class _Outcome:
    row: Mapping[str, object]
    run_id: UUID
    response: dict[str, object] | None
    extraction_version: str
    prompt_hash: str
    schema_hash: str
    input_sha256: str
    attempts: int
    usage: Mapping[str, int | float | None]
    failure_code: str | None


@dataclass(frozen=True)
class _PreparedExtraction:
    response_model: type[BaseModel]
    prompt: str
    prompt_hash: str
    schema_hash: str
    extraction_version: str
    substitution: RedactionSubstitution
    input_sha256: str


def _prepare_extraction(
    row: Mapping[str, object], scope: Scope, client: StructuredExtractionClient
) -> _PreparedExtraction:
    passage = ScopedPassageIdentity(
        scope.tenant_id,
        scope.project_id,
        str(row["source_id"]),
        str(row["source_version"]),
        str(row["passage_id"]),
    )
    bundles = tuple(row["selected_bundles"])
    response_model = build_response_model(bundles)
    prompt = assemble_prompt(bundles)
    prompt_hash = sha256(prompt.encode()).hexdigest()
    schema_hash = sha256(canonical_json(response_model.model_json_schema()).encode()).hexdigest()
    profile = {
        "name": row["profile_name"],
        "aliases": row["profile_aliases"] or [],
        "alias_catalog_version": row["alias_catalog_version"],
        "descriptor_sha256": row["profile_source_sha256"],
        "profile_version": row["profile_version"],
    }
    extraction_version = derive_extraction_version(
        model=client.identity.model,
        instructor_mode=client.identity.instructor_mode,
        selected_bundles=bundles,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        normalization="redaction-v1",
        profile=profile,
    )
    substitution = substitute_redactions(passage, str(row["text"]))
    input_sha256 = sha256(substitution.model_text.encode()).hexdigest()
    return _PreparedExtraction(
        response_model,
        prompt,
        prompt_hash,
        schema_hash,
        extraction_version,
        substitution,
        input_sha256,
    )


def _reusable_run_id(
    engine: Engine,
    scope: Scope,
    row: Mapping[str, object],
    client: StructuredExtractionClient,
) -> UUID | None:
    prepared = _prepare_extraction(row, scope, client)
    with engine.connect() as connection:
        return connection.execute(
            text(
                """SELECT extraction_run_id FROM llm_extraction_run
                   WHERE tenant_id=:tenant AND project_id=:project
                     AND source_id=:source_id AND source_version=:source_version
                     AND passage_id=:passage_id AND status='succeeded' AND response IS NOT NULL
                     AND extraction_version=:extraction_version
                     AND model=:model AND instructor_mode=:instructor_mode
                     AND prompt_hash=:prompt_hash AND schema_hash=:schema_hash
                     AND normalization='redaction-v1'
                     AND selected_bundles=CAST(:bundles AS jsonb)
                     AND input_sha256=:input_sha256
                     AND profile_name IS NOT DISTINCT FROM :profile_name
                     AND profile_aliases=CAST(:profile_aliases AS jsonb)
                     AND profile_source_sha256 IS NOT DISTINCT FROM :profile_source_sha256
                     AND profile_version=:profile_version
                     AND alias_catalog_version=:alias_catalog_version
                   ORDER BY created_at DESC,extraction_run_id DESC LIMIT 1"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "source_id": row["source_id"],
                "source_version": row["source_version"],
                "passage_id": row["passage_id"],
                "extraction_version": prepared.extraction_version,
                "model": client.identity.model,
                "instructor_mode": client.identity.instructor_mode,
                "prompt_hash": prepared.prompt_hash,
                "schema_hash": prepared.schema_hash,
                "bundles": json.dumps(row["selected_bundles"]),
                "input_sha256": prepared.input_sha256,
                "profile_name": row["profile_name"],
                "profile_aliases": json.dumps(row["profile_aliases"] or []),
                "profile_source_sha256": row["profile_source_sha256"],
                "profile_version": row["profile_version"],
                "alias_catalog_version": row["alias_catalog_version"],
            },
        ).scalar_one_or_none()


def _activate_reused_run(
    engine: Engine,
    scope: Scope,
    row: Mapping[str, object],
    run_id: UUID,
) -> None:
    with engine.begin() as connection:
        result = connection.execute(
            text(
                """UPDATE passage_processing
                   SET state='selected',reason='extraction_validated',
                       validated_extraction_run_id=:run_id,updated_at=now()
                   WHERE tenant_id=:tenant AND project_id=:project
                     AND source_id=:source_id AND source_version=:source_version
                     AND passage_id=:passage_id AND state='selected'
                     AND active_jev_selection_id=:selection_id"""
            ),
            {
                "run_id": run_id,
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "source_id": row["source_id"],
                "source_version": row["source_version"],
                "passage_id": row["passage_id"],
                "selection_id": row["active_jev_selection_id"],
            },
        )
        if result.rowcount != 1:
            raise RuntimeError("active Jev selection changed before extraction reuse activation")


def _request_usage(client: StructuredExtractionClient) -> dict[str, int | float | None]:
    accessor = getattr(client, "usage_for_current_thread", None)
    raw = accessor() if callable(accessor) else getattr(client, "last_usage", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): value
        for name, value in raw.items()
        if (isinstance(value, (int, float)) and not isinstance(value, bool)) or value is None
    }


def _request_attempts(client: StructuredExtractionClient) -> int:
    accessor = getattr(client, "attempts_for_current_thread", None)
    raw = accessor() if callable(accessor) else 1
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 1


def _merge_usage(
    accumulated: Mapping[str, int | float | None],
    current: Mapping[str, int | float | None],
) -> dict[str, int | float | None]:
    merged = dict(accumulated)
    for name, value in current.items():
        prior = merged.get(name)
        if isinstance(prior, (int, float)) and isinstance(value, (int, float)):
            merged[name] = prior + value
        else:
            merged[name] = value
    return merged


def _extract_one(row: Mapping[str, object], scope: Scope, client: StructuredExtractionClient) -> _Outcome:
    bundles = tuple(row["selected_bundles"])
    prepared = _prepare_extraction(row, scope, client)
    response_model = prepared.response_model
    prompt = prepared.prompt
    substitution = prepared.substitution
    attempts = 0
    usage: dict[str, int | float | None] = {}
    first_metadata_recorded = False
    try:
        parsed = client.extract(substitution.model_text, response_model, prompt)
        attempts += _request_attempts(client)
        usage = _merge_usage(usage, _request_usage(client))
        first_metadata_recorded = True
        parsed = response_model.model_validate(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed)
        grounded = validate_complete_response(parsed, substitution, bundles)
        return _Outcome(row, uuid4(), grounded, prepared.extraction_version, prepared.prompt_hash, prepared.schema_hash, prepared.input_sha256, attempts, usage, None)
    except Exception as first_error:
        if not first_metadata_recorded:
            attempts += _request_attempts(client)
            usage = _merge_usage(usage, _request_usage(client))
        if _unselected_error(first_error, set(selected_entity_types(bundles))):
            emphasized = f"{prompt}\n{RETRY_INSTRUCTION}\nUse only these entity types: {', '.join(selected_entity_types(bundles))}."
            second_metadata_recorded = False
            try:
                parsed = client.extract(substitution.model_text, response_model, emphasized)
                attempts += _request_attempts(client)
                usage = _merge_usage(usage, _request_usage(client))
                second_metadata_recorded = True
                parsed = response_model.model_validate(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed)
                grounded = validate_complete_response(parsed, substitution, bundles)
                return _Outcome(row, uuid4(), grounded, prepared.extraction_version, prepared.prompt_hash, prepared.schema_hash, prepared.input_sha256, attempts, usage, None)
            except Exception as second_error:
                if not second_metadata_recorded:
                    attempts += _request_attempts(client)
                    usage = _merge_usage(usage, _request_usage(client))
                code = "unselected_type" if _unselected_error(second_error, set(selected_entity_types(bundles))) else "validation_failed"
                return _Outcome(row, uuid4(), None, prepared.extraction_version, prepared.prompt_hash, prepared.schema_hash, prepared.input_sha256, attempts, usage, code)
        return _Outcome(row, uuid4(), None, prepared.extraction_version, prepared.prompt_hash, prepared.schema_hash, prepared.input_sha256, attempts, usage, "validation_failed")


def _persist_outcome(engine: Engine, scope: Scope, client: StructuredExtractionClient, outcome: _Outcome) -> None:
    row = outcome.row
    params = {
        "run_id": outcome.run_id,
        "tenant": scope.tenant_id,
        "project": scope.project_id,
        "source_id": row["source_id"],
        "source_version": row["source_version"],
        "passage_id": row["passage_id"],
        "extraction_version": outcome.extraction_version,
        "model": client.identity.model,
        "origin": client.identity.sanitized_origin,
        "credential": client.identity.credential_env,
        "mode": client.identity.instructor_mode,
        "bundles": json.dumps(row["selected_bundles"]),
        "prompt_hash": outcome.prompt_hash,
        "schema_hash": outcome.schema_hash,
        "input_sha256": outcome.input_sha256,
        "profile_name": row["profile_name"],
        "profile_aliases": json.dumps(row["profile_aliases"] or []),
        "profile_sha": row["profile_source_sha256"],
        "profile_version": row["profile_version"],
        "alias_catalog_version": row["alias_catalog_version"],
        "selection_id": row["active_jev_selection_id"],
        "usage": json.dumps(dict(outcome.usage)),
        "attempts": outcome.attempts,
        "response": json.dumps(outcome.response) if outcome.response is not None else None,
    }
    with engine.begin() as connection:
        if outcome.response is not None:
            connection.execute(
                text(
                    """INSERT INTO llm_extraction_run
                       (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,extraction_version,
                        model,sanitized_origin,credential_env,instructor_mode,selected_bundles,prompt_hash,schema_hash,
                        normalization,input_sha256,profile_name,profile_aliases,profile_source_sha256,profile_version,alias_catalog_version,
                        usage,attempts,status,response)
                       VALUES (:run_id,:tenant,:project,:source_id,:source_version,:passage_id,:extraction_version,
                        :model,:origin,:credential,:mode,CAST(:bundles AS jsonb),:prompt_hash,:schema_hash,
                        'redaction-v1',:input_sha256,:profile_name,CAST(:profile_aliases AS jsonb),:profile_sha,:profile_version,:alias_catalog_version,
                        CAST(:usage AS jsonb),:attempts,'succeeded',CAST(:response AS jsonb))"""
                ),
                params,
            )
            activated = connection.execute(
                text("""UPDATE passage_processing
                        SET state='selected',reason='extraction_validated',
                            validated_extraction_run_id=:run_id,updated_at=now()
                        WHERE (tenant_id,project_id,source_id,source_version,passage_id)=
                              (:tenant,:project,:source_id,:source_version,:passage_id)
                          AND active_jev_selection_id=:selection_id"""),
                params,
            )
            if activated.rowcount != 1:
                raise RuntimeError("active Jev selection changed before extraction activation")
        else:
            connection.execute(
                text(
                    """INSERT INTO llm_extraction_run
                       (extraction_run_id,tenant_id,project_id,source_id,source_version,passage_id,extraction_version,
                        model,sanitized_origin,credential_env,instructor_mode,selected_bundles,prompt_hash,schema_hash,
                        normalization,input_sha256,profile_name,profile_aliases,profile_source_sha256,profile_version,alias_catalog_version,
                        usage,attempts,status,error_detail)
                       VALUES (:run_id,:tenant,:project,:source_id,:source_version,:passage_id,:extraction_version,
                        :model,:origin,:credential,:mode,CAST(:bundles AS jsonb),:prompt_hash,:schema_hash,
                        'redaction-v1',:input_sha256,:profile_name,CAST(:profile_aliases AS jsonb),:profile_sha,:profile_version,:alias_catalog_version,
                        CAST(:usage AS jsonb),:attempts,'failed','extraction validation failed')"""
                ),
                params,
            )
            connection.execute(
                text(
                    """INSERT INTO validation_failure
                       (validation_failure_id,tenant_id,project_id,stage,immutable_run_id,code,json_path,detail)
                       VALUES (:failure_id,:tenant,:project,'extraction',:run_id,:code,NULL,'complete extraction response rejected')"""
                ),
                {**params, "failure_id": uuid4(), "code": outcome.failure_code},
            )
            connection.execute(
                text("""UPDATE passage_processing
                        SET state='failed',reason='extraction_failed',active_extraction_run_id=NULL,
                            validated_extraction_run_id=NULL,updated_at=now()
                        WHERE (tenant_id,project_id,source_id,source_version,passage_id)=
                              (:tenant,:project,:source_id,:source_version,:passage_id)
                          AND active_jev_selection_id=:selection_id"""),
                params,
            )


def extract_selected_passages(
    engine: Engine,
    scope: Scope,
    client: StructuredExtractionClient,
    max_concurrency: int,
) -> ExtractionReport:
    """Extract selected passages concurrently, validating each response atomically before persistence."""
    rows = _selected_rows(engine, scope)
    reused: dict[tuple[str, str, str], UUID] = {}
    pending: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))
        run_id = _reusable_run_id(engine, scope, row, client)
        if run_id is None:
            pending.append(row)
        else:
            _activate_reused_run(engine, scope, row, run_id)
            reused[key] = run_id
    outcomes: list[_Outcome] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
        futures = {executor.submit(_extract_one, row, scope, client): row for row in pending}
        for future in as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda value: (str(value.row["source_id"]), str(value.row["source_version"]), str(value.row["passage_id"])))
    for outcome in outcomes:
        _persist_outcome(engine, scope, client, outcome)
    outcome_by_key = {
        (str(outcome.row["source_id"]), str(outcome.row["source_version"]), str(outcome.row["passage_id"])): outcome
        for outcome in outcomes
    }
    run_ids = tuple(
        reused[key] if key in reused else outcome_by_key[key].run_id
        for row in rows
        for key in [(str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))]
    )
    return ExtractionReport(
        eligible_passages=len(rows),
        succeeded=len(reused) + sum(outcome.response is not None for outcome in outcomes),
        failed=sum(outcome.response is None for outcome in outcomes),
        extraction_run_ids=run_ids,
    )
