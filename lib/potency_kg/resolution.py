"""Trusted profile resolution and atomic relational semantic materialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .alias_discovery import AliasProposal, record_alias_proposal
from .aliases import AliasCatalog, approved_alias_catalog, lock_alias_catalog, normalize_alias
from .contracts import Scope, ScopedPassageIdentity
from .extraction import build_response_model
from .ids import canonical_json, derive_extraction_version, derive_scoped_id
from .project_profiles import (
    trusted_applications_for_profile,
    trusted_descriptor_path_for_profile,
)
from .prompt_resources.extraction import assemble_prompt


@dataclass(frozen=True)
class ApplicationBootstrapReport:
    application_count: int


@dataclass(frozen=True)
class EntityMentionDraft:
    local_id: str
    entity_type: str
    text: str
    source_ref: ScopedPassageIdentity
    original_quote: str
    extraction_run_id: UUID | None = None
    extraction_version: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.local_id.strip()
            or not self.entity_type.strip()
            or not self.text.strip()
        ):
            raise ValueError(
                "entity mention needs nonempty local identity, type, and text"
            )
        if not self.original_quote:
            raise ValueError("entity mention needs an exact original quote")


@dataclass(frozen=True)
class ResolvedEntity:
    node_id: str
    node_type: str
    canonical_key: str
    canonical_name: str
    matched_by: str


@dataclass(frozen=True)
class MaterializationReport:
    materialization_run_id: UUID
    extraction_run_id: UUID
    previous_active_extraction_run_id: UUID | None
    mentions: int
    alias_proposals: int
    assertions: int
    evidence_edges: int


_ENTITY_VALUE_FIELDS: dict[str, str] = {
    "Product": "name",
    "Application": "application_id",
    "Indication": "name",
    "PotencyStrategy": "description",
    "Assay": "name",
    "MeasuredAttribute": "name",
    "RegulatoryInteraction": "description",
    "ReviewConclusion": "conclusion",
}


def _catalog_entries(catalog: AliasCatalog) -> list[dict[str, str]]:
    return [
        {
            "alias_id": entry.alias_id,
            "alias_value": entry.alias_value,
            "normalized_alias": entry.normalized_alias,
            "entity_type": entry.entity_type,
            "canonical_key": entry.canonical_key,
            "canonical_name": entry.canonical_name,
            "normalization_version": entry.normalization_version,
        }
        for entry in catalog.entries
    ]


def bootstrap_profile_applications(
    connection: Connection,
    scope: Scope,
    *,
    profile_name: str,
    profile_version: str,
    profile_source_sha256: str,
) -> ApplicationBootstrapReport:
    """Create the profile's stable Application anchors without consulting aliases."""
    applications = trusted_applications_for_profile(profile_name)
    for application_id, canonical_id in applications.items():
        if not application_id.strip() or not canonical_id.strip():
            raise ValueError("trusted profile applications must be nonempty")
        node_id = derive_scoped_id("application", scope, canonical_id)
        attributes = {
            "canonical_id": canonical_id,
            "application_id": application_id,
            "profile_name": profile_name,
            "profile_version": profile_version,
            "profile_source_path": trusted_descriptor_path_for_profile(profile_name),
            "profile_source_sha256": profile_source_sha256,
            "system_generated": True,
        }
        connection.execute(
            text(
                """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
                   VALUES (:tenant,:project,:node_id,'Application',CAST(:attributes AS jsonb))
                   ON CONFLICT (tenant_id,project_id,node_id) DO UPDATE
                   SET attributes=EXCLUDED.attributes
                   WHERE kg_node.node_type='Application'"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "node_id": node_id,
                "attributes": json.dumps(attributes),
            },
        )
    return ApplicationBootstrapReport(len(applications))


def _resolve_profile_application(
    scope: Scope,
    mention: EntityMentionDraft,
    profile_name: str,
) -> ResolvedEntity | None:
    if mention.entity_type != "Application":
        return None
    canonical_id = trusted_applications_for_profile(profile_name).get(
        mention.text.strip()
    )
    if canonical_id is None:
        return None
    return ResolvedEntity(
        derive_scoped_id("application", scope, canonical_id),
        "Application",
        canonical_id,
        canonical_id,
        "trusted_profile_application",
    )


def resolve_mention(
    connection: Connection,
    scope: Scope,
    mention: EntityMentionDraft,
    catalog: AliasCatalog,
) -> ResolvedEntity:
    """Resolve approved aliases, canonical names, or deterministic normalized names."""
    if mention.source_ref.scope != scope or catalog.scope != scope:
        raise ValueError(
            "entity resolution requires matching full tenant/project scope"
        )
    normalized_name = normalize_alias(mention.text)
    alias_matches = tuple(
        entry
        for entry in catalog.entries
        if entry.entity_type == mention.entity_type
        and entry.normalized_alias == normalized_name
    )
    match = alias_matches[0] if len(alias_matches) == 1 else None
    if match is None and alias_matches:
        alias_targets = {(entry.entity_type, entry.canonical_key) for entry in alias_matches}
        if len(alias_targets) == 1:
            match = alias_matches[0]
    if match is None:
        canonical_matches = tuple(
            entry
            for entry in catalog.entries
            if entry.entity_type == mention.entity_type
            and normalize_alias(entry.canonical_name) == normalized_name
        )
        canonical_targets = {
            (entry.entity_type, entry.canonical_key) for entry in canonical_matches
        }
        if len(canonical_targets) == 1:
            match = canonical_matches[0]
    if match is not None:
        node_id = derive_scoped_id(
            "entity", scope, match.entity_type, match.canonical_key
        )
        attributes = {
            "canonical_key": match.canonical_key,
            "canonical_name": match.canonical_name,
            "alias_id": match.alias_id,
            "alias_catalog_version": catalog.version,
            "normalization_version": match.normalization_version,
            "system_generated": True,
        }
        connection.execute(
            text(
                """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
            VALUES (:tenant,:project,:node_id,:node_type,CAST(:attributes AS jsonb))
            ON CONFLICT (tenant_id,project_id,node_id) DO UPDATE SET attributes=EXCLUDED.attributes"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "node_id": node_id,
                "node_type": match.entity_type,
                "attributes": json.dumps(attributes),
            },
        )
        return ResolvedEntity(
            node_id,
            match.entity_type,
            match.canonical_key,
            match.canonical_name,
            "approved_alias",
        )

    node_id = derive_scoped_id(
        "entity",
        scope,
        mention.entity_type,
        normalized_name,
    )
    attributes = {
        "canonical_name": mention.text,
        "canonical_key": normalized_name,
        "normalized_name": normalized_name,
        "extraction_run_id": (
            str(mention.extraction_run_id) if mention.extraction_run_id else None
        ),
        "extraction_version": mention.extraction_version,
        **dict(mention.attributes),
    }
    connection.execute(
        text("""INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
               VALUES (:tenant,:project,:node_id,:node_type,CAST(:attributes AS jsonb))
               ON CONFLICT (tenant_id,project_id,node_id) DO NOTHING"""),
        {
            "tenant": scope.tenant_id,
            "project": scope.project_id,
            "node_id": node_id,
            "node_type": mention.entity_type,
            "attributes": json.dumps(attributes),
        },
    )
    return ResolvedEntity(
        node_id,
        mention.entity_type,
        normalize_alias(mention.text),
        mention.text,
        "new_entity",
    )


def _params(passage: ScopedPassageIdentity) -> dict[str, object]:
    return dict(
        zip(
            ("tenant", "project", "source_id", "source_version", "passage_id"),
            passage.as_tuple(),
            strict=True,
        )
    )


def _matching_active_extraction(
    connection: Connection,
    passage: ScopedPassageIdentity,
    extraction_run_id: UUID,
) -> Mapping[str, object] | None:
    row = (
        connection.execute(
            text(
                """SELECT run.extraction_version,run.model,run.instructor_mode,
                      run.selected_bundles,run.prompt_hash,run.schema_hash,run.normalization,
                      run.input_sha256,run.profile_name,run.profile_aliases,
                      run.profile_source_sha256,run.profile_version,run.alias_catalog_version,run.response,
                      selection.selected_bundles AS active_bundles,
                      response.input_sha256 AS active_input_sha256,
                      response.profile_name AS active_profile_name,
                      response.profile_aliases AS active_profile_aliases,
                      response.profile_source_sha256 AS active_profile_source_sha256,
                      response.profile_version AS active_profile_version,
                      response.alias_catalog_version AS active_alias_catalog_version,
                      response.normalization AS active_normalization,
                      processing.validated_extraction_run_id
               FROM passage_processing AS processing
               JOIN jev_selection AS selection
                 ON (selection.selection_id,selection.tenant_id,selection.project_id,
                     selection.source_id,selection.source_version,selection.passage_id)=
                    (processing.active_jev_selection_id,processing.tenant_id,processing.project_id,
                     processing.source_id,processing.source_version,processing.passage_id)
               JOIN jev_response AS response
                 ON (response.response_id,response.tenant_id,response.project_id,response.source_id,
                     response.source_version,response.passage_id)=
                    (selection.response_id,selection.tenant_id,selection.project_id,selection.source_id,
                     selection.source_version,selection.passage_id)
               JOIN llm_extraction_run AS run
                 ON (run.tenant_id,run.project_id,run.source_id,run.source_version,run.passage_id)=
                    (processing.tenant_id,processing.project_id,processing.source_id,
                     processing.source_version,processing.passage_id)
               WHERE processing.tenant_id=:tenant AND processing.project_id=:project
                 AND processing.source_id=:source_id AND processing.source_version=:source_version
                 AND processing.passage_id=:passage_id
                 AND run.extraction_run_id=:extraction_run_id AND run.status='succeeded'"""
            ),
            {**_params(passage), "extraction_run_id": extraction_run_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    if (
        row["validated_extraction_run_id"] != extraction_run_id
        or row["normalization"] != row["active_normalization"]
        or list(row["selected_bundles"] or []) != list(row["active_bundles"] or [])
        or row["input_sha256"] != row["active_input_sha256"]
        or row["profile_name"] != row["active_profile_name"]
        or list(row["profile_aliases"] or [])
        != list(row["active_profile_aliases"] or [])
        or row["profile_source_sha256"] != row["active_profile_source_sha256"]
        or row["profile_version"] != row["active_profile_version"]
        or row["alias_catalog_version"] != row["active_alias_catalog_version"]
    ):
        return None
    profile = {
        "name": row["profile_name"],
        "aliases": list(row["profile_aliases"] or []),
        "alias_catalog_version": row["alias_catalog_version"],
        "descriptor_sha256": row["profile_source_sha256"],
        "profile_version": row["profile_version"],
    }
    active_bundles = tuple(row["active_bundles"] or ())
    current_prompt_hash = sha256(assemble_prompt(active_bundles).encode()).hexdigest()
    current_schema_hash = sha256(
        canonical_json(
            build_response_model(active_bundles).model_json_schema()
        ).encode()
    ).hexdigest()
    if (
        row["prompt_hash"] != current_prompt_hash
        or row["schema_hash"] != current_schema_hash
    ):
        return None
    expected_version = derive_extraction_version(
        model=str(row["model"]),
        instructor_mode=str(row["instructor_mode"]),
        selected_bundles=active_bundles,
        prompt_hash=current_prompt_hash,
        schema_hash=current_schema_hash,
        normalization=str(row["active_normalization"]),
        profile=profile,
    )
    return row if row["extraction_version"] == expected_version else None


def _source_ref(
    connection: Connection,
    table: str,
    identity_column: str,
    identity: str,
    passage: ScopedPassageIdentity,
) -> None:
    connection.execute(
        text(f"""INSERT INTO {table}
                 (tenant_id,project_id,{identity_column},source_id,source_version,passage_id)
                 VALUES (:tenant,:project,:identity,:source_id,:source_version,:passage_id)
                 ON CONFLICT DO NOTHING"""),
        {**_params(passage), "identity": identity},
    )


def _check_failure(failure_at: str | None, point: str) -> None:
    if failure_at == point:
        raise RuntimeError("injected materialization failure")


def _entity_value(entity: Mapping[str, object]) -> str:
    entity_type = str(entity["type"])
    field_name = _ENTITY_VALUE_FIELDS.get(entity_type)
    if field_name is None:
        raise ValueError(f"entity type is not materializable: {entity_type}")
    value = entity.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"materialized {entity_type} needs {field_name}")
    return value


def _record_failed_materialization(
    engine: Engine,
    materialization_run_id: UUID,
    passage: ScopedPassageIdentity,
    extraction_run_id: UUID,
    previous: UUID | None,
    resolution_alias_catalog_version: str | None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """INSERT INTO materialization_run
                   (materialization_run_id,extraction_run_id,tenant_id,project_id,source_id,
                    source_version,passage_id,previous_active_extraction_run_id,
                    resolution_alias_catalog_version,status,error_detail,finished_at)
                   VALUES (:materialization_id,:extraction_run_id,:tenant,:project,:source_id,
                    :source_version,:passage_id,:previous,:resolution_catalog_version,
                    'failed','materialization transaction failed',now())"""
            ),
            {
                **_params(passage),
                "materialization_id": materialization_run_id,
                "extraction_run_id": extraction_run_id,
                "previous": previous,
                "resolution_catalog_version": resolution_alias_catalog_version,
            },
        )


def materialize_extraction(
    engine: Engine,
    passage: ScopedPassageIdentity,
    extraction_run_id: UUID,
    *,
    failure_at: str | None = None,
    failure_injector: Callable[[str], None] | None = None,
) -> MaterializationReport:
    """Materialize one immutable extraction using the current approved alias catalog."""
    allowed_failure_points = {
        "after_resolution",
        "after_assertions",
        "after_relationships",
        "before_activation",
        "after_completion",
    }
    if failure_at is not None and failure_at not in allowed_failure_points:
        raise ValueError("unknown materialization failure point")
    materialization_run_id = uuid4()
    previous: UUID | None = None
    resolution_alias_catalog_version: str | None = None
    counts = {"mentions": 0, "alias_proposals": 0, "assertions": 0, "evidence_edges": 0}
    try:
        with engine.begin() as connection:
            lock_alias_catalog(connection, passage.scope)
            params = {**_params(passage), "extraction_run_id": extraction_run_id}
            identity_matches = connection.execute(
                text("""SELECT 1 FROM llm_extraction_run
                       WHERE extraction_run_id=:extraction_run_id AND tenant_id=:tenant
                         AND project_id=:project AND source_id=:source_id
                         AND source_version=:source_version AND passage_id=:passage_id
                         AND status='succeeded'"""),
                params,
            ).scalar_one_or_none()
            if identity_matches is None:
                raise ValueError(
                    "extraction run does not match the full passage identity"
                )
            current = (
                connection.execute(
                    text("""SELECT active_extraction_run_id,validated_extraction_run_id
                       FROM passage_processing
                       WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                         AND source_version=:source_version AND passage_id=:passage_id
                       FOR UPDATE"""),
                    params,
                )
                .mappings()
                .one_or_none()
            )
            if current is None:
                raise ValueError("passage has no matching processing state")
            if current["validated_extraction_run_id"] != extraction_run_id:
                raise ValueError(
                    "extraction run does not match the active extraction identity"
                )
            extraction = _matching_active_extraction(
                connection, passage, extraction_run_id
            )
            if extraction is None:
                raise ValueError(
                    "extraction run does not match the active Jev selection identity"
                )
            catalog = approved_alias_catalog(connection, passage.scope)
            resolution_alias_catalog_version = catalog.version
            previous = current["active_extraction_run_id"]
            response = extraction["response"]
            if not isinstance(response, dict):
                raise ValueError("successful extraction run has no complete response")
            groups = tuple(
                response.get(name)
                for name in ("entities", "assertions", "relationships", "source_refs")
            )
            if not all(isinstance(group, list) for group in groups):
                raise ValueError("successful extraction response is incomplete")
            entities, assertions, relationships, source_refs = groups
            profile_name = str(extraction["profile_name"] or "")
            bootstrap_profile_applications(
                connection,
                passage.scope,
                profile_name=profile_name,
                profile_version=str(extraction["profile_version"]),
                profile_source_sha256=str(extraction["profile_source_sha256"] or ""),
            )
            connection.execute(
                text("""INSERT INTO materialization_run
                       (materialization_run_id,extraction_run_id,tenant_id,project_id,source_id,
                        source_version,passage_id,previous_active_extraction_run_id,
                        resolution_alias_catalog_version,status)
                       VALUES (:materialization_id,:extraction_run_id,:tenant,:project,:source_id,
                        :source_version,:passage_id,:previous,:resolution_catalog_version,'running')"""),
                {
                    **params,
                    "materialization_id": materialization_run_id,
                    "previous": previous,
                    "resolution_catalog_version": resolution_alias_catalog_version,
                },
            )

            local_nodes: dict[str, str] = {
                "source_passage": derive_scoped_id(
                    "passage",
                    passage.scope,
                    passage.source_id,
                    passage.source_version,
                    passage.passage_id,
                )
            }
            local_types: dict[str, str] = {"source_passage": "SourcePassage"}
            local_entities: dict[str, ResolvedEntity] = {}
            for entity in entities:
                entity_type = str(entity["type"])
                if entity_type in {"AliasCandidate", "RedactionMention"}:
                    continue
                local_id = str(entity["local_id"])
                entity_value = _entity_value(entity)
                mention = EntityMentionDraft(
                    local_id=local_id,
                    entity_type=entity_type,
                    text=entity_value,
                    source_ref=passage,
                    original_quote=str(entity["original_quote"]),
                    extraction_run_id=extraction_run_id,
                    extraction_version=str(extraction["extraction_version"]),
                    attributes={
                        key: value
                        for key, value in entity.items()
                        if key
                        not in {
                            "type",
                            "local_id",
                            "supporting_quote",
                            "occurrence_index",
                            "original_quote",
                            "original_start",
                            "original_end",
                            "model_start",
                            "model_end",
                        }
                    },
                )
                resolved = _resolve_profile_application(
                    passage.scope,
                    mention,
                    profile_name,
                ) or resolve_mention(connection, passage.scope, mention, catalog)
                local_nodes[local_id] = resolved.node_id
                local_types[local_id] = entity_type
                local_entities[local_id] = resolved
                mention_id = derive_scoped_id(
                    "mention",
                    passage.scope,
                    passage.source_id,
                    passage.source_version,
                    passage.passage_id,
                    str(extraction_run_id),
                    local_id,
                )
                connection.execute(
                    text("""INSERT INTO entity_mention
                           (tenant_id,project_id,source_id,source_version,passage_id,mention_id,
                            entity_type,text,extraction_run_id,original_start,original_end,
                            model_start,model_end,exact_quote)
                           VALUES (:tenant,:project,:source_id,:source_version,:passage_id,:mention_id,
                            :entity_type,:entity_text,:extraction_run_id,:original_start,:original_end,
                            :model_start,:model_end,:exact_quote)"""),
                    {
                        **params,
                        "mention_id": mention_id,
                        "entity_type": entity_type,
                        "entity_text": entity_value,
                        "original_start": entity.get("original_start"),
                        "original_end": entity.get("original_end"),
                        "model_start": entity.get("model_start"),
                        "model_end": entity.get("model_end"),
                        "exact_quote": entity["original_quote"],
                    },
                )
                connection.execute(
                    text("""INSERT INTO kg_node_mention_ref
                           (tenant_id,project_id,node_id,source_id,source_version,passage_id,
                            mention_id,extraction_run_id)
                           VALUES (:tenant,:project,:node_id,:source_id,:source_version,:passage_id,
                            :mention_id,:extraction_run_id)"""),
                    {
                        **params,
                        "node_id": resolved.node_id,
                        "mention_id": mention_id,
                    },
                )
                _source_ref(
                    connection,
                    "kg_node_source_ref",
                    "node_id",
                    resolved.node_id,
                    passage,
                )
                counts["mentions"] += 1

            for entity in entities:
                entity_type = str(entity["type"])
                local_id = str(entity["local_id"])
                if entity_type == "RedactionMention":
                    mention_id = derive_scoped_id(
                        "redaction",
                        passage.scope,
                        passage.source_id,
                        passage.source_version,
                        passage.passage_id,
                        str(extraction_run_id),
                        local_id,
                    )
                    connection.execute(
                        text(
                            """INSERT INTO redaction_mention
                               (tenant_id,project_id,source_id,source_version,passage_id,mention_id,
                                original_start,original_end,model_start,model_end)
                               VALUES (:tenant,:project,:source_id,:source_version,:passage_id,:mention_id,
                                :original_start,:original_end,:model_start,:model_end)"""
                        ),
                        {
                            **params,
                            "mention_id": mention_id,
                            "original_start": entity["original_start"],
                            "original_end": entity["original_end"],
                            "model_start": entity["model_start"],
                            "model_end": entity["model_end"],
                        },
                    )
                    counts["mentions"] += 1
                elif entity_type == "AliasCandidate":
                    target_local_id = str(entity["target_local_id"])
                    target_type = local_types.get(target_local_id)
                    if target_type not in {"Product", "Assay"}:
                        raise ValueError(
                            "alias candidate does not reference a resolved local Product or Assay"
                        )
                    target = local_entities[target_local_id]
                    record_alias_proposal(
                        connection,
                        passage.scope,
                        AliasProposal(
                            str(entity["candidate"]),
                            target.canonical_name,
                            target_type,
                            target.canonical_key,
                            0.5,
                            passage,
                            str(entity["original_quote"]),
                            int(entity["original_start"]),
                            int(entity["original_end"]),
                            "extraction",
                            "instructor",
                            str(extraction["extraction_version"]),
                        ),
                        model=str(extraction["model"]),
                        prompt_hash=str(extraction["prompt_hash"]),
                        schema_hash=str(extraction["schema_hash"]),
                        extraction_run_id=extraction_run_id,
                        input_sha256=str(extraction["input_sha256"]),
                    )
                    counts["alias_proposals"] += 1
            _check_failure(failure_at, "after_resolution")
            if failure_injector:
                failure_injector("after_resolution")

            for assertion in assertions:
                local_id = str(assertion["local_id"])
                assertion_id = derive_scoped_id(
                    "assertion",
                    passage.scope,
                    passage.source_id,
                    passage.source_version,
                    passage.passage_id,
                    str(extraction_run_id),
                    local_id,
                )
                connection.execute(
                    text(
                        """INSERT INTO assertion
                           (tenant_id,project_id,assertion_id,source_id,source_version,passage_id,
                            extraction_run_id,claim,chronology_state,review_state,graph_representation,
                            authority,status,original_quote,original_start,original_end,model_start,model_end)
                           VALUES (:tenant,:project,:assertion_id,:source_id,:source_version,:passage_id,
                            :extraction_run_id,:claim,:chronology_state,:review_state,'{}',:authority,:status,
                            :original_quote,:original_start,:original_end,:model_start,:model_end)"""
                    ),
                    {
                        **params,
                        "assertion_id": assertion_id,
                        "claim": assertion["claim"],
                        "chronology_state": assertion.get(
                            "chronology_state", "reported"
                        ),
                        "review_state": assertion.get("review_state", "unreviewed"),
                        "authority": assertion.get("authority"),
                        "status": assertion.get("status"),
                        "original_quote": assertion["original_quote"],
                        "original_start": assertion["original_start"],
                        "original_end": assertion["original_end"],
                        "model_start": assertion["model_start"],
                        "model_end": assertion["model_end"],
                    },
                )
                connection.execute(
                    text(
                        """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
                           VALUES (:tenant,:project,:node_id,'Assertion',CAST(:attributes AS jsonb))"""
                    ),
                    {
                        **params,
                        "node_id": assertion_id,
                        "attributes": json.dumps(
                            {
                                "claim": assertion["claim"],
                                "extraction_run_id": str(extraction_run_id),
                                "chronology_state": assertion.get(
                                    "chronology_state", "reported"
                                ),
                                "review_state": assertion.get(
                                    "review_state", "unreviewed"
                                ),
                            }
                        ),
                    },
                )
                _source_ref(
                    connection, "kg_node_source_ref", "node_id", assertion_id, passage
                )
                local_nodes[local_id] = assertion_id
                local_types[local_id] = "Assertion"
                counts["assertions"] += 1
            _check_failure(failure_at, "after_assertions")
            if failure_injector:
                failure_injector("after_assertions")

            local_edges: dict[str, str] = {}
            for relationship in relationships:
                local_id = str(relationship["local_id"])
                source_local_id = str(relationship["source_local_id"])
                target_local_id = str(relationship["target_local_id"])
                assertion_local_id = str(relationship["assertion_local_id"])
                if (
                    source_local_id not in local_nodes
                    or target_local_id not in local_nodes
                ):
                    raise ValueError(
                        "materialization relationship has an unresolved local endpoint"
                    )
                if (
                    assertion_local_id not in local_nodes
                    or local_types.get(assertion_local_id) != "Assertion"
                ):
                    raise ValueError(
                        "materialization relationship has an unresolved assertion"
                    )
                edge_id = derive_scoped_id(
                    "evidence",
                    passage.scope,
                    passage.source_id,
                    passage.source_version,
                    passage.passage_id,
                    str(extraction_run_id),
                    local_id,
                )
                connection.execute(
                    text("""INSERT INTO kg_edge
                           (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,
                            assertion_id,is_evidence,attributes)
                           VALUES (:tenant,:project,:edge_id,:edge_type,:source_node_id,:target_node_id,
                            :assertion_id,true,CAST(:attributes AS jsonb))"""),
                    {
                        **params,
                        "edge_id": edge_id,
                        "edge_type": relationship["relation_type"],
                        "source_node_id": local_nodes[source_local_id],
                        "target_node_id": local_nodes[target_local_id],
                        "assertion_id": local_nodes[assertion_local_id],
                        "attributes": json.dumps(relationship.get("attributes", {})),
                    },
                )
                _source_ref(
                    connection, "kg_edge_source_ref", "edge_id", edge_id, passage
                )
                local_edges[local_id] = edge_id
                counts["evidence_edges"] += 1
            for source_ref in source_refs:
                record_local_id = str(source_ref["record_local_id"])
                if record_local_id in local_nodes:
                    _source_ref(
                        connection,
                        "kg_node_source_ref",
                        "node_id",
                        local_nodes[record_local_id],
                        passage,
                    )
                elif record_local_id in local_edges:
                    _source_ref(
                        connection,
                        "kg_edge_source_ref",
                        "edge_id",
                        local_edges[record_local_id],
                        passage,
                    )
                else:
                    raise ValueError(
                        "materialization source reference has an unresolved local record"
                    )
            _check_failure(failure_at, "after_relationships")
            if failure_injector:
                failure_injector("after_relationships")

            _check_failure(failure_at, "before_activation")
            if failure_injector:
                failure_injector("before_activation")
            connection.execute(
                text("""UPDATE materialization_run
                       SET status='succeeded',counts=CAST(:counts AS jsonb),finished_at=now()
                       WHERE tenant_id=:tenant AND project_id=:project
                         AND materialization_run_id=:materialization_id"""),
                {
                    **params,
                    "materialization_id": materialization_run_id,
                    "counts": json.dumps(counts),
                },
            )
            _check_failure(failure_at, "after_completion")
            if failure_injector:
                failure_injector("after_completion")
            connection.execute(
                text(
                    """UPDATE passage_processing
                       SET state='extracted',reason='extracted_successfully',
                           active_extraction_run_id=:extraction_run_id,updated_at=now()
                       WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                         AND source_version=:source_version AND passage_id=:passage_id"""
                ),
                params,
            )
    except ValueError:
        raise
    except Exception as error:
        _record_failed_materialization(
            engine,
            materialization_run_id,
            passage,
            extraction_run_id,
            previous,
            resolution_alias_catalog_version,
        )
        if (
            isinstance(error, RuntimeError)
            and str(error) == "injected materialization failure"
        ):
            raise
        raise RuntimeError("materialization failed") from None

    return MaterializationReport(
        materialization_run_id,
        extraction_run_id,
        previous,
        counts["mentions"],
        counts["alias_proposals"],
        counts["assertions"],
        counts["evidence_edges"],
    )
