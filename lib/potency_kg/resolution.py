"""Trusted profile resolution and atomic relational semantic materialization."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .contracts import Scope, ScopedPassageIdentity
from .ids import derive_scoped_id
from .project_profiles import (
    ProjectProfile,
    trusted_applications_for_profile,
    trusted_descriptor_path_for_profile,
)


@dataclass(frozen=True)
class AliasBootstrapReport:
    product_count: int
    application_count: int
    alias_count: int


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
        if not self.local_id.strip() or not self.entity_type.strip() or not self.text.strip():
            raise ValueError("entity mention needs nonempty local identity, type, and text")
        if not self.original_quote:
            raise ValueError("entity mention needs an exact original quote")


@dataclass(frozen=True)
class ResolvedEntity:
    node_id: str
    node_type: str
    canonical_name: str
    matched_by: str


@dataclass(frozen=True)
class MaterializationReport:
    materialization_run_id: UUID
    extraction_run_id: UUID
    previous_active_extraction_run_id: UUID | None
    mentions: int
    alias_candidates: int
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


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _profile_scope(profile: ProjectProfile) -> Scope:
    if profile.scope is None:
        raise ValueError("project profile must carry its tenant/project scope")
    return profile.scope


def bootstrap_profile_aliases(
    connection: Connection,
    profile: ProjectProfile,
) -> AliasBootstrapReport:
    """Create canonical Product anchors from trusted profile aliases only."""
    scope = _profile_scope(profile)
    grouped: dict[str, list[str]] = {}
    for alias, canonical_id in profile.aliases.items():
        if not alias.strip() or not canonical_id.strip():
            raise ValueError("trusted profile aliases must be nonempty")
        grouped.setdefault(canonical_id, []).append(alias)
    for canonical_id, aliases in grouped.items():
        node_id = derive_scoped_id("product", scope, canonical_id)
        attributes = {
            "canonical_id": canonical_id,
            "name": aliases[0],
            "trusted_aliases": aliases,
            "profile_name": profile.name,
            "profile_version": profile.profile_version,
            "profile_source_path": profile.descriptor_path,
            "profile_source_sha256": profile.descriptor_sha256,
            "system_generated": True,
        }
        connection.execute(
            text(
                """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
                   VALUES (:tenant,:project,:node_id,'Product',CAST(:attributes AS jsonb))
                   ON CONFLICT (tenant_id,project_id,node_id) DO UPDATE
                   SET attributes=EXCLUDED.attributes
                   WHERE kg_node.node_type='Product'"""
            ),
            {
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "node_id": node_id,
                "attributes": json.dumps(attributes),
            },
        )
    for application_id, canonical_id in profile.applications.items():
        if not application_id.strip() or not canonical_id.strip():
            raise ValueError("trusted profile applications must be nonempty")
        node_id = derive_scoped_id("application", scope, canonical_id)
        attributes = {
            "canonical_id": canonical_id,
            "application_id": application_id,
            "profile_name": profile.name,
            "profile_version": profile.profile_version,
            "profile_source_path": profile.descriptor_path,
            "profile_source_sha256": profile.descriptor_sha256,
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
    return AliasBootstrapReport(len(grouped), len(profile.applications), len(profile.aliases))


def resolve_mention(
    connection: Connection,
    scope: Scope,
    mention: EntityMentionDraft,
    profile: ProjectProfile,
) -> ResolvedEntity:
    """Resolve exact trusted Product aliases; never similarity-merge extracted text."""
    if mention.source_ref.scope != scope or _profile_scope(profile) != scope:
        raise ValueError("entity resolution requires matching full tenant/project scope")
    bootstrap_profile_aliases(connection, profile)
    trusted = {
        _normalized(alias): canonical_id
        for alias, canonical_id in profile.aliases.items()
    }
    canonical_id = trusted.get(_normalized(mention.text)) if mention.entity_type == "Product" else None
    if canonical_id is not None:
        return ResolvedEntity(
            derive_scoped_id("product", scope, canonical_id),
            "Product",
            canonical_id,
            "trusted_profile_alias",
        )
    application_id = (
        profile.applications.get(mention.text.strip())
        if mention.entity_type == "Application"
        else None
    )
    if application_id is not None:
        return ResolvedEntity(
            derive_scoped_id("application", scope, application_id),
            "Application",
            application_id,
            "trusted_profile_application",
        )

    node_id = derive_scoped_id(
        "entity",
        scope,
        mention.entity_type,
        mention.source_ref.source_id,
        mention.source_ref.source_version,
        mention.source_ref.passage_id,
        str(mention.extraction_run_id or "unmaterialized"),
        mention.local_id,
    )
    attributes = {
        "canonical_name": mention.text,
        "extraction_run_id": str(mention.extraction_run_id) if mention.extraction_run_id else None,
        "extraction_version": mention.extraction_version,
        **dict(mention.attributes),
    }
    connection.execute(
        text(
            """INSERT INTO kg_node (tenant_id,project_id,node_id,node_type,attributes)
               VALUES (:tenant,:project,:node_id,:node_type,CAST(:attributes AS jsonb))
               ON CONFLICT (tenant_id,project_id,node_id) DO NOTHING"""
        ),
        {
            "tenant": scope.tenant_id,
            "project": scope.project_id,
            "node_id": node_id,
            "node_type": mention.entity_type,
            "attributes": json.dumps(attributes),
        },
    )
    return ResolvedEntity(node_id, mention.entity_type, mention.text, "new_entity")


def _params(passage: ScopedPassageIdentity) -> dict[str, object]:
    return dict(
        zip(
            ("tenant", "project", "source_id", "source_version", "passage_id"),
            passage.as_tuple(),
            strict=True,
        )
    )


def _source_ref(connection: Connection, table: str, identity_column: str, identity: str, passage: ScopedPassageIdentity) -> None:
    connection.execute(
        text(
            f"""INSERT INTO {table}
                 (tenant_id,project_id,{identity_column},source_id,source_version,passage_id)
                 VALUES (:tenant,:project,:identity,:source_id,:source_version,:passage_id)
                 ON CONFLICT DO NOTHING"""
        ),
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
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """INSERT INTO materialization_run
                   (materialization_run_id,extraction_run_id,tenant_id,project_id,source_id,
                    source_version,passage_id,previous_active_extraction_run_id,status,error_detail,finished_at)
                   VALUES (:materialization_id,:extraction_run_id,:tenant,:project,:source_id,
                    :source_version,:passage_id,:previous,'failed','materialization transaction failed',now())"""
            ),
            {
                **_params(passage),
                "materialization_id": materialization_run_id,
                "extraction_run_id": extraction_run_id,
                "previous": previous,
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
    """Resolve and materialize one validated extraction in one pointer-owning transaction."""
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
    counts = {"mentions": 0, "alias_candidates": 0, "assertions": 0, "evidence_edges": 0}
    try:
        with engine.begin() as connection:
            params = {**_params(passage), "extraction_run_id": extraction_run_id}
            extraction = connection.execute(
                text(
                    """SELECT extraction_version,profile_name,profile_aliases,profile_source_sha256,
                              profile_version,response
                       FROM llm_extraction_run
                       WHERE extraction_run_id=:extraction_run_id AND tenant_id=:tenant
                         AND project_id=:project AND source_id=:source_id
                         AND source_version=:source_version AND passage_id=:passage_id
                         AND status='succeeded'"""
                ),
                params,
            ).mappings().one_or_none()
            if extraction is None:
                raise ValueError("extraction run does not match the full passage identity")
            current = connection.execute(
                text(
                    """SELECT active_extraction_run_id FROM passage_processing
                       WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                         AND source_version=:source_version AND passage_id=:passage_id
                       FOR UPDATE"""
                ),
                params,
            ).mappings().one_or_none()
            if current is None:
                raise ValueError("passage has no matching processing state")
            previous = current["active_extraction_run_id"]
            response = extraction["response"]
            if not isinstance(response, dict):
                raise ValueError("successful extraction run has no complete response")
            groups = tuple(response.get(name) for name in ("entities", "assertions", "relationships", "source_refs"))
            if not all(isinstance(group, list) for group in groups):
                raise ValueError("successful extraction response is incomplete")
            entities, assertions, relationships, source_refs = groups
            profile = ProjectProfile(
                str(extraction["profile_name"] or "snapshot"),
                dict(extraction["profile_aliases"] or {}),
                trusted_descriptor_path_for_profile(str(extraction["profile_name"] or "snapshot")),
                str(extraction["profile_source_sha256"] or ""),
                str(extraction["profile_version"]),
                passage.scope,
                trusted_applications_for_profile(str(extraction["profile_name"] or "snapshot")),
            )
            bootstrap_profile_aliases(connection, profile)
            connection.execute(
                text(
                    """INSERT INTO materialization_run
                       (materialization_run_id,extraction_run_id,tenant_id,project_id,source_id,
                        source_version,passage_id,previous_active_extraction_run_id,status)
                       VALUES (:materialization_id,:extraction_run_id,:tenant,:project,:source_id,
                        :source_version,:passage_id,:previous,'running')"""
                ),
                {
                    **params,
                    "materialization_id": materialization_run_id,
                    "previous": previous,
                },
            )

            local_nodes: dict[str, str] = {
                "source_passage": derive_scoped_id(
                    "passage", passage.scope, passage.source_id, passage.source_version, passage.passage_id
                )
            }
            local_types: dict[str, str] = {"source_passage": "SourcePassage"}
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
                        if key not in {
                            "type", "local_id", "supporting_quote", "occurrence_index",
                            "original_quote", "original_start", "original_end", "model_start", "model_end",
                        }
                    },
                )
                resolved = resolve_mention(connection, passage.scope, mention, profile)
                local_nodes[local_id] = resolved.node_id
                local_types[local_id] = entity_type
                mention_id = derive_scoped_id(
                    "mention", passage.scope, passage.source_id, passage.source_version,
                    passage.passage_id, str(extraction_run_id), local_id,
                )
                connection.execute(
                    text(
                        """INSERT INTO entity_mention
                           (tenant_id,project_id,source_id,source_version,passage_id,mention_id,
                            entity_type,text,extraction_run_id,original_start,original_end,
                            model_start,model_end,exact_quote)
                           VALUES (:tenant,:project,:source_id,:source_version,:passage_id,:mention_id,
                            :entity_type,:entity_text,:extraction_run_id,:original_start,:original_end,
                            :model_start,:model_end,:exact_quote)"""
                    ),
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
                _source_ref(connection, "kg_node_source_ref", "node_id", resolved.node_id, passage)
                counts["mentions"] += 1

            for entity in entities:
                entity_type = str(entity["type"])
                local_id = str(entity["local_id"])
                if entity_type == "RedactionMention":
                    mention_id = derive_scoped_id(
                        "redaction", passage.scope, passage.source_id, passage.source_version,
                        passage.passage_id, str(extraction_run_id), local_id,
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
                    product_local_id = str(entity["product_local_id"])
                    if local_types.get(product_local_id) != "Product":
                        raise ValueError("alias candidate does not reference a resolved local Product")
                    alias_id = derive_scoped_id(
                        "alias", passage.scope, passage.source_id, passage.source_version,
                        passage.passage_id, str(extraction_run_id), local_id,
                    )
                    connection.execute(
                        text(
                            """INSERT INTO alias_candidate
                               (tenant_id,project_id,source_id,source_version,passage_id,alias_id,candidate,
                                extraction_run_id,proposer,extraction_version,original_quote,status)
                               VALUES (:tenant,:project,:source_id,:source_version,:passage_id,:alias_id,:candidate,
                                :extraction_run_id,'instructor',:extraction_version,:original_quote,'pending')"""
                        ),
                        {
                            **params,
                            "alias_id": alias_id,
                            "candidate": entity["candidate"],
                            "extraction_version": extraction["extraction_version"],
                            "original_quote": entity["original_quote"],
                        },
                    )
                    counts["alias_candidates"] += 1
            _check_failure(failure_at, "after_resolution")
            if failure_injector:
                failure_injector("after_resolution")

            for assertion in assertions:
                local_id = str(assertion["local_id"])
                assertion_id = derive_scoped_id(
                    "assertion", passage.scope, passage.source_id, passage.source_version,
                    passage.passage_id, str(extraction_run_id), local_id,
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
                        "chronology_state": assertion.get("chronology_state", "reported"),
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
                                "chronology_state": assertion.get("chronology_state", "reported"),
                                "review_state": assertion.get("review_state", "unreviewed"),
                            }
                        ),
                    },
                )
                _source_ref(connection, "kg_node_source_ref", "node_id", assertion_id, passage)
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
                if source_local_id not in local_nodes or target_local_id not in local_nodes:
                    raise ValueError("materialization relationship has an unresolved local endpoint")
                if assertion_local_id not in local_nodes or local_types.get(assertion_local_id) != "Assertion":
                    raise ValueError("materialization relationship has an unresolved assertion")
                edge_id = derive_scoped_id(
                    "evidence", passage.scope, passage.source_id, passage.source_version,
                    passage.passage_id, str(extraction_run_id), local_id,
                )
                connection.execute(
                    text(
                        """INSERT INTO kg_edge
                           (tenant_id,project_id,edge_id,edge_type,source_node_id,target_node_id,
                            assertion_id,is_evidence,attributes)
                           VALUES (:tenant,:project,:edge_id,:edge_type,:source_node_id,:target_node_id,
                            :assertion_id,true,CAST(:attributes AS jsonb))"""
                    ),
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
                _source_ref(connection, "kg_edge_source_ref", "edge_id", edge_id, passage)
                local_edges[local_id] = edge_id
                counts["evidence_edges"] += 1
            for source_ref in source_refs:
                record_local_id = str(source_ref["record_local_id"])
                if record_local_id in local_nodes:
                    _source_ref(connection, "kg_node_source_ref", "node_id", local_nodes[record_local_id], passage)
                elif record_local_id in local_edges:
                    _source_ref(connection, "kg_edge_source_ref", "edge_id", local_edges[record_local_id], passage)
                else:
                    raise ValueError("materialization source reference has an unresolved local record")
            _check_failure(failure_at, "after_relationships")
            if failure_injector:
                failure_injector("after_relationships")

            _check_failure(failure_at, "before_activation")
            if failure_injector:
                failure_injector("before_activation")
            connection.execute(
                text(
                    """UPDATE materialization_run
                       SET status='succeeded',counts=CAST(:counts AS jsonb),finished_at=now()
                       WHERE tenant_id=:tenant AND project_id=:project
                         AND materialization_run_id=:materialization_id"""
                ),
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
            engine, materialization_run_id, passage, extraction_run_id, previous
        )
        if isinstance(error, RuntimeError) and str(error) == "injected materialization failure":
            raise
        raise RuntimeError("materialization failed") from None

    return MaterializationReport(
        materialization_run_id,
        extraction_run_id,
        previous,
        counts["mentions"],
        counts["alias_candidates"],
        counts["assertions"],
        counts["evidence_edges"],
    )
