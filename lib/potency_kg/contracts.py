"""Small, scoped contracts shared by every evidence-pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, order=True)
class Scope:
    tenant_id: str
    project_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.project_id:
            raise ValueError("tenant_id and project_id are required")

    def as_tuple(self) -> tuple[str, str]:
        return self.tenant_id, self.project_id


@dataclass(frozen=True, order=True)
class ScopedSourceVersion:
    tenant_id: str
    project_id: str
    source_id: str
    source_version: str

    @property
    def scope(self) -> Scope:
        return Scope(self.tenant_id, self.project_id)

    def as_tuple(self) -> tuple[str, str, str, str]:
        return self.tenant_id, self.project_id, self.source_id, self.source_version


@dataclass(frozen=True, order=True)
class ScopedPassageIdentity:
    tenant_id: str
    project_id: str
    source_id: str
    source_version: str
    passage_id: str

    @property
    def source(self) -> ScopedSourceVersion:
        return ScopedSourceVersion(self.tenant_id, self.project_id, self.source_id, self.source_version)

    @property
    def scope(self) -> Scope:
        return Scope(self.tenant_id, self.project_id)

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return self.tenant_id, self.project_id, self.source_id, self.source_version, self.passage_id


SourceRef = ScopedPassageIdentity

NODE_TYPES = frozenset(
    {
        "SourceDocument", "SourcePassage", "Product", "Application", "Indication", "PotencyStrategy",
        "Assay", "MeasuredAttribute", "RegulatoryInteraction", "ReviewConclusion", "Assertion",
    }
)
ALLOWED_RELATION_ENDPOINTS: dict[str, frozenset[tuple[str, str]]] = {
    "contains_passage": frozenset({("SourceDocument", "SourcePassage")}),
    "supported_by": frozenset({("Assertion", "SourcePassage")}),
    "applies_to": frozenset({("Assertion", "Product"), ("Assertion", "Application"), ("Assertion", "Indication")}),
    "measures": frozenset({("Assay", "MeasuredAttribute")}),
    "used_as": frozenset({("Assay", "PotencyStrategy"), ("Assay", "MeasuredAttribute")}),
    "proposed_in": frozenset({("PotencyStrategy", "RegulatoryInteraction")}),
    "requested_in": frozenset({("PotencyStrategy", "RegulatoryInteraction")}),
    "revised_by": frozenset({("PotencyStrategy", "RegulatoryInteraction")}),
    "concluded_in": frozenset({("ReviewConclusion", "RegulatoryInteraction")}),
    "cites": frozenset({("Assertion", "Assertion")}),
    "supersedes_or_supplements": frozenset({("SourceDocument", "SourceDocument")}),
}


@dataclass(frozen=True)
class AssertionDraft:
    local_id: str
    claim: str
    chronology_state: str = "reported"
    review_state: str = "unreviewed"

    def __post_init__(self) -> None:
        if not isinstance(self.local_id, str) or not self.local_id.strip():
            raise ValueError("an assertion needs a nonempty passage-local ID")
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("an assertion needs a local ID and one nonempty atomic claim")


@dataclass(frozen=True)
class EvidenceRelationshipDraft:
    relation_type: str
    source_type: str
    target_type: str
    assertion_local_id: str | None

    def __post_init__(self) -> None:
        if (self.source_type, self.target_type) not in ALLOWED_RELATION_ENDPOINTS.get(self.relation_type, frozenset()):
            raise ValueError("relation endpoints are not allowlisted")
        if not self.assertion_local_id:
            raise ValueError("every evidence relationship must reference an assertion")


@dataclass(frozen=True)
class GraphNodeRecord:
    scope: Scope
    node_id: str
    node_type: str
    attributes: dict[str, object]


@dataclass(frozen=True)
class GraphEdgeRecord:
    scope: Scope
    edge_id: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    assertion_id: str | None = None


ProjectionMode = Literal["structural", "semantic"]
