"""Conservative, auditable alias discovery over active source documents."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .aliases import ALIAS_NORMALIZATION_VERSION, normalize_alias
from .contracts import Scope, ScopedPassageIdentity
from .ids import canonical_json, derive_scoped_id


DETERMINISTIC_ALIAS_DETECTOR = "document-alias-patterns"
DETERMINISTIC_ALIAS_DETECTOR_VERSION = "document-alias-patterns-v1"


@dataclass(frozen=True)
class AliasProposal:
    alias_value: str
    canonical_name: str | None
    entity_type: Literal["Product", "Assay"] | None
    canonical_key: str | None
    confidence: float
    source: ScopedPassageIdentity
    exact_quote: str
    original_start: int
    original_end: int
    source_type: Literal["deterministic_rule", "llm_mined", "extraction"]
    detector_name: str
    detector_version: str


@dataclass(frozen=True)
class ProposalPersistence:
    alias_id: str
    alias_inserted: bool
    evidence_inserted: bool


@dataclass(frozen=True)
class AliasDiscoveryReport:
    method: Literal["deterministic", "llm"]
    documents: int
    succeeded: int
    failed: int
    reused: int
    proposals: int
    evidence_inserted: int
    discovery_run_ids: Sequence[UUID]

    def __post_init__(self) -> None:
        object.__setattr__(self, "discovery_run_ids", tuple(self.discovery_run_ids))


_PARENTHETICAL = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){1,})\s*\((?P<short>[A-Za-z][A-Za-z0-9-]*)\)"
)
_PARENTHETICAL_REVERSED = re.compile(
    r"(?P<short>[A-Za-z][A-Za-z0-9-]*)\s*\((?P<long>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){1,})\)"
)
_EXPLICIT = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){1,})\s*,\s*"
    r"(?P<phrase>also known as|abbreviated as|formerly called)\s+(?P<short>[A-Za-z][A-Za-z0-9-]*)",
    re.IGNORECASE,
)


def _letters(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalpha())


def _is_ordered_subsequence(short: str, long: str) -> bool:
    remaining = iter(_letters(long))
    return all(any(candidate == character for candidate in remaining) for character in _letters(short))


def _valid_pair(long: str, short: str) -> bool:
    normalized_long = normalize_alias(long)
    normalized_short = normalize_alias(short)
    short_letters = _letters(short)
    if not normalized_long or not normalized_short or normalized_long == normalized_short:
        return False
    if short.isdigit() or long.isdigit() or "page" in normalized_long.split():
        return False
    if len(short_letters) < 2 or not any(character.isalpha() for character in long):
        return False
    if len(_letters(long)) > len(short_letters) * 8:
        return False
    return _is_ordered_subsequence(short, long)


def _tightest_valid_long_form(raw_long: str, short: str) -> tuple[str, int] | None:
    """Choose the shortest grounded multiword suffix, not the sentence framing."""
    words = tuple(re.finditer(r"\S+", raw_long))
    for word in reversed(words[:-1]):
        candidate = raw_long[word.start():].strip()
        if _valid_pair(candidate, short):
            return candidate, word.start()
    return None


def _proposal(source: ScopedPassageIdentity, body: str, match: re.Match[str]) -> AliasProposal | None:
    short = match.group("short").strip()
    chosen = _tightest_valid_long_form(match.group("long"), short)
    if chosen is None:
        return None
    long, long_offset = chosen
    start = (
        match.start("long") + long_offset
        if match.start("long") < match.start("short")
        else match.start()
    )
    end = match.end()
    return AliasProposal(
        alias_value=short,
        canonical_name=long,
        entity_type=None,
        canonical_key=None,
        confidence=0.8,
        source=source,
        exact_quote=body[start:end],
        original_start=start,
        original_end=end,
        source_type="deterministic_rule",
        detector_name=DETERMINISTIC_ALIAS_DETECTOR,
        detector_version=DETERMINISTIC_ALIAS_DETECTOR_VERSION,
    )


def discover_alias_patterns(source: ScopedPassageIdentity, body: str) -> tuple[AliasProposal, ...]:
    """Return only grounded, conservative aliases found in one original passage."""
    if not isinstance(body, str):
        raise ValueError("passage body must be a string")
    candidates: list[AliasProposal] = []
    occupied: set[tuple[int, int]] = set()
    for pattern in (_PARENTHETICAL, _PARENTHETICAL_REVERSED, _EXPLICIT):
        for match in pattern.finditer(body):
            span = match.span()
            if span in occupied:
                continue
            proposal = _proposal(source, body, match)
            if proposal is not None:
                candidates.append(proposal)
                occupied.add(span)
    return tuple(sorted(candidates, key=lambda proposal: (proposal.original_start, proposal.original_end, proposal.alias_value)))


def _proposal_alias_id(scope: Scope, proposal: AliasProposal) -> str:
    return derive_scoped_id(
        "alias", scope, normalize_alias(proposal.alias_value), proposal.entity_type or "<null>",
        proposal.canonical_key or "<null>", proposal.canonical_name or "<null>",
    )


def _proposal_input_hash(proposal: AliasProposal) -> str:
    return sha256(canonical_json({
        "source": proposal.source.as_tuple(), "alias_value": proposal.alias_value,
        "canonical_name": proposal.canonical_name, "entity_type": proposal.entity_type,
        "canonical_key": proposal.canonical_key, "quote": proposal.exact_quote,
        "start": proposal.original_start, "end": proposal.original_end,
        "detector": proposal.detector_name, "version": proposal.detector_version,
    }).encode()).hexdigest()


def record_alias_proposal(
    connection: Connection,
    scope: Scope,
    proposal: AliasProposal,
    *,
    discovery_run_id: UUID | None = None,
) -> ProposalPersistence:
    """Persist one pending proposal and its immutable occurrence evidence."""
    if proposal.source.scope != scope:
        raise ValueError("proposal source scope does not match requested scope")
    # The caller passes original passage text only through a validated detector; offsets remain immutable evidence.
    if not proposal.exact_quote or proposal.original_start < 0 or proposal.original_end <= proposal.original_start:
        raise ValueError("proposal offsets must define a nonempty quote")
    normalized = normalize_alias(proposal.alias_value)
    if not normalized:
        raise ValueError("proposal alias must normalize to nonempty text")
    target = {
        "entity_type": proposal.entity_type,
        "canonical_key": proposal.canonical_key,
        "canonical_name": proposal.canonical_name,
    }
    lookup_params = {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "normalized": normalized, **target}
    if all(value is None for value in target.values()):
        lookup = """SELECT alias_id FROM aliases
                     WHERE tenant_id=:tenant_id AND project_id=:project_id AND normalized_alias=:normalized
                       AND entity_type IS NULL AND canonical_key IS NULL AND canonical_name IS NULL
                     ORDER BY alias_id LIMIT 1 FOR UPDATE"""
    else:
        lookup = """SELECT alias_id FROM aliases
                     WHERE tenant_id=:tenant_id AND project_id=:project_id AND normalized_alias=:normalized
                       AND entity_type IS NOT DISTINCT FROM :entity_type
                       AND canonical_key IS NOT DISTINCT FROM :canonical_key
                       AND canonical_name IS NOT DISTINCT FROM :canonical_name
                     ORDER BY alias_id LIMIT 1 FOR UPDATE"""
    existing = connection.execute(text(lookup), lookup_params).scalar_one_or_none()
    alias_inserted = existing is None
    alias_id = str(existing) if existing is not None else _proposal_alias_id(scope, proposal)
    if alias_inserted:
        connection.execute(text(
            """INSERT INTO aliases
               (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,
                status,managed_by,normalization_version,confidence)
               VALUES (:tenant_id,:project_id,:alias_id,:alias_value,:normalized,:entity_type,:canonical_key,
                :canonical_name,'pending','review',:normalization_version,:confidence)"""
        ), {
            "tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": alias_id,
            "alias_value": proposal.alias_value, "normalized": normalized, **target,
            "normalization_version": ALIAS_NORMALIZATION_VERSION, "confidence": proposal.confidence,
        })
    evidence_id = derive_scoped_id(
        "alias-evidence", scope, alias_id, proposal.source.source_id, proposal.source.source_version,
        proposal.source.passage_id, proposal.detector_name, proposal.detector_version,
        str(proposal.original_start), str(proposal.original_end), _proposal_input_hash(proposal),
    )
    evidence = connection.execute(text(
        """INSERT INTO alias_evidence
           (tenant_id,project_id,evidence_id,alias_id,source_type,source_id,source_version,passage_id,
            discovery_run_id,exact_quote,original_start,original_end,detector_name,detector_version,input_sha256)
           VALUES (:tenant_id,:project_id,:evidence_id,:alias_id,:source_type,:source_id,:source_version,:passage_id,
            :discovery_run_id,:exact_quote,:original_start,:original_end,:detector_name,:detector_version,:input_sha256)
           ON CONFLICT (tenant_id,project_id,evidence_id) DO NOTHING"""
    ), {
        "tenant_id": scope.tenant_id, "project_id": scope.project_id, "evidence_id": evidence_id,
        "alias_id": alias_id, "source_type": proposal.source_type, "source_id": proposal.source.source_id,
        "source_version": proposal.source.source_version, "passage_id": proposal.source.passage_id,
        "discovery_run_id": discovery_run_id,
        "exact_quote": proposal.exact_quote, "original_start": proposal.original_start,
        "original_end": proposal.original_end, "detector_name": proposal.detector_name,
        "detector_version": proposal.detector_version, "input_sha256": _proposal_input_hash(proposal),
    })
    return ProposalPersistence(alias_id, alias_inserted, evidence.rowcount == 1)


@dataclass(frozen=True)
class _Document:
    source_id: str
    source_version: str
    passages: tuple[tuple[ScopedPassageIdentity, str, str], ...]


def _active_documents(connection: Connection, scope: Scope) -> tuple[_Document, ...]:
    rows = connection.execute(text(
        """SELECT passage.source_id,passage.source_version,passage.passage_id,passage.text,passage.text_sha256
             FROM source_head AS head
             JOIN source_passage AS passage ON (passage.tenant_id,passage.project_id,passage.source_id,passage.source_version)
                 = (head.tenant_id,head.project_id,head.source_id,head.source_version)
             JOIN passage_processing AS processing ON (processing.tenant_id,processing.project_id,processing.source_id,
                 processing.source_version,processing.passage_id) = (passage.tenant_id,passage.project_id,passage.source_id,
                 passage.source_version,passage.passage_id)
            WHERE head.tenant_id=:tenant_id AND head.project_id=:project_id
              AND processing.reason='eligible_for_processing'
            ORDER BY passage.source_id,passage.source_version,passage.passage_id"""
    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id}).mappings()
    grouped: dict[tuple[str, str], list[tuple[ScopedPassageIdentity, str, str]]] = {}
    for row in rows:
        source_id, source_version = str(row["source_id"]), str(row["source_version"])
        identity = ScopedPassageIdentity(scope.tenant_id, scope.project_id, source_id, source_version, str(row["passage_id"]))
        grouped.setdefault((source_id, source_version), []).append((identity, str(row["text"]), str(row["text_sha256"])))
    return tuple(_Document(source_id, version, tuple(passages)) for (source_id, version), passages in grouped.items())


def _document_hash(document: _Document) -> str:
    return sha256(canonical_json({
        "detector_version": DETERMINISTIC_ALIAS_DETECTOR_VERSION,
        "passages": [{"passage_id": item[0].passage_id, "text_sha256": item[2]} for item in document.passages],
    }).encode()).hexdigest()


def _successful_run_id(connection: Connection, scope: Scope, document: _Document, input_sha256: str) -> UUID | None:
    run_id = connection.execute(text(
        """SELECT discovery_run_id FROM alias_discovery_run WHERE tenant_id=:tenant_id AND project_id=:project_id
              AND method='deterministic' AND source_id=:source_id AND source_version=:source_version
              AND input_sha256=:input_sha256 AND detector_version=:detector_version AND status='succeeded'
            ORDER BY finished_at DESC NULLS LAST, discovery_run_id DESC LIMIT 1"""
    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "source_id": document.source_id,
        "source_version": document.source_version, "input_sha256": input_sha256,
        "detector_version": DETERMINISTIC_ALIAS_DETECTOR_VERSION}).scalar_one_or_none()
    return UUID(str(run_id)) if run_id is not None else None


def _insert_run(connection: Connection, scope: Scope, run_id: UUID, document: _Document, input_sha256: str, status: str, counts: dict[str, int], error: str | None = None) -> None:
    connection.execute(text(
        """INSERT INTO alias_discovery_run
           (tenant_id,project_id,discovery_run_id,method,source_id,source_version,input_sha256,detector_version,
            status,counts,error_detail,finished_at)
           VALUES (:tenant_id,:project_id,:run_id,'deterministic',:source_id,:source_version,:input_sha256,
            :detector_version,:status,CAST(:counts AS jsonb),:error,
            CASE WHEN :status = 'running' THEN NULL ELSE now() END)"""
    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "run_id": run_id,
        "source_id": document.source_id, "source_version": document.source_version, "input_sha256": input_sha256,
        "detector_version": DETERMINISTIC_ALIAS_DETECTOR_VERSION, "status": status,
        "counts": json.dumps(counts, sort_keys=True), "error": error})


def discover_aliases_deterministic(engine: Engine, scope: Scope) -> AliasDiscoveryReport:
    """Discover aliases independently for each active complete document, safely reusing successes."""
    with engine.connect() as connection:
        documents = _active_documents(connection, scope)
    succeeded = failed = reused = proposals = evidence_inserted = 0
    run_ids: list[UUID] = []
    for document in documents:
        input_sha256 = _document_hash(document)
        with engine.connect() as connection:
            reusable_run_id = _successful_run_id(connection, scope, document, input_sha256)
        if reusable_run_id is not None:
            reused += 1
            run_ids.append(reusable_run_id)
            continue
        run_id = uuid4()
        try:
            with engine.begin() as connection:
                document_proposals = tuple(
                    proposal
                    for passage, body, _text_hash in document.passages
                    for proposal in discover_alias_patterns(passage, body)
                )
                _insert_run(connection, scope, run_id, document, input_sha256, "running", {})
                persistences = [
                    record_alias_proposal(connection, scope, proposal, discovery_run_id=run_id)
                    for proposal in document_proposals
                ]
                inserted = sum(int(persistence.evidence_inserted) for persistence in persistences)
                connection.execute(text(
                    """UPDATE alias_discovery_run SET status='succeeded',counts=CAST(:counts AS jsonb),
                       finished_at=now() WHERE tenant_id=:tenant_id AND project_id=:project_id AND discovery_run_id=:run_id"""
                ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "run_id": run_id,
                    "counts": json.dumps({"proposals": len(document_proposals), "evidence_inserted": inserted}, sort_keys=True)})
            succeeded += 1
            proposals += len(document_proposals)
            evidence_inserted += inserted
            run_ids.append(run_id)
        except Exception as error:
            with engine.begin() as connection:
                _insert_run(connection, scope, run_id, document, input_sha256, "failed", {},
                            f"deterministic discovery failed: {type(error).__name__}")
            failed += 1
            run_ids.append(run_id)
    return AliasDiscoveryReport("deterministic", len(documents), succeeded, failed, reused, proposals, evidence_inserted, run_ids)
