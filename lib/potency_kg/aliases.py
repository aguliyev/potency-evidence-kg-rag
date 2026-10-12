"""Tenant/project-scoped approved alias snapshots and review operations."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .contracts import Scope
from .ids import canonical_json, derive_scoped_id


ALIAS_NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
ALIAS_CATALOG_VERSION = "approved-alias-catalog-v1"
STATIC_ALIAS_DETECTOR_VERSION = "scoped-alias-seeds-v1"
DEFAULT_ALIAS_SEED_PATH = Path(__file__).with_name("config_data") / "scoped_aliases.json"

EntityType = Literal["Product", "Assay"]
AliasStatus = Literal["pending", "approved", "rejected"]
ManagedBy = Literal["static_config", "review"]


@dataclass(frozen=True)
class AliasRecord:
    alias_id: str
    alias_value: str
    normalized_alias: str
    entity_type: EntityType | None
    canonical_key: str | None
    canonical_name: str | None
    status: AliasStatus
    managed_by: ManagedBy
    confidence: float | None


@dataclass(frozen=True, order=True)
class ApprovedAlias:
    alias_id: str
    alias_value: str
    normalized_alias: str
    entity_type: EntityType
    canonical_key: str
    canonical_name: str
    normalization_version: str


@dataclass(frozen=True)
class AliasCatalog:
    scope: Scope
    version: str
    entries: Sequence[ApprovedAlias]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))


@dataclass(frozen=True)
class AliasDecision:
    alias_id: str
    status: Literal["approved", "rejected"]
    decided_by: str
    entity_type: EntityType | None = None
    canonical_key: str | None = None
    canonical_name: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class AliasSeedReport:
    scope: Scope
    inserted: int
    unchanged: int
    retired: int
    replaced: int
    evidence_inserted: int
    catalog_version: str


@dataclass(frozen=True)
class AliasCatalogStatus:
    scope: Scope
    catalog_version: str
    latest_jev_catalog_versions: Sequence[str]
    latest_extraction_catalog_versions: Sequence[str]
    semantic_stale: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "latest_jev_catalog_versions", tuple(self.latest_jev_catalog_versions))
        object.__setattr__(self, "latest_extraction_catalog_versions", tuple(self.latest_extraction_catalog_versions))


@dataclass(frozen=True)
class _SeedAlias:
    alias_value: str
    normalized_alias: str
    entity_type: EntityType
    canonical_key: str
    canonical_name: str


def normalize_alias(value: str) -> str:
    """Normalize exact-match alias text without applying fuzzy matching."""
    if not isinstance(value, str):
        raise ValueError("alias value must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"seed {field} must be a nonempty string")
    return value.strip()


def _load_seed_document(seed_path: Path, scope: Scope) -> tuple[_SeedAlias, ...]:
    try:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid alias seed file: {seed_path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != STATIC_ALIAS_DETECTOR_VERSION:
        raise ValueError("unsupported alias seed schema version")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("seed entries must be an array")

    parsed: list[_SeedAlias] = []
    seen: set[tuple[str, str, str, EntityType]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each seed entry must be an object")
        tenant_id = _required_string(entry.get("tenant_id"), "tenant_id")
        project_id = _required_string(entry.get("project_id"), "project_id")
        entity_type = _required_string(entry.get("entity_type"), "entity_type")
        if entity_type not in ("Product", "Assay"):
            raise ValueError("seed entity_type must be Product or Assay")
        canonical_key = _required_string(entry.get("canonical_key"), "canonical_key")
        canonical_name = _required_string(entry.get("canonical_name"), "canonical_name")
        aliases = entry.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("seed aliases must be a nonempty array")
        for value in aliases:
            alias_value = _required_string(value, "aliases item")
            normalized = normalize_alias(alias_value)
            if not normalized:
                raise ValueError("seed aliases item must normalize to a nonempty string")
            typed_entity = cast(EntityType, entity_type)
            key = (tenant_id, project_id, normalized, typed_entity)
            if key in seen:
                raise ValueError(f"duplicate normalized alias in scope: {normalized} ({entity_type})")
            seen.add(key)
            if (tenant_id, project_id) != scope.as_tuple():
                continue
            item = _SeedAlias(alias_value, normalized, typed_entity, canonical_key, canonical_name)
            parsed.append(item)
    return tuple(sorted(parsed, key=lambda item: (item.normalized_alias, item.entity_type, item.canonical_key, item.canonical_name, item.alias_value)))


def _alias_id(scope: Scope, item: _SeedAlias) -> str:
    return derive_scoped_id(
        "alias", scope, item.normalized_alias, item.entity_type, item.canonical_key, item.canonical_name
    )


def _evidence_id(scope: Scope, alias_id: str, item: _SeedAlias) -> str:
    return derive_scoped_id(
        "alias-evidence", scope, alias_id, STATIC_ALIAS_DETECTOR_VERSION,
        item.alias_value, item.normalized_alias, item.entity_type, item.canonical_key, item.canonical_name,
    )


def _item_hash(item: _SeedAlias) -> str:
    return sha256(canonical_json({
        "alias_value": item.alias_value,
        "normalized_alias": item.normalized_alias,
        "entity_type": item.entity_type,
        "canonical_key": item.canonical_key,
        "canonical_name": item.canonical_name,
    }).encode()).hexdigest()


def _row_record(row: object) -> AliasRecord:
    values = cast(dict[str, object], row)
    return AliasRecord(
        alias_id=cast(str, values["alias_id"]),
        alias_value=cast(str, values["alias_value"]),
        normalized_alias=cast(str, values["normalized_alias"]),
        entity_type=cast(EntityType | None, values["entity_type"]),
        canonical_key=cast(str | None, values["canonical_key"]),
        canonical_name=cast(str | None, values["canonical_name"]),
        status=cast(AliasStatus, values["status"]),
        managed_by=cast(ManagedBy, values["managed_by"]),
        confidence=cast(float | None, values["confidence"]),
    )


def load_alias_seeds(engine: Engine, scope: Scope, seed_path: Path = DEFAULT_ALIAS_SEED_PATH) -> AliasSeedReport:
    """Reconcile authoritative static aliases after validating the complete JSON document."""
    desired = _load_seed_document(seed_path, scope)
    inserted = unchanged = retired = replaced = evidence_inserted = 0
    with engine.begin() as connection:
        existing = [dict(row) for row in connection.execute(text(
            """SELECT alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,
                      status,normalization_version
                 FROM aliases
                WHERE tenant_id=:tenant_id AND project_id=:project_id AND managed_by='static_config'
                FOR UPDATE"""
        ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id}).mappings()]
        active = {
            (cast(str, row["normalized_alias"]), cast(EntityType, row["entity_type"])): row
            for row in existing
            if row["status"] == "approved"
        }

        for item in desired:
            current = active.pop((item.normalized_alias, item.entity_type), None)
            equivalent = current is not None and (
                current["entity_type"], current["canonical_key"],
                current["canonical_name"], current["normalization_version"],
            ) == (item.entity_type, item.canonical_key, item.canonical_name, ALIAS_NORMALIZATION_VERSION)
            if equivalent:
                if current["alias_value"] != item.alias_value:
                    connection.execute(text(
                        """UPDATE aliases SET alias_value=:alias_value,updated_at=now()
                           WHERE tenant_id=:tenant_id AND project_id=:project_id AND alias_id=:alias_id"""
                    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id,
                        "alias_id": current["alias_id"], "alias_value": item.alias_value})
                unchanged += 1
                continue
            if current is not None:
                connection.execute(text(
                    """UPDATE aliases SET status='rejected',decided_at=now(),decided_by='static_config',
                       decision_note='removed or replaced by authoritative static configuration',updated_at=now()
                       WHERE tenant_id=:tenant_id AND project_id=:project_id AND alias_id=:alias_id"""
                ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": current["alias_id"]})
                retired += 1
                replaced += 1
            alias_id = _alias_id(scope, item)
            connection.execute(text(
                """INSERT INTO aliases
                   (tenant_id,project_id,alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,
                    status,managed_by,normalization_version)
                   VALUES (:tenant_id,:project_id,:alias_id,:alias_value,:normalized_alias,:entity_type,:canonical_key,
                    :canonical_name,'approved','static_config',:normalization_version)"""
            ), {
                "tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": alias_id,
                "alias_value": item.alias_value, "normalized_alias": item.normalized_alias,
                "entity_type": item.entity_type, "canonical_key": item.canonical_key,
                "canonical_name": item.canonical_name, "normalization_version": ALIAS_NORMALIZATION_VERSION,
            })
            evidence = connection.execute(text(
                """INSERT INTO alias_evidence
                   (tenant_id,project_id,evidence_id,alias_id,source_type,detector_name,detector_version,input_sha256)
                   VALUES (:tenant_id,:project_id,:evidence_id,:alias_id,'static_config','scoped_alias_seeds',
                    :detector_version,:input_sha256)
                   ON CONFLICT (tenant_id,project_id,evidence_id) DO NOTHING"""
            ), {
                "tenant_id": scope.tenant_id, "project_id": scope.project_id,
                "evidence_id": _evidence_id(scope, alias_id, item), "alias_id": alias_id,
                "detector_version": STATIC_ALIAS_DETECTOR_VERSION, "input_sha256": _item_hash(item),
            })
            inserted += 1
            evidence_inserted += int(evidence.rowcount == 1)

        for current in active.values():
            connection.execute(text(
                """UPDATE aliases SET status='rejected',decided_at=now(),decided_by='static_config',
                   decision_note='removed from authoritative static configuration',updated_at=now()
                   WHERE tenant_id=:tenant_id AND project_id=:project_id AND alias_id=:alias_id"""
            ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": current["alias_id"]})
            retired += 1
        catalog = approved_alias_catalog(connection, scope)
    return AliasSeedReport(scope, inserted, unchanged, retired, replaced, evidence_inserted, catalog.version)


def approved_alias_catalog(connection: Connection, scope: Scope) -> AliasCatalog:
    """Return the immutable, canonical ordered snapshot used for semantic identity."""
    rows = [dict(row) for row in connection.execute(text(
        """SELECT alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,normalization_version
             FROM aliases WHERE tenant_id=:tenant_id AND project_id=:project_id AND status='approved'"""
    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id}).mappings()]
    entries = tuple(sorted((
        ApprovedAlias(
            alias_id=cast(str, row["alias_id"]), alias_value=cast(str, row["alias_value"]),
            normalized_alias=cast(str, row["normalized_alias"]), entity_type=cast(EntityType, row["entity_type"]),
            canonical_key=cast(str, row["canonical_key"]), canonical_name=cast(str, row["canonical_name"]),
            normalization_version=cast(str, row["normalization_version"]),
        ) for row in rows
    ), key=lambda entry: (entry.normalized_alias, entry.entity_type, entry.canonical_key, entry.canonical_name, entry.alias_value, entry.alias_id)))
    version_payload = {
        "scope": scope.as_tuple(), "catalog_version": ALIAS_CATALOG_VERSION,
        "entries": [
            {"alias_id": entry.alias_id, "alias_value": entry.alias_value,
             "normalized_alias": entry.normalized_alias, "entity_type": entry.entity_type,
             "canonical_key": entry.canonical_key, "canonical_name": entry.canonical_name,
             "normalization_version": entry.normalization_version}
            for entry in entries
        ],
    }
    return AliasCatalog(scope, f"{ALIAS_CATALOG_VERSION}-{sha256(canonical_json(version_payload).encode()).hexdigest()}", entries)


def list_aliases(
    connection: Connection, scope: Scope, *, status: str | None = None, entity_type: str | None = None,
) -> Sequence[AliasRecord]:
    """List scoped registry rows using optional public filters."""
    if status is not None and status not in ("pending", "approved", "rejected"):
        raise ValueError("unsupported alias status")
    if entity_type is not None and entity_type not in ("Product", "Assay"):
        raise ValueError("unsupported alias entity type")
    clauses = ["tenant_id=:tenant_id", "project_id=:project_id"]
    params: dict[str, object] = {"tenant_id": scope.tenant_id, "project_id": scope.project_id}
    if status is not None:
        clauses.append("status=:status")
        params["status"] = status
    if entity_type is not None:
        clauses.append("entity_type=:entity_type")
        params["entity_type"] = entity_type
    rows = connection.execute(text(
        "SELECT alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,confidence "
        f"FROM aliases WHERE {' AND '.join(clauses)} "
        "ORDER BY normalized_alias,entity_type NULLS FIRST,canonical_key NULLS FIRST,alias_id"
    ), params).mappings()
    return tuple(_row_record(dict(row)) for row in rows)


def decide_alias(engine: Engine, scope: Scope, decision: AliasDecision) -> AliasRecord:
    """Apply one explicit review decision, or return an identical prior decision without a write."""
    if decision.status not in ("approved", "rejected"):
        raise ValueError("unsupported alias decision")
    if not decision.alias_id.strip() or not decision.decided_by.strip():
        raise ValueError("alias_id and decided_by are required")
    with engine.begin() as connection:
        row = connection.execute(text(
            """SELECT alias_id,alias_value,normalized_alias,entity_type,canonical_key,canonical_name,status,managed_by,
                      confidence,decided_by,decision_note
                 FROM aliases WHERE tenant_id=:tenant_id AND project_id=:project_id AND alias_id=:alias_id FOR UPDATE"""
        ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": decision.alias_id}).mappings().one_or_none()
        if row is None:
            raise ValueError("alias does not exist in this scope")
        values = dict(row)
        if values["managed_by"] == "static_config":
            raise ValueError("static aliases are managed by authoritative configuration")

        entity_type = decision.entity_type if decision.entity_type is not None else values["entity_type"]
        canonical_key = decision.canonical_key if decision.canonical_key is not None else values["canonical_key"]
        canonical_name = decision.canonical_name if decision.canonical_name is not None else values["canonical_name"]
        if decision.status == "approved" and (
            entity_type not in ("Product", "Assay")
            or not isinstance(canonical_key, str) or not canonical_key.strip()
            or not isinstance(canonical_name, str) or not canonical_name.strip()
        ):
            raise ValueError("approval requires entity_type, canonical_key, and canonical_name")
        desired = (
            decision.status, entity_type, canonical_key, canonical_name, decision.decided_by, decision.note,
        )
        current = (
            values["status"], values["entity_type"], values["canonical_key"], values["canonical_name"],
            values["decided_by"], values["decision_note"],
        )
        if values["status"] != "pending":
            if current == desired:
                return _row_record(values)
            raise ValueError("a contradictory decision already exists for this alias")
        connection.execute(text(
            """UPDATE aliases SET entity_type=:entity_type,canonical_key=:canonical_key,canonical_name=:canonical_name,
               status=:status,decided_at=now(),decided_by=:decided_by,decision_note=:decision_note,updated_at=now()
               WHERE tenant_id=:tenant_id AND project_id=:project_id AND alias_id=:alias_id"""
        ), {
            "tenant_id": scope.tenant_id, "project_id": scope.project_id, "alias_id": decision.alias_id,
            "entity_type": entity_type, "canonical_key": canonical_key, "canonical_name": canonical_name,
            "status": decision.status, "decided_by": decision.decided_by, "decision_note": decision.note,
        })
        values.update({
            "entity_type": entity_type, "canonical_key": canonical_key, "canonical_name": canonical_name,
            "status": decision.status, "decided_by": decision.decided_by, "decision_note": decision.note,
        })
        return _row_record(values)


_SEMANTIC_HISTORY_TABLES = {
    "jev": "jev_response",
    "extraction": "llm_extraction_run",
}


def _latest_catalog_versions(connection: Connection, history_kind: Literal["jev", "extraction"], scope: Scope) -> tuple[str, ...]:
    table = _SEMANTIC_HISTORY_TABLES[history_kind]
    result = connection.execute(text(
        f"""WITH latest AS (
              SELECT max(created_at) AS created_at FROM {table}
               WHERE tenant_id=:tenant_id AND project_id=:project_id AND status='succeeded'
            )
            SELECT DISTINCT alias_catalog_version FROM {table}, latest
             WHERE {table}.tenant_id=:tenant_id AND {table}.project_id=:project_id
               AND {table}.status='succeeded' AND {table}.created_at=latest.created_at
             ORDER BY alias_catalog_version"""
    ), {"tenant_id": scope.tenant_id, "project_id": scope.project_id})
    return tuple(cast(str, value) for value in result.scalars())


def alias_catalog_status(connection: Connection, scope: Scope) -> AliasCatalogStatus:
    """Report whether the most recent Jev/extraction state uses this exact catalog."""
    catalog = approved_alias_catalog(connection, scope)
    jev_versions = _latest_catalog_versions(connection, "jev", scope)
    extraction_versions = _latest_catalog_versions(connection, "extraction", scope)
    observed = (*jev_versions, *extraction_versions)
    return AliasCatalogStatus(
        scope, catalog.version, jev_versions, extraction_versions,
        bool(observed) and any(version != catalog.version for version in observed),
    )
