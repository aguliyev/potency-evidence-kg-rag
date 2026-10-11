"""Manifest-first, source-transactional importer for supplied evidence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .contracts import Scope, ScopedSourceVersion
from .graph_store import GraphWriteReport, write_structural_version
from .ids import derive_scoped_id


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hash_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"supplied path escapes package root: {path}") from error
    return resolved


@dataclass(frozen=True)
class CorpusPreflightReport:
    descriptor_sha256: str
    manifest_sha256: str
    records_sha256: str
    raw_sha256: Mapping[str, str]
    document_count: int
    passage_outcome_count: int
    per_source_counts: Mapping[str, int]
    valid_passage_outcomes: int
    invalid_passage_outcomes: int


@dataclass(frozen=True)
class ImportRequest:
    descriptor: Path
    source_id: str | None = None
    failure_at: str | None = None


@dataclass(frozen=True)
class ImportReport:
    source_count: int
    passage_outcome_count: int
    imported_passages: int
    invalid_passage_outcomes: int
    new_source_versions: int
    unchanged_source_versions: int
    structural_nodes: int
    structural_edges: int
    per_source_counts: Mapping[str, int]


@dataclass(frozen=True)
class _LoadedInputs:
    descriptor: Mapping[str, Any]
    manifest_path: Path
    records_path: Path
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]


def _load_inputs(descriptor: Path) -> _LoadedInputs:
    descriptor = descriptor.resolve()
    root = descriptor.parent.parent
    descriptor_data = json.loads(descriptor.read_text(encoding="utf-8"))
    manifest_path = _within(descriptor.parent / str(descriptor_data["corpus_manifest"]), root)
    records_path = _within(descriptor.parent / str(descriptor_data["source_records"]), root)
    if manifest_path.parent != root / "corpus" or records_path.parent != root / "corpus":
        raise ValueError("manifest and normalized records must be supplied corpus inputs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = tuple(
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return _LoadedInputs(descriptor_data, manifest_path, records_path, manifest, records)


def _source_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    return str(value.get("source_id", "")), str(value.get("source_version", ""))


def _passage_error(record: Mapping[str, Any], scope: Scope, source_id: str, source_version: str) -> str | None:
    if _source_identity(record) != (source_id, source_version):
        return "source_identity_mismatch"
    record_scope = record.get("scope")
    if record_scope != {"tenant_id": scope.tenant_id, "project_id": scope.project_id}:
        return "scope_mismatch"
    passage_id = record.get("passage_id")
    if not isinstance(passage_id, str) or not passage_id.strip():
        return "invalid_passage_id"
    body = record.get("text")
    if not isinstance(body, str):
        return "invalid_passage_text"
    expected_hash = record.get("text_sha256")
    if not isinstance(expected_hash, str) or _hash_text(body) != expected_hash:
        return "passage_digest_mismatch"
    return None


def _replacement_declarations(descriptor: Mapping[str, Any]) -> Mapping[tuple[str, str], str]:
    declarations: dict[tuple[str, str], str] = {}
    raw = descriptor.get("replacements", [])
    if not isinstance(raw, list):
        raise ValueError("descriptor replacements must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("descriptor replacement entries must be objects")
        source_id = item.get("source_id")
        previous = item.get("from_source_version")
        replacement = item.get("to_source_version")
        if not all(isinstance(value, str) and value for value in (source_id, previous, replacement)):
            raise ValueError("descriptor replacement requires source_id, from_source_version, and to_source_version")
        key = (source_id, replacement)
        if key in declarations:
            raise ValueError(f"duplicate descriptor replacement for {source_id}/{replacement}")
        declarations[key] = previous
    return declarations


def _preflight(loaded: _LoadedInputs, descriptor_path: Path) -> CorpusPreflightReport:
    descriptor = loaded.descriptor
    manifest = loaded.manifest
    records = loaded.records
    root = descriptor_path.resolve().parent.parent
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("manifest sources must be a list")
    if len(sources) != manifest.get("source_count") or len(records) != manifest.get("record_count"):
        raise ValueError("manifest source/record count does not reconcile")
    if descriptor.get("corpus_snapshot") != manifest.get("snapshot"):
        raise ValueError("descriptor and manifest corpus snapshots do not reconcile")
    if _hash(loaded.records_path) != manifest.get("record_sha256"):
        raise ValueError("normalized records digest mismatch")

    scope_data = descriptor.get("scope")
    if not isinstance(scope_data, dict):
        raise ValueError("descriptor scope is required")
    scope = Scope(str(scope_data.get("tenant_id", "")), str(scope_data.get("project_id", "")))
    manifest_scope = manifest.get("scope")
    if manifest_scope is not None and manifest_scope != scope_data:
        raise ValueError("descriptor and manifest scopes do not reconcile")

    source_identities = [_source_identity(source) for source in sources]
    if any(not all(identity) for identity in source_identities) or len(set(source_identities)) != len(source_identities):
        raise ValueError("manifest source identities must be nonempty and unique")
    source_identity_set = set(source_identities)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        identity = _source_identity(record)
        if identity not in source_identity_set:
            raise ValueError("normalized record does not belong to a manifest source version")
        grouped[identity].append(record)

    duplicated_source_ids = {source_id for source_id, count in Counter(identity[0] for identity in source_identities).items() if count > 1}
    per_source: dict[str, int] = {}
    raw_hashes: dict[str, str] = {}
    valid = invalid = 0
    for source in sources:
        source_id, source_version = _source_identity(source)
        rows = grouped[(source_id, source_version)]
        if len(rows) != source.get("passage_count"):
            raise ValueError(f"record count mismatch for {source_id}/{source_version}")
        raw_path = _within(root / str(source.get("filename", "")), root / "corpus" / "raw")
        actual_hash = _hash(raw_path)
        if actual_hash != source.get("sha256"):
            raise ValueError(f"raw source digest mismatch for {source_id}/{source_version}")
        report_key = f"{source_id}@{source_version}" if source_id in duplicated_source_ids else source_id
        raw_hashes[report_key] = actual_hash
        per_source[report_key] = len(rows)
        seen_passages: set[str] = set()
        for record in rows:
            error = _passage_error(record, scope, source_id, source_version)
            passage_id = record.get("passage_id")
            if error is None and passage_id in seen_passages:
                error = "duplicate_passage_id"
            if error is None:
                seen_passages.add(str(passage_id))
                valid += 1
            else:
                invalid += 1
    replacements = _replacement_declarations(descriptor)
    for (source_id, replacement), previous in replacements.items():
        if (source_id, replacement) not in source_identity_set:
            raise ValueError(f"replacement target is not a manifest source version: {source_id}/{replacement}")
        if previous == replacement:
            raise ValueError(f"replacement must name two distinct versions: {source_id}/{replacement}")
    return CorpusPreflightReport(
        descriptor_sha256=_hash(descriptor_path),
        manifest_sha256=_hash(loaded.manifest_path),
        records_sha256=_hash(loaded.records_path),
        raw_sha256=raw_hashes,
        document_count=len(sources),
        passage_outcome_count=len(records),
        per_source_counts=dict(sorted(per_source.items())),
        valid_passage_outcomes=valid,
        invalid_passage_outcomes=invalid,
    )


def preflight_official_inputs(descriptor: Path) -> CorpusPreflightReport:
    loaded = _load_inputs(descriptor)
    return _preflight(loaded, descriptor.resolve())


def _existing_version_is_unchanged(
    connection: Connection,
    source: ScopedSourceVersion,
    source_data: Mapping[str, Any],
    valid_records: tuple[Mapping[str, Any], ...],
    corpus_snapshot: str,
) -> bool:
    params = {
        "tenant": source.tenant_id,
        "project": source.project_id,
        "source_id": source.source_id,
        "source_version": source.source_version,
    }
    stored = connection.execute(
        text("""
            SELECT title, source_sha256, expected_passage_count, status, authority,
                   source_type, source_url, filename, corpus_snapshot, metadata
            FROM source_document
            WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id AND source_version=:source_version
        """),
        params,
    ).mappings().one_or_none()
    if stored is None:
        return False
    expected_document = {
        "title": source_data["title"],
        "source_sha256": source_data["sha256"],
        "expected_passage_count": source_data["passage_count"],
        "status": source_data["status"],
        "authority": source_data.get("authority"),
        "source_type": source_data.get("source_type"),
        "source_url": source_data.get("source_url"),
        "filename": source_data.get("filename"),
        "corpus_snapshot": corpus_snapshot,
        "metadata": {"raw_sha256": source_data["sha256"]},
    }
    if dict(stored) != expected_document:
        raise ValueError(f"immutable source version conflicts with supplied manifest: {source.source_id}/{source.source_version}")
    stored_passages = {
        str(row["passage_id"]): dict(row)
        for row in connection.execute(
            text("""
                SELECT passage_id, text, text_sha256, locator, subject_scope, metadata
                FROM source_passage
                WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id AND source_version=:source_version
            """),
            params,
        ).mappings()
    }
    expected_passages = {
        str(record["passage_id"]): {
            "passage_id": record["passage_id"],
            "text": record["text"],
            "text_sha256": record["text_sha256"],
            "locator": record.get("locator", {}),
            "subject_scope": record.get("subject_scope", {}),
            "metadata": {"status": record.get("status"), "authority": record.get("authority")},
        }
        for record in valid_records
    }
    if stored_passages != expected_passages:
        raise ValueError(f"immutable passage rows conflict with supplied source version: {source.source_id}/{source.source_version}")
    expected_node_ids = {derive_scoped_id("document", source.scope, source.source_id, source.source_version)}
    expected_node_ids.update(
        derive_scoped_id("passage", source.scope, source.source_id, source.source_version, passage_id)
        for passage_id in expected_passages
    )
    actual_node_count = connection.execute(
        text("SELECT count(*) FROM kg_node WHERE tenant_id=:tenant AND project_id=:project AND node_id = ANY(:node_ids)"),
        {**params, "node_ids": list(expected_node_ids)},
    ).scalar_one()
    if actual_node_count != len(expected_node_ids):
        raise ValueError(f"structural graph is incomplete for immutable source version: {source.source_id}/{source.source_version}")
    return True


def _partition_records(
    records: tuple[Mapping[str, Any], ...],
    scope: Scope,
    source_id: str,
    source_version: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[tuple[Mapping[str, Any], str], ...]]:
    valid: list[Mapping[str, Any]] = []
    invalid: list[tuple[Mapping[str, Any], str]] = []
    seen_passages: set[str] = set()
    for record in records:
        error = _passage_error(record, scope, source_id, source_version)
        passage_id = record.get("passage_id")
        if error is None and passage_id in seen_passages:
            error = "duplicate_passage_id"
        if error is None:
            seen_passages.add(str(passage_id))
            valid.append(record)
        else:
            invalid.append((record, error))
    return tuple(valid), tuple(invalid)


def _insert_passage(connection: Connection, scope: Scope, record: Mapping[str, Any]) -> None:
    connection.execute(
        text("""
            INSERT INTO source_passage
              (tenant_id, project_id, source_id, source_version, passage_id, text, text_sha256, locator, subject_scope, metadata)
            VALUES
              (:tenant, :project, :source_id, :source_version, :passage_id, :body, :text_sha256,
               CAST(:locator AS jsonb), CAST(:subject_scope AS jsonb), CAST(:metadata AS jsonb))
        """),
        {
            "tenant": scope.tenant_id,
            "project": scope.project_id,
            "source_id": record["source_id"],
            "source_version": record["source_version"],
            "passage_id": record["passage_id"],
            "body": record["text"],
            "text_sha256": record["text_sha256"],
            "locator": json.dumps(record.get("locator", {})),
            "subject_scope": json.dumps(record.get("subject_scope", {})),
            "metadata": json.dumps({"status": record.get("status"), "authority": record.get("authority")}),
        },
    )


def _record_import_errors(
    connection: Connection,
    *,
    import_run_id: UUID,
    scope: Scope,
    source_id: str,
    source_version: str,
    invalid_records: tuple[tuple[Mapping[str, Any], str], ...],
) -> None:
    for record, code in invalid_records:
        passage_id = record.get("passage_id")
        connection.execute(
            text("""
                INSERT INTO import_error
                  (error_id, run_id, tenant_id, project_id, source_id, source_version, passage_id, code, detail)
                VALUES
                  (:error_id, :run_id, :tenant, :project, :source_id, :source_version, :passage_id, :code, :detail)
            """),
            {
                "error_id": uuid4(),
                "run_id": import_run_id,
                "tenant": scope.tenant_id,
                "project": scope.project_id,
                "source_id": source_id,
                "source_version": source_version,
                "passage_id": passage_id if isinstance(passage_id, str) and passage_id else None,
                "code": code,
                "detail": f"passage outcome rejected: {code}",
            },
        )


def import_source_version(engine: Engine, run: ImportRequest) -> ImportReport:
    """Import each requested source version in one caller-visible owner transaction."""
    descriptor_path = run.descriptor.resolve()
    loaded = _load_inputs(descriptor_path)
    preflight = _preflight(loaded, descriptor_path)
    descriptor = loaded.descriptor
    manifest = loaded.manifest
    scope_data = descriptor["scope"]
    scope = Scope(str(scope_data["tenant_id"]), str(scope_data["project_id"]))
    replacements = _replacement_declarations(descriptor)
    records_by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in loaded.records:
        records_by_source[_source_identity(record)].append(record)
    sources = [
        source
        for source in manifest["sources"]
        if run.source_id is None or str(source["source_id"]) == run.source_id
    ]
    if run.source_id is not None and not sources:
        raise ValueError(f"source not present in manifest: {run.source_id}")

    new = unchanged = imported_passages = invalid_outcomes = structural_nodes = structural_edges = 0
    passage_outcome_count = 0
    for source_data in sources:
        source_id, source_version = _source_identity(source_data)
        rows = tuple(records_by_source[(source_id, source_version)])
        passage_outcome_count += len(rows)
        valid_records, invalid_records = _partition_records(rows, scope, source_id, source_version)
        invalid_outcomes += len(invalid_records)
        source = ScopedSourceVersion(scope.tenant_id, scope.project_id, source_id, source_version)

        with engine.begin() as connection:
            import_run_id = uuid4()
            connection.execute(
                text("""
                    INSERT INTO import_run
                      (run_id, tenant_id, project_id, descriptor_sha256, status, finished_at)
                    VALUES (:run_id, :tenant, :project, :descriptor, 'succeeded', now())
                """),
                {
                    "run_id": import_run_id,
                    "tenant": scope.tenant_id,
                    "project": scope.project_id,
                    "descriptor": preflight.descriptor_sha256,
                },
            )
            existing_version = _existing_version_is_unchanged(
                connection,
                source,
                source_data,
                valid_records,
                str(manifest["snapshot"]),
            )
            if existing_version:
                unchanged += 1
            else:
                connection.execute(
                    text("""
                        INSERT INTO source_document
                          (tenant_id, project_id, source_id, source_version, title, source_sha256,
                           expected_passage_count, status, authority, source_type, source_url, filename,
                           corpus_snapshot, metadata)
                        VALUES
                          (:tenant, :project, :source_id, :source_version, :title, :source_sha256,
                           :expected, :status, :authority, :source_type, :source_url, :filename,
                           :snapshot, CAST(:metadata AS jsonb))
                    """),
                    {
                        "tenant": scope.tenant_id,
                        "project": scope.project_id,
                        "source_id": source_id,
                        "source_version": source_version,
                        "title": source_data["title"],
                        "source_sha256": source_data["sha256"],
                        "expected": source_data["passage_count"],
                        "status": source_data["status"],
                        "authority": source_data.get("authority"),
                        "source_type": source_data.get("source_type"),
                        "source_url": source_data.get("source_url"),
                        "filename": source_data.get("filename"),
                        "snapshot": manifest["snapshot"],
                        "metadata": json.dumps({"raw_sha256": source_data["sha256"]}),
                    },
                )
                for record in valid_records:
                    _insert_passage(connection, scope, record)
                imported_passages += len(valid_records)

            _record_import_errors(
                connection,
                import_run_id=import_run_id,
                scope=scope,
                source_id=source_id,
                source_version=source_version,
                invalid_records=invalid_records,
            )
            if run.failure_at == "after_passages":
                raise RuntimeError("injected failure after passage insert")

            graph = (
                GraphWriteReport(node_count=0, edge_count=0)
                if existing_version
                else write_structural_version(connection, source)
            )
            if run.failure_at == "after_structural":
                raise RuntimeError("injected failure after structural writes")

            complete = not invalid_records and len(valid_records) == int(source_data["passage_count"])
            head_params = {"tenant": scope.tenant_id, "project": scope.project_id, "source_id": source_id}
            prior_head = connection.execute(
                text("SELECT source_version FROM source_head WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id"),
                head_params,
            ).scalar()
            declared_previous = replacements.get((source_id, source_version))
            if declared_previous is not None and prior_head not in {declared_previous, source_version}:
                raise ValueError(
                    f"replacement declaration for {source_id}/{source_version} expected active version {declared_previous}"
                )
            activate = complete and (
                prior_head == source_version
                or prior_head is None
                or declared_previous == prior_head
            )
            reason = (
                "incomplete_source_version"
                if not complete
                else "eligible_for_processing"
                if activate
                else "inactive_source_version"
            )
            if not existing_version:
                for record in valid_records:
                    connection.execute(
                        text("""
                            INSERT INTO passage_processing
                              (tenant_id, project_id, source_id, source_version, passage_id, state, reason)
                            VALUES
                              (:tenant, :project, :source_id, :source_version, :passage_id, 'not_run', :reason)
                        """),
                        {
                            "tenant": scope.tenant_id,
                            "project": scope.project_id,
                            "source_id": source_id,
                            "source_version": source_version,
                            "passage_id": record["passage_id"],
                            "reason": reason,
                        },
                    )
            if run.failure_at == "before_activation":
                raise RuntimeError("injected failure before head activation")

            if activate and prior_head is None:
                connection.execute(
                    text("""
                        INSERT INTO source_head (tenant_id, project_id, source_id, source_version)
                        VALUES (:tenant, :project, :source_id, :source_version)
                    """),
                    {**head_params, "source_version": source_version},
                )
            elif activate and prior_head != source_version:
                connection.execute(
                    text("""
                        UPDATE source_head SET source_version=:source_version, activated_at=now()
                        WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                    """),
                    {**head_params, "source_version": source_version},
                )
                if existing_version:
                    connection.execute(
                        text("""
                            UPDATE passage_processing
                            SET state='not_run', reason='eligible_for_processing',
                                active_jev_selection_id=NULL, active_extraction_run_id=NULL, updated_at=now()
                            WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                              AND source_version=:source_version
                        """),
                        {**head_params, "source_version": source_version},
                    )
                connection.execute(
                    text("""
                        UPDATE passage_processing
                        SET state='not_run', reason='inactive_source_version',
                            active_jev_selection_id=NULL, active_extraction_run_id=NULL, updated_at=now()
                        WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source_id
                          AND source_version <> :source_version
                          AND reason <> 'incomplete_source_version'
                    """),
                    {**head_params, "source_version": source_version},
                )
            if not existing_version:
                new += 1
            structural_nodes += graph.node_count
            structural_edges += graph.edge_count

    selected_source_ids = {str(source["source_id"]) for source in sources}
    selected_counts = {
        key: value
        for key, value in preflight.per_source_counts.items()
        if key in selected_source_ids or key.split("@", 1)[0] in selected_source_ids
    }
    return ImportReport(
        source_count=len(sources),
        passage_outcome_count=passage_outcome_count,
        imported_passages=imported_passages,
        invalid_passage_outcomes=invalid_outcomes,
        new_source_versions=new,
        unchanged_source_versions=unchanged,
        structural_nodes=structural_nodes,
        structural_edges=structural_edges,
        per_source_counts=selected_counts,
    )
