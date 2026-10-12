"""Offline fixture execution through the same evidence-pipeline interfaces as normal runs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .age_projection import GraphRefreshReport, refresh_graph
from .contracts import ScopedPassageIdentity
from .embedding_client import EmbeddingClient
from .embeddings import EmbeddingReport, embed_active_passages
from .export import ExportReport, export_graph_version
from .extraction import ExtractionReport, extract_selected_passages
from .fixture_catalog import FixtureCatalog, FixtureRunConfiguration
from .ids import canonical_json
from .ingest import ImportReport, ImportRequest, import_source_version
from .jev_client import JevClient
from .jev_processing import JevProcessingReport, classify_active_passages
from .llm_client import StructuredExtractionClient
from .processing_policy import JevSelectionPolicy
from .project_profiles import assert_protected_inputs_unchanged, protected_input_hashes
from .output_safety import guard_output_path
from .resolution import materialize_extraction
from .retrieval import RetrievalLimits, RetrievalTrace, retrieve_graph_smoke


@dataclass(frozen=True)
class ModelClients:
    embedding: EmbeddingClient
    jev: JevClient
    extraction: StructuredExtractionClient
    jev_policy: JevSelectionPolicy = JevSelectionPolicy()
    extraction_concurrency: int = 1


@dataclass(frozen=True)
class FixtureRunReport:
    configuration: FixtureRunConfiguration
    output_dir: Path
    imported: ImportReport
    embeddings: EmbeddingReport
    jev: JevProcessingReport
    extraction: ExtractionReport
    graph: GraphRefreshReport
    retrieval: RetrievalTrace
    export: ExportReport


@dataclass(frozen=True)
class FixtureGraphReport:
    runs: tuple[FixtureRunReport, ...]
    protected_hashes: dict[str, str]
    reconnect_verified: bool


def _require_offline_test_doubles(clients: ModelClients) -> None:
    providers = (
        clients.embedding.identity.provider,
        clients.jev.identity.provider,
        clients.extraction.identity.provider,
    )
    if providers != ("test-double", "test-double", "test-double"):
        raise ValueError("fixture orchestration requires test-double adapters and makes no external calls")


def _read_records(configuration: FixtureRunConfiguration) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path in configuration.input_records:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return tuple(records)


def _write_import_inputs(configuration: FixtureRunConfiguration, output_dir: Path) -> Path:
    """Create an output-local, importer-compatible manifest without modifying fixtures."""
    run_root = output_dir.resolve(strict=False)
    records = _read_records(configuration)
    config_dir = guard_output_path(output_dir / "configs", boundary=run_root, label="fixture output")
    corpus_dir = guard_output_path(output_dir / "corpus", boundary=run_root, label="fixture output")
    raw_dir = guard_output_path(corpus_dir / "raw" / "fixture-inputs", boundary=run_root, label="fixture output")
    config_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    records_path = guard_output_path(corpus_dir / "records.jsonl", boundary=run_root, label="fixture output")
    records_path.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["source_id"]), str(row["source_version"]))].append(row)
    sources: list[dict[str, object]] = []
    for index, ((source_id, source_version), source_rows) in enumerate(sorted(grouped.items())):
        identity_digest = hashlib.sha256(
            canonical_json([source_id, source_version]).encode("utf-8")
        ).hexdigest()[:20]
        raw_name = f"{index:03d}-{identity_digest}.jsonl"
        raw_path = guard_output_path(raw_dir / raw_name, boundary=run_root, label="fixture output")
        raw_path.write_text("".join(canonical_json(row) + "\n" for row in source_rows), encoding="utf-8")
        sample = source_rows[0]
        sources.append(
            {
                "source_id": source_id,
                "source_version": source_version,
                "title": str(sample["title"]),
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "passage_count": len(source_rows),
                "status": str(sample["status"]),
                "authority": sample.get("authority"),
                "source_type": sample.get("source_type"),
                "source_url": sample.get("source_url"),
                "filename": f"corpus/raw/fixture-inputs/{raw_name}",
            }
        )
    manifest = {
        "snapshot": configuration.corpus_snapshot,
        "source_count": len(sources),
        "record_count": len(records),
        "record_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "scope": {"tenant_id": configuration.scope.tenant_id, "project_id": configuration.scope.project_id},
        "sources": sources,
    }
    guard_output_path(corpus_dir / "manifest.json", boundary=run_root, label="fixture output").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    descriptor = {
        "scope": manifest["scope"],
        "corpus_snapshot": configuration.corpus_snapshot,
        "corpus_manifest": "../corpus/manifest.json",
        "source_records": "../corpus/records.jsonl",
        "replacements": [
            {
                "source_id": replacement.source_id,
                "from_source_version": replacement.previous_source_version,
                "to_source_version": replacement.source_version,
            }
            for replacement in configuration.replacements
        ],
    }
    descriptor_path = guard_output_path(config_dir / "import.json", boundary=run_root, label="fixture output")
    descriptor_path.write_text(canonical_json(descriptor) + "\n", encoding="utf-8")
    return descriptor_path


def _materialize_all(engine: Engine, extraction: ExtractionReport) -> None:
    for run_id in extraction.extraction_run_ids:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    """SELECT tenant_id,project_id,source_id,source_version,passage_id
                       FROM llm_extraction_run WHERE extraction_run_id=:run_id"""
                ),
                {"run_id": run_id},
            ).mappings().one()
        passage = ScopedPassageIdentity(
            str(row["tenant_id"]), str(row["project_id"]), str(row["source_id"]),
            str(row["source_version"]), str(row["passage_id"]),
        )
        materialize_extraction(engine, passage, run_id)


def _verify_reconnect(engine: Engine, reports: tuple[FixtureRunReport, ...], clients: ModelClients) -> bool:
    """Dispose pooled connections, then repeat scoped smoke retrieval from persisted state."""
    engine.dispose()
    with engine.connect() as connection:
        for scope in {report.configuration.scope for report in reports}:
            count = connection.execute(
                text("SELECT count(*) FROM source_head WHERE tenant_id=:tenant AND project_id=:project"),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).scalar_one()
            if count == 0:
                return False
    latest_by_scope = {report.configuration.scope: report for report in reports}
    for report in latest_by_scope.values():
        repeated = retrieve_graph_smoke(
            engine,
            report.configuration.scope,
            report.configuration.question or "What supplied fixture evidence is current?",
            clients.embedding,
            report.embeddings.index_version,
            RetrievalLimits(),
        )
        if not repeated.source_refs or any(ref.project_id != report.configuration.scope.project_id for ref in repeated.source_refs):
            return False
    return True


def run_kg_fixtures(
    engine: Engine,
    catalog: FixtureCatalog,
    output_root: Path,
    clients: ModelClients,
) -> FixtureGraphReport:
    """Run every supplied fixture through import, semantic processing, retrieval, and export."""
    _require_offline_test_doubles(clients)
    root = guard_output_path(output_root, package_root=catalog.package_root)
    before = dict(protected_input_hashes(catalog.package_root))
    reports: list[FixtureRunReport] = []
    try:
        for configuration in catalog.runs:
            output_dir = guard_output_path(
                root / configuration.scenario_id / configuration.stage,
                package_root=catalog.package_root,
                boundary=root,
                label="fixture output",
            )
            output_dir.mkdir(parents=True, exist_ok=False)
            descriptor = _write_import_inputs(configuration, output_dir)
            imported = import_source_version(engine, ImportRequest(descriptor))
            embeddings = embed_active_passages(engine, configuration.scope, clients.embedding)
            jev = classify_active_passages(engine, configuration.scope, clients.jev, clients.jev_policy)
            extraction = extract_selected_passages(
                engine, configuration.scope, clients.extraction, clients.extraction_concurrency
            )
            _materialize_all(engine, extraction)
            graph = refresh_graph(engine, configuration.scope, mode="semantic")
            retrieval = retrieve_graph_smoke(
                engine,
                configuration.scope,
                configuration.question or "What supplied fixture evidence is current?",
                clients.embedding,
                embeddings.index_version,
                RetrievalLimits(),
            )
            exported = export_graph_version(engine, configuration.scope, output_dir / "kg", graph.graph_version)
            reports.append(FixtureRunReport(
                configuration, output_dir, imported, embeddings, jev, extraction, graph, retrieval, exported
            ))
        reconnect_verified = _verify_reconnect(engine, tuple(reports), clients)
        return FixtureGraphReport(tuple(reports), before, reconnect_verified)
    finally:
        assert_protected_inputs_unchanged(catalog.package_root, before)
