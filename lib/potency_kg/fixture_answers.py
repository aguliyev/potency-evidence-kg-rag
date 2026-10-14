"""Persist fixture-stage answer attempts and inspect their saved evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .answer_runs import _trace_base, answer_from_retrieval
from .embedding_client import EmbeddingClient
from .embeddings import embed_query
from .fixture_catalog import FixtureRunConfiguration
from .fixture_runs import FixtureGraphReport, FixtureRunReport
from .ids import canonical_json
from .llm_client import StructuredExtractionClient
from .output_safety import guard_output_path
from .retrieval import RetrievalResult, retrieve
from .retrieval_config import HybridRetrievalConfig


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summed_generation_usage(traces: list[dict[str, object]]) -> dict[str, int | float]:
    """Preserve the token field names actually returned by the provider."""
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
    return {
        field: sum(trace.get("usage", {}).get(field) or 0 for trace in traces)
        for field in fields
        if any(field in (trace.get("usage") or {}) for trace in traces)
    }


class FixtureAnswerRecorder:
    """Copy each stage's evidence before a later source replacement can activate."""

    def __init__(
        self, output_root: Path, *, package_root: Path,
        config: HybridRetrievalConfig | None = None, git_commit: str = "unknown",
    ) -> None:
        self.package_root = package_root.resolve()
        self.root = guard_output_path(output_root, package_root=self.package_root)
        self.config = config or HybridRetrievalConfig()
        self.git_commit = git_commit
        self.answers: list[dict[str, object]] = []
        self.traces: list[dict[str, object]] = []
        self.history_artifacts: dict[str, dict[str, object]] = {}

    def _path(self, relative: str) -> Path:
        return guard_output_path(
            self.root / relative, package_root=self.package_root,
            boundary=self.root, label="fixture answer output",
        )

    def preflight(self, runs: tuple[FixtureRunConfiguration, ...]) -> None:
        """Reject redirected output paths before fixture import or provider calls."""
        for name in ("answers.jsonl", "traces.jsonl", "report.json"):
            destination = self._path(name)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"fixture answer output already exists: {name}")
        for run in runs:
            base = f"history/{run.scenario_id}/{run.stage}"
            names = (
                "manifest.json", "source-records.jsonl", "import-report.json",
                "graph/nodes.jsonl", "graph/edges.jsonl", "retrieval-traces.jsonl",
                "answers.jsonl", "traces.jsonl",
            )
            if run.scenario_id.startswith("isolation-"):
                names += ("cross-scope-retrieval-traces.jsonl",)
            for name in names:
                destination = self._path(f"{base}/{name}")
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"fixture answer output already exists: {base}/{name}")

    def _write(self, relative: str, value: object) -> None:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    def _write_rows(self, relative: str, rows: list[dict[str, object]]) -> None:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")

    def _copy(self, source: Path, relative: str) -> None:
        destination = self._path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    @staticmethod
    def _retrieval_row(
        configuration: FixtureRunConfiguration, retrieved: RetrievalResult, run_id: str,
        *, cross_scope_check: bool = False,
    ) -> dict[str, object]:
        return {
            "scenario_id": configuration.scenario_id, "stage": configuration.stage,
            "run_id": run_id, "scope": {
                "tenant_id": configuration.scope.tenant_id,
                "project_id": configuration.scope.project_id,
            },
            "question": configuration.question or "What supplied fixture evidence is current?",
            "mode": "graph", "status": "succeeded", "cross_scope_check": cross_scope_check,
            "query_hash": retrieved.query_hash,
            "retriever_version": retrieved.retriever_version,
            "index_version": retrieved.index_version,
            "graph_version": retrieved.graph_version,
            "passages": [
                {"source_id": ref.source_id, "source_version": ref.source_version,
                 "passage_id": ref.passage_id, "tenant_id": ref.tenant_id,
                 "project_id": ref.project_id, "text": ref.text}
                for ref in retrieved.source_refs
            ],
            "node_ids": list(retrieved.node_ids),
            "edge_ids": list(retrieved.edge_ids),
            "candidate_passage_ids": [candidate.passage_id for candidate in retrieved.candidates],
        }

    def record_stage(
        self, configuration: FixtureRunConfiguration, report: FixtureRunReport,
        generation_client: StructuredExtractionClient, question_ids: Mapping[str, str],
    ) -> None:
        """Save one retrieval and, for behavior stages, one answer attempt."""
        if not isinstance(report.retrieval, RetrievalResult):
            raise ValueError("fixture answering needs the completed graph retrieval result")
        base = f"history/{configuration.scenario_id}/{configuration.stage}"
        run_id = str(uuid4())
        retrieved = report.retrieval
        retrieval_row = self._retrieval_row(configuration, retrieved, run_id)
        self._copy(report.output_dir / "corpus" / "records.jsonl", f"{base}/source-records.jsonl")
        for name in ("import-report.json", "graph/nodes.jsonl", "graph/edges.jsonl"):
            self._copy(report.export.version_dir / name, f"{base}/{name}")
        self._write_rows(f"{base}/retrieval-traces.jsonl", [retrieval_row])

        answer_rows: list[dict[str, object]] = []
        trace_rows: list[dict[str, object]] = []
        if configuration.question is not None:
            question_id = question_ids[configuration.scenario_id]
            try:
                answer, trace = answer_from_retrieval(
                    question_id, configuration.question, "graph", retrieved,
                    configuration.scope, configuration.corpus_snapshot,
                    generation_client, self.config.default_budget_chars,
                    self.config, git_commit=self.git_commit,
                )
            except Exception as error:
                # A callback failure must not prevent the correction's next stage.
                trace = _trace_base(
                    question_id, configuration.question, "graph", configuration.scope,
                    configuration.corpus_snapshot, retrieved.index_version,
                    self.config.default_budget_chars, self.config,
                    generation_client, self.git_commit,
                )
                trace.update({
                    "failure_kind": "schema" if isinstance(error, (ValueError, TypeError, AssertionError)) else "provider",
                    "error_type": type(error).__name__, "failed_stage": "fixture_generation",
                })
                answer = None
            metadata = {"scenario_id": configuration.scenario_id,
                        "stage": configuration.stage, "run_id": run_id}
            trace.update(metadata)
            trace_rows.append(trace)
            self.traces.append(trace)
            if answer is not None:
                answer.update(metadata)
                answer_rows.append(answer)
                self.answers.append(answer)

        self._write_rows(f"{base}/answers.jsonl", answer_rows)
        self._write_rows(f"{base}/traces.jsonl", trace_rows)
        source_versions = sorted({
            (str(row["source_id"]), str(row["source_version"]))
            for row in _rows(self._path(f"{base}/source-records.jsonl"))
        })
        manifest: dict[str, object] = {
            "scenario_id": configuration.scenario_id, "stage": configuration.stage,
            "run_id": run_id, "scope": retrieval_row["scope"],
            "corpus_snapshot": configuration.corpus_snapshot,
            "graph_version": retrieved.graph_version,
            "index_version": retrieved.index_version,
            "source_versions": [{"source_id": source_id, "source_version": version}
                                for source_id, version in source_versions],
            "generation_performed": configuration.question is not None,
            "answer_count": len(answer_rows),
            "retrieval_trace_path": "retrieval-traces.jsonl",
            "answer_path": "answers.jsonl",
            "generation_trace_path": "traces.jsonl",
        }
        self._write(f"{base}/manifest.json", manifest)
        self.history_artifacts[f"{configuration.scenario_id}/{configuration.stage}"] = {
            "corpus_snapshot": configuration.corpus_snapshot,
            "index_version": retrieved.index_version,
            "graph_version": retrieved.graph_version,
            "source_versions": manifest["source_versions"],
            "manifest": f"{base}/manifest.json",
            "source_records": f"{base}/source-records.jsonl",
            "import_report": f"{base}/import-report.json",
            "graph_nodes": f"{base}/graph/nodes.jsonl",
            "graph_edges": f"{base}/graph/edges.jsonl",
            "answers": f"{base}/answers.jsonl",
            "generation_traces": f"{base}/traces.jsonl",
            "retrieval_traces": f"{base}/retrieval-traces.jsonl",
        }

    def record_cross_scope_isolation(
        self, engine: Engine, graph: FixtureGraphReport, embedding_client: EmbeddingClient,
    ) -> None:
        """Query each isolation scope again after both graphs coexist in the store."""
        isolation = [run for run in graph.runs if run.configuration.scenario_id.startswith("isolation-")]
        if len(isolation) != 2 or len({run.configuration.scope for run in isolation}) != 2:
            raise ValueError("cross-scope isolation requires two distinct completed fixture scopes")
        with engine.connect() as connection:
            for run in isolation:
                scope = run.configuration.scope
                count = connection.execute(text("""SELECT count(*) FROM source_head
                    WHERE tenant_id=:tenant AND project_id=:project"""), {
                    "tenant": scope.tenant_id, "project": scope.project_id,
                }).scalar_one()
                if count == 0:
                    raise ValueError("both isolation scopes must exist before cross-scope retrieval")

        for run in isolation:
            configuration = run.configuration
            base = f"history/{configuration.scenario_id}/{configuration.stage}"
            relative = f"{base}/cross-scope-retrieval-traces.jsonl"
            run_id = str(uuid4())
            question = configuration.question or "What supplied fixture evidence is current?"
            try:
                prepared = embed_query(embedding_client, question)
                retrieved = retrieve(
                    engine, configuration.scope, question, "graph", embedding_client,
                    run.embeddings.index_version, self.config, prepared,
                )
                row = self._retrieval_row(configuration, retrieved, run_id,
                                          cross_scope_check=True)
            except Exception as error:
                row = {
                    "scenario_id": configuration.scenario_id, "stage": configuration.stage,
                    "run_id": run_id, "scope": {
                        "tenant_id": configuration.scope.tenant_id,
                        "project_id": configuration.scope.project_id,
                    },
                    "question": question, "mode": "graph", "cross_scope_check": True,
                    "status": "failed", "failure_kind": (
                        "schema" if isinstance(error, (ValueError, TypeError, AssertionError))
                        else "provider"
                    ), "error_type": type(error).__name__,
                    "passages": [], "node_ids": [], "edge_ids": [],
                }
            self._write_rows(relative, [row])
            manifest_path = self._path(f"{base}/manifest.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["cross_scope_retrieval_trace_path"] = "cross-scope-retrieval-traces.jsonl"
                self._write(f"{base}/manifest.json", manifest)
            key = f"{configuration.scenario_id}/{configuration.stage}"
            self.history_artifacts.setdefault(key, {})["cross_scope_retrieval_traces"] = relative

    def check_superseded_after_retrieval(self) -> bool:
        """Require the saved after-stage result to contain v2 and exclude v1."""
        path = self._path("history/source-correction/after/retrieval-traces.jsonl")
        if not path.is_file():
            return False
        traces = _rows(path)
        if len(traces) != 1 or traces[0].get("status") != "succeeded":
            return False
        versions = {
            (str(row.get("source_id")), str(row.get("source_version")),
             str(row.get("passage_id")))
            for row in traces[0].get("passages", [])
        }
        identity = "assay-attribute-note", "assay-attribute-note:p0001"
        return ((identity[0], "fixture-v2", identity[1]) in versions
                and (identity[0], "fixture-v1", identity[1]) not in versions)

    def finish(self, engine: Engine, graph: FixtureGraphReport) -> dict[str, object]:
        """Compute checks from saved records plus scoped database membership."""
        self._write_rows("answers.jsonl", self.answers)
        self._write_rows("traces.jsonl", self.traces)
        checks: dict[str, str] = {}
        observed: dict[str, object] = {
            "answer_count": len(self.answers), "generation_trace_count": len(self.traces),
            "reconnect_verified": graph.reconnect_verified,
            "reconnect_query_embedding_calls": len({run.configuration.scope for run in graph.runs}),
            "model_usage_note": "Reconnect verification embeds one query per fixture scope; post-build isolation adds one query per isolation scope. Neither generates an answer. Dollar cost is not estimated without configured provider rates.",
            "generation_usage": _summed_generation_usage(self.traces),
        }
        failures: list[str] = []

        def check(name: str, result: bool | None) -> None:
            status = "not_run" if result is None else ("passed" if result else "failed")
            checks[name] = status
            if status == "failed":
                failures.append(name)

        saved_answers = _rows(self._path("answers.jsonl"))
        saved_traces = _rows(self._path("traces.jsonl"))
        by_stage = {(str(row["scenario_id"]), str(row["stage"])): row for row in saved_answers}
        by_trace = {(str(row["scenario_id"]), str(row["stage"])): row for row in saved_traces}
        before = by_stage.get(("source-correction", "before"))
        after = by_stage.get(("source-correction", "after"))
        conflict = by_stage.get(("unresolved-conflict", "combined"))
        check("three_generation_records", len(saved_traces) == 3)

        def cited_versions(row: dict[str, Any] | None) -> set[str]:
            return {str(citation["source_version"]) for citation in row.get("citations", [])} if row else set()

        check("before_answer_grounded_in_v1", None if before is None else cited_versions(before) == {"fixture-v1"})
        check("after_answer_grounded_in_v2", None if after is None else cited_versions(after) == {"fixture-v2"})
        check("superseded_passage_absent_after", self.check_superseded_after_retrieval())
        check("conflict_both_claims_cited", None if conflict is None else {
            str(citation["source_id"]) for citation in conflict.get("citations", [])
        } == {"plan-k-note-a", "plan-k-note-b"})
        conflict_trace = by_trace.get(("unresolved-conflict", "combined"))
        check("conflict_both_sources_in_generation_evidence", None if conflict_trace is None else {
            str(passage["source_id"]) for passage in conflict_trace.get("selected_passages", [])
        } == {"plan-k-note-a", "plan-k-note-b"})
        check("conflict_unresolved_wording_heuristic", None if conflict is None else any(
            term in str(conflict.get("answer", "")).lower()
            for term in ("unresolved", "conflict", "contradict", "disagree")
        ))
        observed["conflict_wording_check_is_heuristic"] = True

        citations_valid = True
        for answer in saved_answers:
            base = f"history/{answer['scenario_id']}/{answer['stage']}"
            records = {(row["source_id"], row["source_version"], row["passage_id"]): row["text"]
                       for row in _rows(self._path(f"{base}/source-records.jsonl"))}
            generation_trace = by_trace.get((str(answer["scenario_id"]), str(answer["stage"])), {})
            sent_texts = {
                (row["source_id"], row["source_version"], row["passage_id"]): row["sent_text"]
                for row in generation_trace.get("selected_passages", [])
            }
            for citation in answer.get("citations", []):
                identity = (citation["source_id"], citation["source_version"], citation["passage_id"])
                local_text = records.get(identity)
                with engine.connect() as connection:
                    db_text = connection.execute(text("""SELECT text FROM source_passage
                        WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source
                          AND source_version=:version AND passage_id=:passage"""), {
                        "tenant": citation["scope"]["tenant_id"],
                        "project": citation["scope"]["project_id"],
                        "source": identity[0], "version": identity[1], "passage": identity[2],
                    }).scalar_one_or_none()
                citations_valid &= bool(local_text and db_text == local_text
                                        and citation["quote"] in local_text
                                        and citation["quote"] in sent_texts.get(identity, ""))
        check("versioned_citations_resolve_exactly", None if not saved_answers else citations_valid)

        with engine.connect() as connection:
            conflict_sources = set(connection.execute(text("""SELECT DISTINCT source_id FROM assertion
                WHERE tenant_id='fixture-tenant' AND project_id='unresolved-conflict'""")).scalars())
        check("conflict_both_claims_preserved", conflict_sources == {"plan-k-note-a", "plan-k-note-b"})

        isolation_ok = True
        isolation_count = 0
        cross_scope_count = 0
        cross_scope_ok = True
        isolation_runs = [run for run in graph.runs
                          if run.configuration.scenario_id.startswith("isolation-")]
        other_projects = {
            run.configuration.scope.project_id: next(
                other.configuration.scope.project_id for other in isolation_runs
                if other.configuration.scope != run.configuration.scope
            )
            for run in isolation_runs
        } if len(isolation_runs) == 2 else {}
        assertions = json.loads((self.package_root / "fixtures" / "isolation-cases.json").read_text(
            encoding="utf-8"))["assertions"]
        expected_text = {item["query_scope"]["project_id"]: item for item in assertions}
        for run in graph.runs:
            if not run.configuration.scenario_id.startswith("isolation-"):
                continue
            base = f"history/{run.configuration.scenario_id}/{run.configuration.stage}"
            for filename in ("retrieval-traces.jsonl", "cross-scope-retrieval-traces.jsonl"):
                path = self._path(f"{base}/{filename}")
                if not path.is_file():
                    if filename.startswith("cross-scope"):
                        cross_scope_ok = False
                    else:
                        isolation_ok = False
                    continue
                for retrieval in _rows(path):
                    is_cross_scope = filename.startswith("cross-scope")
                    isolation_ok &= retrieval.get("status") == "succeeded"
                    if is_cross_scope:
                        cross_scope_count += 1
                        cross_scope_ok &= retrieval.get("cross_scope_check") is True
                        cross_scope_ok &= retrieval.get("status") == "succeeded"
                        cross_scope_ok &= bool(retrieval.get("passages"))
                    isolation_count += 1
                    scope = retrieval["scope"]
                    own = {"tenant": scope["tenant_id"], "project": scope["project_id"]}
                    foreign_project = other_projects.get(own["project"])
                    markers = expected_text.get(own["project"], {})
                    with engine.connect() as connection:
                        for item in retrieval["passages"]:
                            isolation_ok &= (item["tenant_id"], item["project_id"]) == (
                                own["tenant"], own["project"])
                            own_text = connection.execute(text("""SELECT text FROM source_passage
                                WHERE tenant_id=:tenant AND project_id=:project AND source_id=:source
                                  AND source_version=:version AND passage_id=:passage"""), own | {
                                "source": item["source_id"], "version": item["source_version"],
                                "passage": item["passage_id"],
                            }).scalar_one_or_none()
                            isolation_ok &= own_text == item["text"]
                            if is_cross_scope:
                                cross_scope_ok &= markers.get("must_contain", "") in item["text"]
                                cross_scope_ok &= markers.get("must_not_contain", "") not in item["text"]
                        for column, ids in (("node_id", retrieval["node_ids"]),
                                            ("edge_id", retrieval["edge_ids"])):
                            table = "kg_node" if column == "node_id" else "kg_edge"
                            for logical_id in ids:
                                isolation_ok &= connection.execute(text(
                                    f"SELECT count(*) FROM {table} WHERE tenant_id=:tenant "
                                    f"AND project_id=:project AND {column}=:logical_id"
                                ), own | {"logical_id": logical_id}).scalar_one() == 1
                                if is_cross_scope and foreign_project is not None:
                                    cross_scope_ok &= connection.execute(text(
                                        f"SELECT count(*) FROM {table} WHERE tenant_id=:tenant "
                                        f"AND project_id=:project AND {column}=:logical_id"
                                    ), {"tenant": own["tenant"], "project": foreign_project,
                                        "logical_id": logical_id}).scalar_one() == 0
                    isolation_ok &= all(item["project_id"] == scope["project_id"]
                                        for item in retrieval["passages"])
        observed["isolation_query_count"] = isolation_count
        observed["cross_scope_isolation_query_count"] = cross_scope_count
        check("isolation_scope_membership", isolation_ok and isolation_count == 4)
        check("isolation_after_both_scopes", cross_scope_ok and cross_scope_count == 2)
        check("reconnect_verified", graph.reconnect_verified)
        for trace in saved_traces:
            if trace.get("status") != "succeeded":
                failures.append(f"generation_failed:{trace['scenario_id']}/{trace['stage']}:{trace.get('failure_kind', 'unknown')}")
        report: dict[str, object] = {
            "commands": [f"run-fixture-answers --output {self.root}"],
            "checks": checks, "observed_results": observed,
            "failures": failures, "history_artifacts": self.history_artifacts,
        }
        self._write("report.json", report)
        return report
