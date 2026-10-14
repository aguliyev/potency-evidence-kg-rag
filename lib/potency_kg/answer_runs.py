"""Persist a question-by-mode answer batch with an auditable trace for every attempt."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from sqlalchemy.engine import Engine

from .answering import AnswerGenerationError, answer_question, expand_citations, prompt_metadata, validate_citations
from .context import ContextResult, build_context, merge_candidates
from .contracts import Scope
from .embedding_client import EmbeddingClient
from .embeddings import embed_query
from .llm_client import StructuredExtractionClient
from .output_safety import guard_output_path
from .retrieval import RetrievalCandidate, RetrievalResult, retrieve
from .retrieval_config import HybridRetrievalConfig


def _reference(candidate: RetrievalCandidate) -> dict[str, str]:
    return {"source_id": candidate.source_id, "source_version": candidate.source_version,
            "passage_id": candidate.passage_id}


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]], root: Path) -> None:
    target = guard_output_path(path, package_root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")


def _trace_base(
    question_id: str, question: str, mode: str, scope: Scope, corpus_snapshot: str,
    index_version: str, max_chars: int, config: HybridRetrievalConfig,
    generation_client: StructuredExtractionClient, commit: str,
) -> dict[str, object]:
    prompt_version, prompt_sha256 = prompt_metadata()
    return {
        "trace_id": str(uuid4()), "question_id": question_id, "question": question, "mode": mode,
        "status": "failed", "scope": {"tenant_id": scope.tenant_id, "project_id": scope.project_id},
        "corpus_snapshot": corpus_snapshot, "index_version": index_version,
        "graph_version": None, "retriever_version": None, "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256, "model": generation_client.identity.model,
        "git_commit": commit, "config": config.as_dict(), "config_hash": config.config_hash(),
        "max_evidence_chars": max_chars, "seed_node_ids": [], "node_ids": [],
        "traversed_edge_ids": [], "hops": [], "traversal_limits": {
            "max_hops": config.max_hops, "max_seeds": config.max_seeds,
            "neighbors_per_entity": config.neighbors_per_entity,
            "frontier_cap_per_hop": config.frontier_cap_per_hop,
            "hub_degree_cap": config.hub_degree_cap, "candidate_cap": config.candidate_cap,
        },
        "candidate_passage_ids": [], "selected_passage_ids": [], "document_ids": [],
        "selected_via": [], "selected_passages": [], "truncated": [], "dropped": [],
        "evidence_text": "", "usage": {}, "attempts": [], "elapsed_ms": 0,
        "citation_check": {"status": "not_run", "failures": []},
    }


def _apply_retrieval(trace: dict[str, object], retrieved: RetrievalResult) -> None:
    trace.update({
        "graph_version": retrieved.graph_version, "retriever_version": retrieved.retriever_version,
        "embedding_version": retrieved.embedding_version, "query_hash": retrieved.query_hash,
        "seed_node_ids": list(retrieved.seed_node_ids), "node_ids": list(retrieved.node_ids),
        "traversed_edge_ids": list(retrieved.edge_ids),
        "candidate_passage_ids": [item.passage_id for item in retrieved.candidates],
        "candidate_passages": [_reference(item) | {"selected_via": item.selected_via,
                                    "score": item.score, "cosine": item.cosine,
                                    "hop": item.hop, "supporting_path_ids": list(item.supporting_path_ids)}
                               for item in retrieved.candidates],
        "retrieval_diagnostics": retrieved.diagnostics,
    })


def _apply_context(trace: dict[str, object], context: ContextResult,
                   candidates: Sequence[RetrievalCandidate]) -> None:
    selected = [item.candidate for item in context.passages]
    trace.update({
        "evidence_text": context.evidence_text,
        "selected_passage_ids": [item.passage_id for item in selected],
        "document_ids": list(dict.fromkeys(item.document_id for item in selected)),
        "selected_via": [_reference(item) | {"selected_via": item.selected_via} for item in selected],
        "selected_passages": [_reference(item.candidate) | {
            "selected_via": item.candidate.selected_via, "sent_text": item.sent_text,
            "truncated": item.truncated, "hop": item.candidate.hop,
            "via_node_id": item.candidate.via_node_id, "via_edge_id": item.candidate.via_edge_id,
            "supporting_path_ids": list(item.candidate.supporting_path_ids),
        } for item in context.passages],
        "hops": [_reference(item) | {"hop": item.hop} for item in selected],
        "truncated": [_reference(item.candidate) | {"chars_kept": len(item.sent_text)}
                      for item in context.passages if item.truncated],
        "dropped": [_reference(item) for key in context.dropped for item in candidates if item.key == key]
                   + [_reference(item) for item in candidates if item.key not in {entry.key for entry in selected}
                      and item.key not in context.dropped],
    })


def answer_from_retrieval(
    question_id: str, question: str, mode: str, retrieved: RetrievalResult,
    scope: Scope, corpus_snapshot: str, generation_client: StructuredExtractionClient,
    max_evidence_chars: int, config: HybridRetrievalConfig, *, git_commit: str,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Build and validate one answer from an existing retrieval; fixture stages reuse it."""
    trace = _trace_base(question_id, question, mode, scope, corpus_snapshot,
                        retrieved.index_version, max_evidence_chars, config,
                        generation_client, git_commit)
    _apply_retrieval(trace, retrieved)
    try:
        if retrieved.config.config_hash() != config.config_hash():
            raise ValueError("retrieval and answering configurations differ")
        merged = merge_candidates(retrieved.candidates, mode, max_evidence_chars, config)
        context = build_context(merged, max_evidence_chars, config)
        _apply_context(trace, context, retrieved.candidates)
        answer, usage, elapsed_ms, attempts = answer_question(
            generation_client, question, context, config=config)
        citations = expand_citations(answer, context, scope,
                                     allow_normalized=config.allow_normalized_quote_match)
        trace.update({"status": "succeeded", "usage": usage, "elapsed_ms": elapsed_ms,
                      "attempts": list(attempts), "citation_check": validate_citations(
                          answer, context, allow_normalized=config.allow_normalized_quote_match)})
        record: dict[str, object] = {
            "question_id": question_id, "mode": mode, "answer": answer.answer,
            "citations": citations, "evidence_gaps": answer.evidence_gaps,
            "trace_id": trace["trace_id"], "scope": trace["scope"],
            "corpus_snapshot": corpus_snapshot,
        }
        return record, trace
    except AnswerGenerationError as error:
        trace.update({"failure_kind": error.failure_kind, "error_type": error.error_type,
                      "usage": error.usage, "elapsed_ms": error.elapsed_ms,
                      "attempts": list(error.attempts), "citation_check": error.citation_check})
    except Exception as error:
        trace.update({"failure_kind": "schema", "error_type": type(error).__name__,
                      "failed_stage": "context_or_record_validation"})
    return None, trace


def run_answer_batch(
    engine: Engine, scope: Scope, descriptor: Mapping[str, object],
    questions: Sequence[Mapping[str, object]], output: Path,
    embedding_client: EmbeddingClient, generation_client: StructuredExtractionClient,
    index_version: str, config: HybridRetrievalConfig, *, package_root: Path,
) -> dict[str, object]:
    """Run both modes using one embedding per question; isolate every failed stage."""
    root = package_root.resolve()
    destination = guard_output_path(output, package_root=root)
    output_paths = {
        "main_answers": destination / "answers.jsonl",
        "main_traces": destination / "traces.jsonl",
        "extra_answers": destination / "extra-mode/answers.jsonl",
        "extra_traces": destination / "extra-mode/traces.jsonl",
    }
    for path in output_paths.values():
        guard_output_path(path, package_root=root)
    snapshot = descriptor.get("corpus_snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("descriptor requires corpus_snapshot")
    if not isinstance(index_version, str) or not index_version:
        raise ValueError("answer run requires a bound index_version")
    commit = _git_commit(root)
    rows: dict[str, list[dict[str, object]]] = {
        "main_answers": [], "main_traces": [], "extra_answers": [], "extra_traces": [],
    }
    seen_ids: set[str] = set()
    for item in questions:
        question_id, question = item.get("id"), item.get("question")
        if (not isinstance(question_id, str) or not question_id or question_id in seen_ids
                or not isinstance(question, str) or not question.strip()):
            raise ValueError("questions require unique nonempty id and question")
        seen_ids.add(question_id)
        cap = item.get("max_evidence_chars_per_mode", config.default_budget_chars)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            raise ValueError("question evidence cap must be a nonnegative integer")
        prepared_embedding = None
        embedding_error: Exception | None = None
        try:
            prepared_embedding = embed_query(embedding_client, question)
        except Exception as error:
            embedding_error = error
        for mode in ("vector", "graph"):
            category = "main" if mode == "graph" or item.get("compare_modes") is True else "extra"
            try:
                if embedding_error is not None:
                    raise embedding_error
                assert prepared_embedding is not None
                retrieved = retrieve(engine, scope, question, mode, embedding_client,
                                     index_version, config, prepared_embedding)
                answer_row, trace = answer_from_retrieval(
                    question_id, question, mode, retrieved, scope, snapshot, generation_client,
                    cap, config, git_commit=commit)
            except Exception as error:
                trace = _trace_base(question_id, question, mode, scope, snapshot, index_version,
                                    cap, config, generation_client, commit)
                trace.update({"failure_kind": "provider", "error_type": type(error).__name__,
                              "failed_stage": "embedding" if embedding_error is not None else "retrieval"})
                answer_row = None
            if answer_row is not None:
                rows[f"{category}_answers"].append(answer_row)
            rows[f"{category}_traces"].append(trace)
    for name, path in output_paths.items():
        _write_jsonl(path, rows[name], root)
    return {"status": "completed", "succeeded": len(rows["main_answers"]) + len(rows["extra_answers"]),
            "failed": len(rows["main_traces"]) + len(rows["extra_traces"]) -
                      len(rows["main_answers"]) - len(rows["extra_answers"]),
            "main_traces": len(rows["main_traces"]), "extra_traces": len(rows["extra_traces"]),
            "output": str(destination)}
