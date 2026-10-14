"""Read-only reconstruction of citation checks from saved answer evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .answering import AnswerModel, validate_citations
from .context import ContextResult, PassageKey


def validate_saved_citations(
    answer: Mapping[str, object], trace: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Rebuild S tags from the exact saved evidence blocks, then use production validation."""
    evidence = trace.get("evidence_text", "")
    selected = trace.get("selected_passages", [])
    if not isinstance(evidence, str) or not isinstance(selected, list):
        return {"status": "failed", "failures": [{"reason": "invalid_saved_trace"}], "matches": []}, []
    blocks = re.split(r"\n\n(?=\[S\d+\] )", evidence) if evidence else []
    if len(blocks) != len(selected):
        return {"status": "failed", "failures": [{"reason": "evidence_block_mismatch"}], "matches": []}, []
    tags: dict[str, PassageKey] = {}
    sent: dict[PassageKey, str] = {}
    for index, (block, passage) in enumerate(zip(blocks, selected, strict=True), 1):
        if not isinstance(passage, dict):
            return {"status": "failed", "failures": [{"reason": "invalid_selected_passage"}], "matches": []}, []
        key = tuple(str(passage.get(name, "")) for name in ("source_id", "source_version", "passage_id"))
        header, separator, body = block.partition("\n")
        if (not separator or not header.startswith(f"[S{index}] ")
                or any(value not in header for value in key)
                or body != passage.get("sent_text")):
            return {"status": "failed", "failures": [{"reason": "evidence_block_mismatch"}], "matches": []}, []
        tags[f"S{index}"] = key  # type: ignore[assignment]
        sent[key] = body  # type: ignore[index]
    citations = answer.get("citations", [])
    if not isinstance(citations, list):
        return {"status": "failed", "failures": [{"reason": "invalid_saved_answer"}], "matches": []}, []
    displayed: list[dict[str, str]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            return {"status": "failed", "failures": [{"reason": "invalid_saved_answer"}], "matches": []}, []
        identity = tuple(str(citation.get(name, "")) for name in ("source_id", "source_version", "passage_id"))
        tag = next((tag for tag, key in tags.items() if key == identity), "unknown")
        displayed.append({"tag": tag, "quote": str(citation.get("quote", "")),
                          "source_id": identity[0], "source_version": identity[1],
                          "passage_id": identity[2]})
    model = AnswerModel(answer=str(answer.get("answer", "")),
                        citations=[{"tag": row["tag"], "quote": row["quote"]} for row in displayed],
                        evidence_gaps=[])
    context = ContextResult(evidence, tags, sent, len(evidence), {}, (), ())
    check = validate_citations(model, context)
    for row in displayed:
        match = next((item["match"] for item in check["matches"]
                      if item["tag"] == row["tag"] and item["quote"] == row["quote"]), "none")
        row["match"] = match
        row["normalized_match"] = str(match == "normalized")
    return check, displayed
