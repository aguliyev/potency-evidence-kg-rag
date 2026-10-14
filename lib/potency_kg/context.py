"""Pure, character-accurate selection and rendering of retrieved source evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor
from typing import Literal, Sequence

from .retrieval import RetrievalCandidate
from .retrieval_config import HybridRetrievalConfig


PassageKey = tuple[str, str, str]


@dataclass(frozen=True)
class SelectedPassage:
    candidate: RetrievalCandidate
    sent_text: str
    truncated: bool = False


@dataclass(frozen=True)
class ContextResult:
    evidence_text: str
    tag_to_key: dict[str, PassageKey]
    sent_text: dict[PassageKey, str]
    chars_used: int
    truncated: dict[PassageKey, int]
    dropped: tuple[PassageKey, ...]
    passages: tuple[SelectedPassage, ...]


class EvidenceBlockRenderer:
    """One format and cost calculation for allocation and final model context."""

    separator = "\n\n"

    @staticmethod
    def header(candidate: RetrievalCandidate, tag: str) -> str:
        fields = [candidate.source_id, candidate.source_version]
        for name in ("source_type", "authority"):
            value = candidate.source_metadata.get(name)
            if value is not None and str(value):
                fields.append(str(value))
        location = []
        for name in ("page", "section"):
            value = candidate.locator.get(name)
            if value is not None and str(value):
                location.append(f"{name} {value}")
        if location:
            fields.append(" / ".join(location))
        fields.append(candidate.passage_id)
        return f"[{tag}] " + " | ".join(fields) + "\n"

    def block(self, candidate: RetrievalCandidate, tag: str, sent_text: str) -> str:
        return self.header(candidate, tag) + sent_text

    def cost(self, candidate: RetrievalCandidate, tag: str, sent_text: str, has_previous: bool) -> int:
        return len(self.separator if has_previous else "") + len(self.block(candidate, tag, sent_text))

    def fit(
        self, candidate: RetrievalCandidate, tag: str, remaining: int,
        has_previous: bool, config: HybridRetrievalConfig,
    ) -> SelectedPassage | None:
        overhead = self.cost(candidate, tag, "", has_previous)
        available = remaining - overhead
        if available >= len(candidate.text):
            return SelectedPassage(candidate, candidate.text)
        if config.truncate_last_passage and available >= config.min_truncated_chars:
            return SelectedPassage(candidate, candidate.text[:available], True)
        return None


def _budget(max_chars: int) -> None:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")


def _dedup(candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
    best: dict[PassageKey, RetrievalCandidate] = {}
    for candidate in candidates:
        prior = best.get(candidate.key)
        if prior is None:
            best[candidate.key] = candidate
            continue
        via = "both" if "both" in (prior.selected_via, candidate.selected_via) or {
            prior.selected_via, candidate.selected_via
        } == {"vector", "graph"} else prior.selected_via
        winner = max((prior, candidate), key=lambda item: (
            item.selected_via in ("vector", "both"),
            item.cosine if item.selected_via in ("vector", "both") else item.score,
        ))
        if winner.selected_via != via:
            winner = replace(winner, selected_via=via)
        best[candidate.key] = winner
    return list(best.values())


def merge_candidates(
    candidates: Sequence[RetrievalCandidate], mode: Literal["vector", "graph"],
    max_chars: int, config: HybridRetrievalConfig,
) -> tuple[SelectedPassage, ...]:
    """Allocate rendered evidence under one total budget and a reserved graph share."""
    _budget(max_chars)
    if mode not in ("vector", "graph"):
        raise ValueError("mode must be vector or graph")
    unique = _dedup(candidates)
    vectors = sorted((item for item in unique if item.selected_via in ("vector", "both")
                      and (config.vector_min_cosine is None or item.cosine >= config.vector_min_cosine)),
                     key=lambda item: (-item.cosine, item.key))
    graphs = sorted((item for item in unique if item.selected_via == "graph"
                     and (config.graph_min_cosine is None or item.cosine >= config.graph_min_cosine)),
                    key=lambda item: (-item.score, item.key))
    selected: list[SelectedPassage] = []
    keys: set[PassageKey] = set()
    per_document: dict[str, int] = {}
    renderer = EvidenceBlockRenderer()
    used = 0

    def fill(items: Sequence[RetrievalCandidate], limit: int) -> None:
        nonlocal used
        for item in items:
            if item.key in keys:
                continue
            if (config.max_passages_per_document is not None and
                    per_document.get(item.document_id, 0) >= config.max_passages_per_document):
                continue
            tag = f"S{len(selected) + 1}"
            fit = renderer.fit(item, tag, limit - used, bool(selected), config)
            if fit is None:
                continue
            selected.append(fit)
            keys.add(item.key)
            per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
            used += renderer.cost(item, tag, fit.sent_text, len(selected) > 1)
            if fit.truncated:
                break

    if mode == "vector":
        fill(vectors, max_chars)
    else:
        vector_limit = floor((1 - config.graph_share) * max_chars)
        fill(vectors, vector_limit)
        graph_limit = (max_chars if config.return_unused_graph_share
                       else used + max_chars - vector_limit)
        fill(graphs, graph_limit)
        if config.return_unused_graph_share:
            fill(vectors, max_chars)
    return tuple(selected)


def build_context(
    candidates: Sequence[RetrievalCandidate | SelectedPassage],
    max_chars: int, config: HybridRetrievalConfig,
) -> ContextResult:
    """Render selected passages with final-order tags and exact sent-text tracking."""
    _budget(max_chars)
    ranked = [item if isinstance(item, SelectedPassage) else SelectedPassage(item, item.text)
              for item in candidates]
    if not all(isinstance(item, SelectedPassage) for item in candidates):
        ranked.sort(key=lambda item: (-item.candidate.score, item.candidate.key))
    renderer = EvidenceBlockRenderer()
    accepted: list[SelectedPassage] = []
    dropped: list[PassageKey] = []
    used = 0
    for index, item in enumerate(ranked):
        candidate = item.candidate
        if candidate.key in {entry.candidate.key for entry in accepted}:
            dropped.append(candidate.key)
            continue
        tag = f"S{len(accepted) + 1}"
        remaining = max_chars - used
        # A merge allocation may already have cut the passage; preserve that prefix.
        budget_candidate = replace(candidate, text=item.sent_text)
        fit = renderer.fit(budget_candidate, tag, remaining, bool(accepted), config)
        if fit is None:
            dropped.append(candidate.key)
            continue
        accepted.append(SelectedPassage(candidate, fit.sent_text, item.truncated or fit.truncated))
        used += renderer.cost(candidate, tag, fit.sent_text, len(accepted) > 1)
        if fit.truncated:
            dropped.extend(later.candidate.key for later in ranked[index + 1:])
            break
    if config.context_order == "document":
        document_rank = dict.fromkeys(item.candidate.document_id for item in accepted)
        document_rank = {document: rank for rank, document in enumerate(document_rank)}

        def page(item: SelectedPassage) -> tuple[int, str]:
            value = item.candidate.locator.get("page")
            try:
                return int(value), str(item.candidate.locator.get("section", ""))
            except (TypeError, ValueError):
                return 10**9, str(item.candidate.locator.get("section", ""))

        accepted.sort(key=lambda item: (document_rank[item.candidate.document_id],
                                        page(item), item.candidate.key))
    else:
        accepted.sort(key=lambda item: (-item.candidate.score, item.candidate.key))
    blocks = [renderer.block(item.candidate, f"S{index}", item.sent_text)
              for index, item in enumerate(accepted, 1)]
    evidence_text = renderer.separator.join(blocks)
    tags = {f"S{index}": item.candidate.key for index, item in enumerate(accepted, 1)}
    sent = {item.candidate.key: item.sent_text for item in accepted}
    truncated = {item.candidate.key: len(item.sent_text) for item in accepted if item.truncated}
    return ContextResult(evidence_text, tags, sent, len(evidence_text), truncated,
                         tuple(dropped), tuple(accepted))
