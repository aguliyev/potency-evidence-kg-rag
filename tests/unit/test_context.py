from __future__ import annotations

from dataclasses import replace

from potency_kg.context import build_context, merge_candidates
from potency_kg.retrieval import RetrievalCandidate
from potency_kg.retrieval_config import HybridRetrievalConfig


def candidate(name: str, text: str, *, score: float = 0.8, via: str = "vector",
              document: str | None = None, page: int | None = None) -> RetrievalCandidate:
    source = document or name
    return RetrievalCandidate(
        (source, "v1", name), source, source, "v1", name, text, "sha",
        {"page": page} if page is not None else {}, via, score, score,
        None, None, None, (), False,
        {"source_type": "guidance", "authority": "FDA", "title": "NEVER SENT"},
    )


def test_context_counts_unicode_headers_and_separator_exactly() -> None:
    first = candidate("p1", "é🙂", page=2)
    second = candidate("p2", "abc", page=3)
    result = build_context((first, second), 100, HybridRetrievalConfig())
    assert result.evidence_text == (
        "[S1] p1 | v1 | guidance | FDA | page 2 | p1\né🙂\n\n"
        "[S2] p2 | v1 | guidance | FDA | page 3 | p2\nabc"
    )
    assert result.chars_used == len(result.evidence_text) <= 100
    assert result.tag_to_key == {"S1": first.key, "S2": second.key}
    assert result.sent_text == {first.key: "é🙂", second.key: "abc"}
    assert "NEVER SENT" not in result.evidence_text


def test_context_truncates_verbatim_prefix_and_drops_later_passages() -> None:
    first = candidate("a", "x" * 400)
    later = candidate("b", "later")
    result = build_context((first, later), 340, HybridRetrievalConfig())
    assert result.sent_text[first.key] == "x" * 307
    assert result.truncated == {first.key: 307}
    assert result.dropped == (later.key,)
    assert result.chars_used == 340


def test_no_truncation_can_skip_large_passage_for_later_small_one() -> None:
    large = candidate("large", "x" * 400)
    small = candidate("small", "yes")
    result = build_context((large, small), 100, HybridRetrievalConfig(truncate_last_passage=False))
    assert result.tag_to_key == {"S1": small.key}
    assert result.dropped == (large.key,)


def test_document_order_changes_tags_but_not_selected_keys() -> None:
    a2 = candidate("a2", "A2", score=0.9, document="a", page=2)
    b1 = candidate("b1", "B1", score=0.8, document="b", page=1)
    a1 = candidate("a1", "A1", score=0.7, document="a", page=1)
    score = build_context((a2, b1, a1), 500, HybridRetrievalConfig())
    document = build_context((a2, b1, a1), 500, HybridRetrievalConfig(context_order="document"))
    assert list(score.tag_to_key.values()) == [a2.key, b1.key, a1.key]
    assert list(document.tag_to_key.values()) == [a1.key, a2.key, b1.key]


def test_document_order_does_not_change_budget_selection() -> None:
    high = candidate("high", "x" * 60, score=0.9, document="same", page=2)
    low = candidate("low", "y" * 60, score=0.1, document="same", page=1)
    score = build_context((high, low), 110, HybridRetrievalConfig())
    document = build_context((high, low), 110,
                             HybridRetrievalConfig(context_order="document"))
    assert score.tag_to_key == document.tag_to_key == {"S1": high.key}
    assert score.dropped == document.dropped == (low.key,)


def test_graph_share_is_reserved_and_both_uses_vector_share() -> None:
    both = candidate("both", "v" * 70, via="both", score=0.9)
    vector = candidate("vector", "v" * 70, score=0.8)
    graph = candidate("graph", "g" * 70, via="graph", score=0.1)
    merged = merge_candidates((both, vector, graph), "graph", 240,
                              HybridRetrievalConfig(graph_share=0.5, min_truncated_chars=300))
    assert [item.candidate.key for item in merged] == [both.key, graph.key]
    assert [item.candidate.selected_via for item in merged] == ["both", "graph"]
    context = build_context(merged, 240, HybridRetrievalConfig())
    assert list(context.sent_text) == [both.key, graph.key]
    assert context.chars_used <= 240


def test_unused_graph_share_returns_to_unused_vector_hits() -> None:
    a = candidate("a", "a" * 60, score=0.9)
    b = candidate("b", "b" * 60, score=0.8)
    merged = merge_candidates((a, b), "graph", 210,
                              HybridRetrievalConfig(graph_share=0.5))
    assert [item.candidate.key for item in merged] == [a.key, b.key]


def test_graph_share_does_not_claim_unused_vector_allocation_when_return_disabled() -> None:
    vector = candidate("vector", "v" * 20, score=0.9)
    graph_one = candidate("g1", "g" * 50, via="graph", score=0.8)
    graph_two = candidate("g2", "h" * 50, via="graph", score=0.7)
    strict = merge_candidates((vector, graph_one, graph_two), "graph", 250,
                              HybridRetrievalConfig(graph_share=0.5,
                                                    return_unused_graph_share=False,
                                                    truncate_last_passage=False))
    returned = merge_candidates((vector, graph_one, graph_two), "graph", 250,
                                HybridRetrievalConfig(graph_share=0.5,
                                                      return_unused_graph_share=True,
                                                      truncate_last_passage=False))
    assert [item.candidate.key for item in strict] == [vector.key, graph_one.key]
    assert [item.candidate.key for item in returned] == [vector.key, graph_one.key, graph_two.key]


def test_score_context_reorders_merged_passages_without_changing_selection() -> None:
    vector = candidate("vector", "v", score=0.5)
    graph = candidate("graph", "g", via="graph", score=0.9)
    merged = merge_candidates((vector, graph), "graph", 200,
                              HybridRetrievalConfig(graph_share=0.5))
    assert [item.candidate.key for item in merged] == [vector.key, graph.key]
    context = build_context(merged, 200, HybridRetrievalConfig(context_order="score"))
    assert context.tag_to_key == {"S1": graph.key, "S2": vector.key}
    assert list(context.sent_text) == [graph.key, vector.key]


def test_vector_floor_dedup_and_document_cap() -> None:
    high = candidate("high", "a", score=0.9, document="same")
    duplicate = replace(high, score=0.8)
    other = candidate("other", "b", score=0.7, document="same")
    low = candidate("low", "c", score=0.2)
    selected = merge_candidates((duplicate, low, other, high), "vector", 500,
                                HybridRetrievalConfig(vector_min_cosine=0.5,
                                                      max_passages_per_document=1))
    assert [item.candidate.key for item in selected] == [high.key]
