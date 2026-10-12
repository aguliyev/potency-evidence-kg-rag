from __future__ import annotations

import pytest

from potency_kg.alias_discovery import discover_alias_patterns
from potency_kg.contracts import ScopedPassageIdentity


PASSAGE = ScopedPassageIdentity("tenant", "project", "source", "v1", "p1")


@pytest.mark.parametrize(
    ("body", "alias_value", "canonical_name", "exact_quote", "start", "end"),
    [
        (
            "Exagamglogene autotemcel (exa-cel) was evaluated.",
            "exa-cel",
            "Exagamglogene autotemcel",
            "Exagamglogene autotemcel (exa-cel)",
            0,
            34,
        ),
        (
            "The study evaluated exa-cel (Exagamglogene autotemcel).",
            "exa-cel",
            "Exagamglogene autotemcel",
            "exa-cel (Exagamglogene autotemcel)",
            20,
            54,
        ),
        (
            "Exagamglogene autotemcel, also known as exa-cel, was evaluated.",
            "exa-cel",
            "Exagamglogene autotemcel",
            "Exagamglogene autotemcel, also known as exa-cel",
            0,
            47,
        ),
        (
            "Exagamglogene autotemcel, abbreviated as exa-cel, was evaluated.",
            "exa-cel",
            "Exagamglogene autotemcel",
            "Exagamglogene autotemcel, abbreviated as exa-cel",
            0,
            48,
        ),
        (
            "Exagamglogene autotemcel, formerly called exa-cel, was evaluated.",
            "exa-cel",
            "Exagamglogene autotemcel",
            "Exagamglogene autotemcel, formerly called exa-cel",
            0,
            49,
        ),
    ],
)
def test_document_alias_patterns_return_exactly_grounded_conservative_proposals(
    body: str,
    alias_value: str,
    canonical_name: str,
    exact_quote: str,
    start: int,
    end: int,
) -> None:
    """Reversing a mapping, losing source identity, or approximating offsets must break this test."""
    proposals = discover_alias_patterns(PASSAGE, body)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert (
        proposal.alias_value,
        proposal.canonical_name,
        proposal.source,
        proposal.exact_quote,
        proposal.original_start,
        proposal.original_end,
    ) == (alias_value, canonical_name, PASSAGE, exact_quote, start, end)
    assert body[proposal.original_start:proposal.original_end] == proposal.exact_quote
    assert proposal.entity_type is None
    assert proposal.canonical_key is None
    assert proposal.source_type == "deterministic_rule"
    assert (proposal.detector_name, proposal.detector_version) == (
        "document-alias-patterns",
        "document-alias-patterns-v1",
    )


@pytest.mark.parametrize(
    "body",
    [
        "Page 12 (Document 4)",
        "1234 (5678)",
        "Exagamglogene autotemcel (exa-cel was evaluated.",
        "CASGEVY (casgevy) was evaluated.",
        "Exagamglogene autotemcel (E) was evaluated.",
        "A very long investigational exagamglogene autotemcel treatment designation (exa) was evaluated.",
        "Exagamglogene autotemcel (XYZ) was evaluated.",
    ],
)
def test_document_alias_patterns_reject_non_alias_or_ungrounded_candidates(body: str) -> None:
    """Accepting page furniture, degenerate forms, bad ratios, or unaligned acronyms creates review noise."""
    assert discover_alias_patterns(PASSAGE, body) == ()


def test_document_alias_patterns_do_not_absorb_preceding_prose_into_the_canonical_name() -> None:
    """A greedy match that stores sentence framing as identity makes review evidence misleading."""
    body = "The study evaluated Exagamglogene autotemcel (exa-cel)."

    proposal = discover_alias_patterns(PASSAGE, body)[0]

    assert (proposal.canonical_name, proposal.exact_quote, proposal.original_start, proposal.original_end) == (
        "Exagamglogene autotemcel",
        "Exagamglogene autotemcel (exa-cel)",
        20,
        54,
    )
