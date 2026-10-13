from __future__ import annotations

import pytest

from potency_kg.contracts import SourceRef
from potency_kg.redactions import restore_model_quote, restore_model_span, substitute_redactions


def test_substitution_handles_spacing_and_restores_exact_nonredacted_offsets() -> None:
    """Losing spacing variants or offset mapping must break this test."""
    original = "Before (b)(4) middle (b) ( 4 ) after"
    substitution = substitute_redactions(SourceRef("t", "p", "s", "v", "x"), original)
    assert substitution.model_text.count("[REDACTED:") == 2
    start = substitution.model_text.index("middle")
    restored = restore_model_span(substitution, start, start + len("middle"))
    assert (restored.start, restored.end, restored.text) == (14, 20, "middle")


def test_redaction_crossing_and_ambiguous_quotes_are_rejected() -> None:
    """Allowing reconstructed content or ambiguous support quotes must break this test."""
    substitution = substitute_redactions(SourceRef("t", "p", "s", "v", "x"), "same (b)(4) same")
    marker = substitution.model_text.index("[REDACTED:")
    with pytest.raises(ValueError):
        restore_model_span(substitution, marker, marker + 1)
    with pytest.raises(ValueError):
        restore_model_quote(substitution, "same")
    assert restore_model_quote(substitution, "same", occurrence_index=1).text == "same"


def test_lenient_quote_matching_for_llm_output() -> None:
    from potency_kg.redactions import locate_model_quote

    sub = substitute_redactions(SourceRef("t", "p", "s", "v", "x"), "alpha beta\ngamma. alpha beta\ngamma.")
    with pytest.raises(ValueError):
        locate_model_quote(sub, "alpha beta gamma.")  # strict: whitespace differs
    assert locate_model_quote(sub, "alpha beta gamma.", lenient=True) == (0, 17)  # first occurrence
    assert locate_model_quote(sub, "alpha beta gamma.", 1, lenient=True) == (18, 35)
    assert locate_model_quote(sub, "alpha beta\ngamma.", 9, lenient=True) == (0, 17)  # bad index tolerated
