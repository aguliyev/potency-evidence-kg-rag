"""Lossless passage-scoped substitutions for supplied FDA redaction markers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import SourceRef

_REDACTION = re.compile(r"\(\s*b\s*\)\s*\(\s*4\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class RedactionSpan:
    original_start: int
    original_end: int
    model_start: int
    model_end: int


@dataclass(frozen=True)
class RedactionSubstitution:
    source_ref: SourceRef
    original_text: str
    model_text: str
    spans: tuple[RedactionSpan, ...]


@dataclass(frozen=True)
class OriginalSpan:
    start: int
    end: int
    text: str


def substitute_redactions(source_ref: SourceRef, text: str) -> RedactionSubstitution:
    pieces: list[str] = []
    spans: list[RedactionSpan] = []
    original_cursor = model_cursor = 0
    for number, match in enumerate(_REDACTION.finditer(text), start=1):
        untouched = text[original_cursor:match.start()]
        pieces.append(untouched)
        model_cursor += len(untouched)
        token = f"[REDACTED:{source_ref.passage_id}:{number}]"
        pieces.append(token)
        spans.append(RedactionSpan(match.start(), match.end(), model_cursor, model_cursor + len(token)))
        model_cursor += len(token)
        original_cursor = match.end()
    pieces.append(text[original_cursor:])
    return RedactionSubstitution(source_ref, text, "".join(pieces), tuple(spans))


def restore_model_span(substitution: RedactionSubstitution, start: int, end: int) -> OriginalSpan:
    if start < 0 or end < start or end > len(substitution.model_text):
        raise ValueError("model span is out of bounds")
    for span in substitution.spans:
        if start < span.model_end and end > span.model_start:
            raise ValueError("requested span crosses an unmappable redaction token")
    delta = 0
    for span in substitution.spans:
        if span.model_end <= start:
            delta += (span.original_end - span.original_start) - (span.model_end - span.model_start)
    original_start = start + delta
    original_end = original_start + (end - start)
    return OriginalSpan(original_start, original_end, substitution.original_text[original_start:original_end])


def locate_model_quote(
    substitution: RedactionSubstitution, quote: str, occurrence_index: int | None = None, *, lenient: bool = False
) -> tuple[int, int]:
    """Model-text span of ``quote``. Strict: exact text, repeated quotes need an index.

    lenient (LLM output): falls back to a whitespace-insensitive match when the exact text is absent,
    and picks the first occurrence when a repeated quote has no index.
    """
    if not quote:
        raise ValueError("supporting quote must be nonempty")
    text = substitution.model_text
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(quote), text)]
    if not spans and lenient and quote.split():
        pattern = r"\s+".join(re.escape(word) for word in quote.split())
        spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    if not spans:
        raise ValueError("supporting quote does not occur in model text")
    if len(spans) > 1 and occurrence_index is None and not lenient:
        raise ValueError("repeated supporting quote requires occurrence_index")
    index = occurrence_index or 0
    if index < 0 or index >= len(spans):
        if lenient and occurrence_index is not None:
            index = 0  # model miscounted occurrences; the quote itself is verbatim
        else:
            raise ValueError("occurrence_index is out of bounds")
    return spans[index]


def restore_model_quote(
    substitution: RedactionSubstitution, quote: str, occurrence_index: int | None = None, *, lenient: bool = False
) -> OriginalSpan:
    start, end = locate_model_quote(substitution, quote, occurrence_index, lenient=lenient)
    return restore_model_span(substitution, start, end)
