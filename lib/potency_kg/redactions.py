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


def restore_model_quote(substitution: RedactionSubstitution, quote: str, occurrence_index: int | None = None) -> OriginalSpan:
    if not quote:
        raise ValueError("supporting quote must be nonempty")
    occurrences = [match.start() for match in re.finditer(re.escape(quote), substitution.model_text)]
    if not occurrences:
        raise ValueError("supporting quote does not occur in model text")
    if len(occurrences) > 1 and occurrence_index is None:
        raise ValueError("repeated supporting quote requires occurrence_index")
    index = occurrence_index or 0
    if index < 0 or index >= len(occurrences):
        raise ValueError("occurrence_index is out of bounds")
    return restore_model_span(substitution, occurrences[index], occurrences[index] + len(quote))
