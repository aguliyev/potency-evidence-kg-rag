"""Structured answering and local validation against evidence actually sent."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from time import monotonic
from pydantic import BaseModel, ConfigDict, ValidationError

from .context import ContextResult
from .contracts import Scope
from .llm_client import GenerationSchemaError, GenerationUnselectedTypeError, StructuredExtractionClient
from .retrieval_config import HybridRetrievalConfig


PROMPT_VERSION = "answer-prompt-v1"
_PROMPT_PATH = Path(__file__).with_name("prompt_resources") / "answer_system.md"


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str
    quote: str


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[AnswerCitation]
    evidence_gaps: list[str]


class AnswerGenerationError(RuntimeError):
    """Safe run failure metadata; never retains a provider exception or its text."""

    def __init__(
        self, failure_kind: str, error_type: str, *, attempts: tuple[dict[str, object], ...],
        usage: dict[str, int | float | None], elapsed_ms: int,
        citation_check: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"answer generation failed: {failure_kind} ({error_type})")
        self.failure_kind = failure_kind
        self.error_type = error_type
        self.attempts = attempts
        self.usage = usage
        self.elapsed_ms = elapsed_ms
        self.citation_check = citation_check or {"status": "not_run", "failures": []}


class CitationValidationError(AnswerGenerationError):
    def __init__(
        self, *, attempts: tuple[dict[str, object], ...],
        usage: dict[str, int | float | None], elapsed_ms: int,
        citation_check: dict[str, object],
    ) -> None:
        super().__init__("citation_validation", "CitationValidationError", attempts=attempts,
                         usage=usage, elapsed_ms=elapsed_ms, citation_check=citation_check)


def prompt_metadata() -> tuple[str, str]:
    """Return the committed prompt label and SHA-256 of its exact UTF-8 bytes."""
    return PROMPT_VERSION, sha256(_PROMPT_PATH.read_bytes()).hexdigest()


def _system_prompt(config: HybridRetrievalConfig) -> str:
    if config.prompt_version != PROMPT_VERSION:
        raise ValueError("configured answer prompt version does not match canonical prompt")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _usage(client: StructuredExtractionClient) -> dict[str, int | float | None]:
    accessor = getattr(client, "usage_for_current_thread", None)
    raw = accessor() if callable(accessor) else getattr(client, "last_usage", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): value for key, value in raw.items()
        if (isinstance(value, (int, float)) and not isinstance(value, bool)) or value is None
    }


def _provider_attempts(client: StructuredExtractionClient) -> int:
    accessor = getattr(client, "attempts_for_current_thread", None)
    raw = accessor() if callable(accessor) else 1
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 1


def _merge_usage(total: dict[str, int | float | None], current: Mapping[str, int | float | None]) -> None:
    for key, value in current.items():
        prior = total.get(key)
        total[key] = prior + value if isinstance(prior, (int, float)) and isinstance(value, (int, float)) else value


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def validate_citations(
    answer: AnswerModel, context: ContextResult, *, allow_normalized: bool = True,
) -> dict[str, object]:
    """Check every quote against the text sent for its own tag, never original text."""
    failures: list[dict[str, str]] = []
    matches: list[dict[str, str]] = []
    if context.evidence_text and not answer.citations and not answer.evidence_gaps:
        failures.append({"tag": "", "quote": "", "reason": "missing_citation"})
    for cited in answer.citations:
        key = context.tag_to_key.get(cited.tag)
        if key is None:
            failures.append({"tag": cited.tag, "quote": cited.quote, "reason": "unknown_tag"})
            continue
        sent = context.sent_text.get(key, "")
        if not cited.quote.strip():
            failures.append({"tag": cited.tag, "quote": cited.quote, "reason": "empty_quote"})
        elif cited.quote in sent:
            matches.append({"tag": cited.tag, "quote": cited.quote, "match": "exact"})
        elif allow_normalized and _collapsed(cited.quote) in _collapsed(sent):
            matches.append({"tag": cited.tag, "quote": cited.quote, "match": "normalized"})
        else:
            failures.append({"tag": cited.tag, "quote": cited.quote, "reason": "quote_not_in_sent_text"})
    return {"status": "failed" if failures else "passed", "failures": failures, "matches": matches}


def _feedback(check: Mapping[str, object]) -> str:
    failures = check["failures"]
    assert isinstance(failures, list)
    details = "; ".join(
        f"tag={item['tag']!r}, quote={item['quote']!r}, reason={item['reason']}"
        for item in failures if isinstance(item, dict)
    )
    return (
        "Citation validation failed for: " + details + ". Return a complete replacement answer. "
        "Use only the evidence above and quote a contiguous span from the cited tag."
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((monotonic() - started) * 1000))


def answer_question(
    client: StructuredExtractionClient, question: str, context: ContextResult,
    *, config: HybridRetrievalConfig | None = None,
) -> tuple[AnswerModel, dict[str, int | float | None], int, tuple[dict[str, object], ...]]:
    """Generate once plus a bounded citation repair; return only a locally valid answer.

    On failure the exception retains safe trace material. The caller writes a failed
    trace and must not write an answer row.
    """
    settings = config or HybridRetrievalConfig()
    system = _system_prompt(settings)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if not context.evidence_text:
        gap = "Nothing was found in the inspected scope for this question."
        return AnswerModel(answer=gap, citations=[], evidence_gaps=[gap]), {}, 0, ()

    base_message = f"Question: {question}\n\nEvidence:\n{context.evidence_text}"
    attempts: list[dict[str, object]] = []
    usage: dict[str, int | float | None] = {}
    elapsed_ms = 0
    feedback: str | None = None
    for _ in range(settings.citation_repair_attempts + 1):
        model_text = base_message if feedback is None else base_message + "\n\n" + feedback
        started = monotonic()
        try:
            raw = client.extract(model_text, AnswerModel, system)
        except Exception as error:
            elapsed_ms += _elapsed_ms(started)
            _merge_usage(usage, _usage(client))
            attempts.append({"validated": False, "feedback": feedback, "provider_attempts": _provider_attempts(client),
                             "error_type": type(error).__name__})
            kind = "schema" if isinstance(error, (GenerationSchemaError, GenerationUnselectedTypeError,
                                                    ValidationError, ValueError)) else "provider"
            raise AnswerGenerationError(kind, type(error).__name__, attempts=tuple(attempts),
                                        usage=usage, elapsed_ms=elapsed_ms) from None
        elapsed_ms += _elapsed_ms(started)
        _merge_usage(usage, _usage(client))
        try:
            answer = AnswerModel.model_validate(raw)
        except ValidationError:
            attempts.append({"validated": False, "feedback": feedback, "provider_attempts": _provider_attempts(client),
                             "error_type": "ValidationError"})
            raise AnswerGenerationError("schema", "ValidationError", attempts=tuple(attempts),
                                        usage=usage, elapsed_ms=elapsed_ms) from None
        check = validate_citations(answer, context, allow_normalized=settings.allow_normalized_quote_match)
        attempts.append({"validated": check["status"] == "passed", "feedback": feedback,
                         "provider_attempts": _provider_attempts(client),
                         "answer": answer.model_dump(mode="json"), "citation_check": check})
        if check["status"] == "passed":
            return answer, usage, elapsed_ms, tuple(attempts)
        feedback = _feedback(check)
    raise CitationValidationError(attempts=tuple(attempts), usage=usage,
                                  elapsed_ms=elapsed_ms, citation_check=check)


def expand_citations(
    answer: AnswerModel, context: ContextResult, scope: Scope,
    *, allow_normalized: bool = True,
) -> list[dict[str, object]]:
    """Expand validated tags to output-contract citations with locator fields."""
    check = validate_citations(answer, context, allow_normalized=allow_normalized)
    if check["status"] != "passed":
        raise ValueError("cannot expand unvalidated citations")
    passages = {item.candidate.key: item.candidate for item in context.passages}
    matches = check["matches"]
    assert isinstance(matches, list)
    expanded: list[dict[str, object]] = []
    for item in matches:
        tag = item["tag"]
        candidate = passages[context.tag_to_key[tag]]
        expanded.append({
            "scope": {"tenant_id": scope.tenant_id, "project_id": scope.project_id},
            "source_id": candidate.source_id, "source_version": candidate.source_version,
            "passage_id": candidate.passage_id, "page": candidate.locator.get("page"),
            "section": candidate.locator.get("section"), "quote": item["quote"],
            "match": item["match"],
        })
    return expanded
