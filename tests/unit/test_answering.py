from __future__ import annotations

import pytest

from potency_kg.answering import (
    AnswerGenerationError,
    AnswerModel,
    CitationValidationError,
    answer_question,
    expand_citations,
    prompt_metadata,
)
from potency_kg.context import build_context, SelectedPassage
from potency_kg.contracts import Scope
from potency_kg.llm_client import GenerationSchemaError
from potency_kg.retrieval import RetrievalCandidate
from potency_kg.retrieval_config import HybridRetrievalConfig


def candidate(text: str, *, locator: dict[str, object] | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        ("fda-review", "v2", "passage-7"), "fda-review", "fda-review", "v2", "passage-7",
        text, "sha", locator or {"page": 11, "section": "3.2"}, "vector", 0.9, 0.9,
        None, None, None, (),
    )


def context(text: str = "The assay measures HbF potency."):
    return build_context((candidate(text),), 2000, HybridRetrievalConfig())


class FakeClient:
    def __init__(self, responses: list[object], usage: list[dict[str, int]] | None = None) -> None:
        self.responses = list(responses)
        self.usages = list(usage or [{} for _ in responses])
        self.calls: list[tuple[str, type, str]] = []
        self.last_usage: dict[str, int] = {}

    def extract(self, model_text: str, response_model: type, prompt: str):
        self.calls.append((model_text, response_model, prompt))
        self.last_usage = self.usages.pop(0)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def usage_for_current_thread(self) -> dict[str, int]:
        return self.last_usage

    def attempts_for_current_thread(self) -> int:
        return 1


def test_valid_answer_expands_exact_citation_to_contract_fields() -> None:
    reply = {"answer": "HbF potency is measured [S1].", "citations": [
        {"tag": "S1", "quote": "assay measures HbF"}], "evidence_gaps": []}
    client = FakeClient([reply], [{"prompt_tokens": 10, "completion_tokens": 4}])
    selected = context()

    answer, usage, elapsed_ms, attempts = answer_question(client, "What measures potency?", selected)

    assert isinstance(answer, AnswerModel)
    assert answer.answer == reply["answer"]
    assert usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert elapsed_ms >= 0
    assert len(attempts) == 1 and attempts[0]["citation_check"]["status"] == "passed"
    assert client.calls[0][1] is AnswerModel
    assert client.calls[0][0] == "Question: What measures potency?\n\nEvidence:\n" + selected.evidence_text
    assert "Answer only from" in client.calls[0][2]
    assert expand_citations(answer, selected, Scope("public-fda", "casgevy-potency")) == [
        {"scope": {"tenant_id": "public-fda", "project_id": "casgevy-potency"},
         "source_id": "fda-review", "source_version": "v2", "passage_id": "passage-7",
         "page": 11, "section": "3.2", "quote": "assay measures HbF", "match": "exact"}
    ]


def test_whitespace_only_quote_difference_is_flagged_normalized() -> None:
    client = FakeClient([{"answer": "Measured [S1].", "citations": [
        {"tag": "S1", "quote": "assay measures HbF"}], "evidence_gaps": []}])
    selected = context("The assay\n\fmeasures  HbF potency.")

    answer, _, _, attempts = answer_question(client, "What measures potency?", selected)

    assert attempts[0]["citation_check"]["status"] == "passed"
    assert expand_citations(answer, selected, Scope("t", "p"))[0]["match"] == "normalized"


@pytest.mark.parametrize("bad_quote", ["not present", "truncated-away phrase"])
def test_invalid_or_truncated_away_quote_fails_after_one_repair(bad_quote: str) -> None:
    item = candidate("Visible words. " + "x" * 300 + " truncated-away phrase")
    selected = build_context((SelectedPassage(item, item.text[:200], True),), 2000, HybridRetrievalConfig())
    reply = {"answer": "Claim [S1].", "citations": [{"tag": "S1", "quote": bad_quote}], "evidence_gaps": []}
    client = FakeClient([reply, reply], [{"prompt_tokens": 5}, {"prompt_tokens": 7}])

    with pytest.raises(CitationValidationError) as caught:
        answer_question(client, "What is stated?", selected)

    error = caught.value
    assert error.failure_kind == "citation_validation"
    assert error.citation_check["status"] == "failed"
    assert len(error.attempts) == 2
    assert all(attempt["validated"] is False for attempt in error.attempts)
    assert error.usage == {"prompt_tokens": 12}
    assert client.calls[0][0] in client.calls[1][0]
    assert bad_quote in client.calls[1][0]
    assert error.attempts[1]["feedback"] in client.calls[1][0]


def test_unknown_tag_fails_even_if_quote_exists_in_other_evidence() -> None:
    reply = {"answer": "Claim [S2].", "citations": [{"tag": "S2", "quote": "assay measures HbF"}], "evidence_gaps": []}
    client = FakeClient([reply, reply])

    with pytest.raises(CitationValidationError) as caught:
        answer_question(client, "What measures potency?", context())

    assert caught.value.citation_check["failures"][0]["tag"] == "S2"
    assert "S2" in caught.value.attempts[1]["feedback"]


def test_nonempty_evidence_rejects_uncited_factual_answer_despite_gap_list() -> None:
    reply = {"answer": "The assay measures HbF potency.", "citations": [],
             "evidence_gaps": ["The acceptance threshold was not found."]}
    client = FakeClient([reply, reply])

    with pytest.raises(CitationValidationError) as caught:
        answer_question(client, "What measures potency?", context())

    assert caught.value.failure_kind == "citation_validation"
    assert caught.value.citation_check["failures"][0]["reason"] == "missing_citation"
    assert len(caught.value.attempts) == 2


def test_repair_is_capped_at_one_even_when_config_allows_more() -> None:
    reply = {"answer": "Unsupported [S1].", "citations": [
        {"tag": "S1", "quote": "words not in evidence"}], "evidence_gaps": []}
    client = FakeClient([reply, reply])

    with pytest.raises(CitationValidationError) as caught:
        answer_question(client, "What measures potency?", context(),
                        config=HybridRetrievalConfig(citation_repair_attempts=5))

    assert len(client.calls) == 2
    assert len(caught.value.attempts) == 2
    assert caught.value.citation_check["status"] == "failed"


def test_empty_evidence_returns_scoped_gap_without_provider_call() -> None:
    selected = build_context((), 200, HybridRetrievalConfig())
    client = FakeClient([])

    answer, usage, elapsed_ms, attempts = answer_question(client, "What is the assay?", selected)

    assert "nothing was found in the inspected scope" in answer.answer.lower()
    assert answer.citations == []
    assert answer.evidence_gaps
    assert usage == {} and elapsed_ms == 0 and attempts == ()
    assert client.calls == []


@pytest.mark.parametrize("exception,kind", [(RuntimeError("secret API key xyz"), "provider"),
                                           (GenerationSchemaError("secret schema text"), "schema")])
def test_provider_and_schema_failures_are_sanitized(exception: Exception, kind: str) -> None:
    client = FakeClient([exception])

    with pytest.raises(AnswerGenerationError) as caught:
        answer_question(client, "What measures potency?", context())

    error = caught.value
    assert error.failure_kind == kind
    assert error.error_type == type(exception).__name__
    assert len(error.attempts) == 1
    assert "secret" not in str(error)
    assert "secret" not in repr(error.attempts)


def test_missing_required_answer_fields_are_schema_failures() -> None:
    client = FakeClient([{"answer": "unsupported"}])

    with pytest.raises(AnswerGenerationError) as caught:
        answer_question(client, "What measures potency?", context())

    assert caught.value.failure_kind == "schema"
    assert caught.value.error_type == "ValidationError"
    assert len(caught.value.attempts) == 1


def test_prompt_metadata_hash_changes_with_resource_bytes() -> None:
    version, digest = prompt_metadata()
    assert version == "answer-prompt-v1"
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
