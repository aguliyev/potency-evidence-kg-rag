from __future__ import annotations

import pytest

from potency_kg.evaluation import aggregate_results, score_case


SCOPE = {"tenant_id": "tenant", "project_id": "project"}
REF = {"scope": SCOPE, "source_id": "source", "source_version": "v1", "passage_id": "p1"}


def case(**changes):
    return {"case_id": "c1", "case_version": "v1", "split": "holdout", "scope": SCOPE,
            "question_id": "q1", "question": "What happened?", "expected_evidence_refs": [REF],
            "retrieval_checks": {"must_include_any_expected_ref": True},
            "deterministic_answer_checks": {"must_mention_all": ["yes"]},
            "risk_rationale": "source grounding"} | changes


def trace(**changes):
    return {"trace_id": "t1", "question_id": "q1", "question": "What happened?", "mode": "graph",
            "scope": SCOPE, "status": "succeeded", "selected_passage_ids": ["p1"],
            "candidate_passages": [{"source_id": "source", "source_version": "v1", "passage_id": "p1"}],
            "selected_passages": [{"source_id": "source", "source_version": "v1", "passage_id": "p1"}]} | changes


def answer(**changes):
    return {"trace_id": "t1", "question_id": "q1", "mode": "graph", "scope": SCOPE,
            "answer": "No", "evidence_gaps": [], "citations": []} | changes


def test_retrieval_and_answer_are_independent_and_aggregate_recomputes():
    row = score_case(case(), trace(), answer())
    assert row["status"] == "completed"
    assert row["scores"]["retrieval"]["passed"] is True
    assert row["scores"]["answer"]["passed"] is False
    assert aggregate_results([row])["holdout"]["retrieval"]["passed"] == 1
    assert aggregate_results([row])["holdout"]["answer"]["passed"] == 0


def test_failed_generation_and_missing_trace_have_distinct_denominators():
    failed = score_case(case(), trace(status="failed"), None)
    missing = score_case(case(case_id="c2"), None, None)
    totals = aggregate_results([failed, missing])
    assert failed["status"] == "failed"
    assert missing["status"] == "unscorable"
    assert totals["holdout"]["statuses"] == {"completed": 0, "failed": 1, "unscorable": 1}
    assert totals["holdout"]["retrieval"]["denominator"] == 1
    assert totals["holdout"]["answer"]["denominator"] == 0


def test_failed_attempt_without_retrieval_evidence_is_explicitly_unscored():
    row = score_case(case(), trace(status="failed", candidate_passages=[],
                                   selected_passages=[], selected_passage_ids=[]), None)
    assert row["status"] == "failed"
    assert row["scores"] == {"retrieval": None, "answer": None}
    assert "retrieval_not_recorded" in row["errors"]


@pytest.mark.parametrize("bad_ref", [
    {**REF, "scope": {"tenant_id": "other", "project_id": "project"}},
    {**REF, "passage_id": "absent"},
])
def test_expected_reference_must_be_in_scope_and_saved_trace(bad_ref):
    with pytest.raises(ValueError, match="expected evidence"):
        score_case(case(expected_evidence_refs=[bad_ref]), trace(), answer())
