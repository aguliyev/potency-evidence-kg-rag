from __future__ import annotations

import pytest

from potency_kg.contracts import AssertionDraft, EvidenceRelationshipDraft, Scope, ScopedPassageIdentity


def test_passage_identity_keeps_project_scope_when_local_ids_collide() -> None:
    """Removing tenant/project fields from identities must break this test."""
    first = ScopedPassageIdentity("tenant", "one", "source", "version", "passage")
    second = ScopedPassageIdentity("tenant", "two", "source", "version", "passage")
    assert first != second
    assert first.as_tuple() == ("tenant", "one", "source", "version", "passage")


def test_evidence_relationship_requires_allowed_endpoints_and_an_assertion_reference() -> None:
    """Allowing unsupported endpoints or evidence without an assertion must break this test."""
    with pytest.raises(ValueError):
        EvidenceRelationshipDraft("measures", "Product", "MeasuredAttribute", "a")
    with pytest.raises(ValueError):
        EvidenceRelationshipDraft("measures", "Assay", "MeasuredAttribute", None)
    relation = EvidenceRelationshipDraft("measures", "Assay", "MeasuredAttribute", "assertion-1")
    assert relation.assertion_local_id == "assertion-1"


def test_assertion_is_one_atomic_claim() -> None:
    """Accepting an empty or compound assertion collection must break this test."""
    with pytest.raises(ValueError):
        AssertionDraft(local_id="a", claim="")
    with pytest.raises(ValueError):
        AssertionDraft(local_id="a", claim=["first", "second"])  # type: ignore[arg-type]
    assert AssertionDraft(local_id="a", claim="The assay measured potency.").claim


def test_prompt_resources_are_complete_and_bundle_scoped() -> None:
    """Omitting schemas or admitting an unselected type must break this test."""
    from potency_kg.processing_policy import BUNDLE_ORDER
    from potency_kg.prompt_resources.extraction import TYPE_SCHEMAS, assemble_prompt, resource_hash
    from potency_kg.prompt_resources.jev_bundles import BUNDLE_TO_TYPES, JEV_QUESTIONS

    assert tuple(JEV_QUESTIONS) == BUNDLE_ORDER
    assert tuple(BUNDLE_TO_TYPES) == BUNDLE_ORDER
    assert all(question.strip() and criterion.strip() for question, criterion in JEV_QUESTIONS.values())
    assert {kind for kinds in BUNDLE_TO_TYPES.values() for kind in kinds} <= set(TYPE_SCHEMAS)
    prompt = assemble_prompt(("product_context",))
    assert all(kind in prompt for kind in BUNDLE_TO_TYPES["product_context"])
    assert "Assertion" in prompt and "SourceRef" in prompt
    assert "Assay" not in prompt
    with pytest.raises(ValueError):
        assemble_prompt(("not_a_bundle",))
    assert len(resource_hash()) == 64
