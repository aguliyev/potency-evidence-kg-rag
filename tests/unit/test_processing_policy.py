from __future__ import annotations

import pytest

from potency_kg.processing_policy import JevSelectionPolicy, select_bundles


def test_policy_uses_documented_defaults_and_inclusive_thresholds_without_mutation() -> None:
    """Changing defaults, mutating defaults, or using > must break this test."""
    policy = JevSelectionPolicy()
    defaults = policy.resolve({})
    overrides = policy.resolve({"product_context": 0.7})
    assert defaults.values["product_context"] == 0.50
    assert defaults.values["review_conclusion"] == 0.35
    assert defaults.values["redaction_uncertainty"] == 0.35
    assert overrides.values["product_context"] == 0.7
    assert defaults.values["product_context"] == 0.5
    selected = select_bundles(
        {"product_context": 0.50, "review_conclusion": 0.35, "redaction_uncertainty": 0.35}, defaults
    )
    assert selected == ("product_context", "review_conclusion", "redaction_uncertainty")
    with pytest.raises(TypeError):
        defaults.values["product_context"] = 0.1  # type: ignore[index]


def test_bundle_selection_has_fixed_seven_bundle_order() -> None:
    """Returning dictionary insertion order instead of policy order must break this test."""
    thresholds = JevSelectionPolicy().resolve({})
    assert select_bundles({name: 1.0 for name in reversed(thresholds.values)}, thresholds) == (
        "product_context",
        "alias_identity",
        "potency_strategy",
        "assay_evidence",
        "regulatory_history",
        "review_conclusion",
        "redaction_uncertainty",
    )
