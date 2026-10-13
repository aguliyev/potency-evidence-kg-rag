"""Bundle -> extraction types; the Jev question texts live in ``potency_kg.prompts``."""

from __future__ import annotations

from ..processing_policy import BUNDLE_ORDER
from ..prompts import JEV_QUESTIONS  # noqa: F401 (re-exported)

BUNDLE_TO_TYPES = {
    "product_context": ("Product", "Application", "Indication"),
    "alias_identity": ("Product", "AliasCandidate"),
    "potency_strategy": ("PotencyStrategy",),
    "assay_evidence": ("Assay", "MeasuredAttribute"),
    "regulatory_history": ("RegulatoryInteraction",),
    "review_conclusion": ("ReviewConclusion",),
    "redaction_uncertainty": ("RedactionMention",),
}
assert tuple(JEV_QUESTIONS) == BUNDLE_ORDER
