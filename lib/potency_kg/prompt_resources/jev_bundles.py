"""The fixed seven Jev questions and their allowed extraction types."""

from __future__ import annotations

from ..processing_policy import BUNDLE_ORDER

BUNDLE_TO_TYPES = {
    "product_context": ("Product", "Application", "Indication"),
    "alias_identity": ("Product", "AliasCandidate"),
    "potency_strategy": ("PotencyStrategy",),
    "assay_evidence": ("Assay", "MeasuredAttribute"),
    "regulatory_history": ("RegulatoryInteraction",),
    "review_conclusion": ("ReviewConclusion",),
    "redaction_uncertainty": ("RedactionMention",),
}
JEV_QUESTIONS = {
    "product_context": ("Does this passage identify product, application, indication, or applicability?", "Identify explicit product context only."),
    "alias_identity": ("Does this passage contain a product name, abbreviation, or alias?", "Capture names without inferring identity."),
    "potency_strategy": ("Does this passage discuss potency strategy?", "Include explicit potency context."),
    "assay_evidence": ("Does this passage describe assays or measured attributes?", "Require an explicit assay or measurement."),
    "regulatory_history": ("Does this passage record a regulatory interaction or chronology?", "Capture proposed, requested, revised, accepted, rejected, or unresolved events."),
    "review_conclusion": ("Does this passage state a reviewer or agency conclusion?", "Require an explicit conclusion or recommendation."),
    "redaction_uncertainty": ("Does this passage contain redaction or unresolved uncertainty?", "Do not reconstruct redacted content."),
}
assert tuple(JEV_QUESTIONS) == BUNDLE_ORDER
