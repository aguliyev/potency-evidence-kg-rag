"""Deterministic Jev selection with an explicit recall/cost tradeoff.

Lower thresholds improve recall but cause more LLM calls, cost, and noise; higher
thresholds reduce cost and noise while increasing false-negative risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

BundleName = str
BUNDLE_ORDER: tuple[BundleName, ...] = (
    "product_context", "alias_identity", "potency_strategy", "assay_evidence",
    "regulatory_history", "review_conclusion", "redaction_uncertainty",
)
DEFAULT_THRESHOLDS: Mapping[BundleName, float] = MappingProxyType({
    "product_context": 0.50,
    "alias_identity": 0.50,
    "potency_strategy": 0.50,
    "assay_evidence": 0.50,
    "regulatory_history": 0.50,
    "review_conclusion": 0.35,
    "redaction_uncertainty": 0.35,
})


@dataclass(frozen=True)
class ResolvedJevThresholds:
    values: Mapping[BundleName, float]


class JevSelectionPolicy:
    """Resolve per-bundle thresholds; lower values trade more cost/noise for recall."""

    def resolve(self, overrides: Mapping[BundleName, float]) -> ResolvedJevThresholds:
        unknown = set(overrides) - set(BUNDLE_ORDER)
        if unknown:
            raise ValueError(f"unknown Jev bundle overrides: {sorted(unknown)}")
        merged = dict(DEFAULT_THRESHOLDS)
        for name, value in overrides.items():
            if not 0 <= value <= 1:
                raise ValueError("thresholds must be probabilities")
            merged[name] = value
        return ResolvedJevThresholds(MappingProxyType(merged))


def select_bundles(probabilities: Mapping[BundleName, float], thresholds: ResolvedJevThresholds) -> tuple[BundleName, ...]:
    return tuple(name for name in BUNDLE_ORDER if probabilities.get(name, -1.0) >= thresholds.values[name])
