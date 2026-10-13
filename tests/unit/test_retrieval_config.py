from __future__ import annotations

import pytest

from potency_kg.retrieval_config import HybridRetrievalConfig


def test_defaults_are_valid_and_hash_is_stable() -> None:
    config = HybridRetrievalConfig()
    assert config.config_hash() == HybridRetrievalConfig().config_hash()
    assert config.as_dict()["graph_share"] == 0.4


def test_any_change_changes_the_hash() -> None:
    assert HybridRetrievalConfig(graph_share=0.5).config_hash() != HybridRetrievalConfig().config_hash()


def test_graph_score_adds_hop_and_edge_bonus() -> None:
    config = HybridRetrievalConfig()
    plain = config.graph_score(0.5, 1, None)
    assert plain == pytest.approx(0.5 + 0.05)
    assert config.graph_score(0.5, 1, "proposed_in") == pytest.approx(plain + 0.05)
    assert config.graph_score(0.5, 3, "applies_to") == pytest.approx(0.5 + 0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_share": 1.5},
        {"neighbors_per_entity": 0},
        {"max_hops": 2},  # hop_bonus still has 3 values
        {"traversal": "age"},  # default 3 hops exceeds the AGE cap
        {"edge_type_bonus": (("a", 0.1), ("a", 0.2))},
        {"similarity_weight": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HybridRetrievalConfig(**kwargs)
