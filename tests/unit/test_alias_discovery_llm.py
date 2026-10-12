from __future__ import annotations

import pytest
from pydantic import ValidationError

from potency_kg.alias_discovery import (
    AliasDiscoveryItem,
    AliasDiscoveryResponse,
    DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY,
)


def _item(**changes: object) -> dict[str, object]:
    item: dict[str, object] = {
        "alias_value": "exa-cel",
        "canonical_name": "Exagamglogene autotemcel",
        "entity_type": "Product",
        "passage_id": "p1",
        "supporting_quote": "Exagamglogene autotemcel (exa-cel)",
        "occurrence_index": 0,
        "confidence": 0.91,
    }
    item.update(changes)
    return item


def test_llm_alias_discovery_is_disabled_unless_explicitly_enabled() -> None:
    """Changing the default would allow paid, non-deterministic work in an ordinary pipeline run."""
    assert DEFAULT_ENABLE_LLM_ALIAS_DISCOVERY is False


def test_alias_discovery_response_accepts_only_complete_typed_grounding_items() -> None:
    """Dropping a required grounding field or accepting an unsupported entity type would make evidence unauditable."""
    response = AliasDiscoveryResponse.model_validate({"aliases": [_item()]})

    assert response.aliases == [AliasDiscoveryItem.model_validate(_item())]
    assert response.aliases[0].entity_type == "Product"


@pytest.mark.parametrize(
    "payload",
    [
        {"aliases": [], "unexpected": "value"},
        {"aliases": [_item(unexpected="value")]},
        {"aliases": [_item(entity_type="Application")]},
        {"aliases": [_item(alias_value="")]},
        {"aliases": [_item(confidence=1.01)]},
        {"aliases": [_item(occurrence_index=-1)]},
    ],
)
def test_alias_discovery_response_rejects_untyped_or_extra_provider_fields(payload: dict[str, object]) -> None:
    """Permissive parsing could turn provider drift into incorrect alias evidence."""
    with pytest.raises(ValidationError):
        AliasDiscoveryResponse.model_validate(payload)
