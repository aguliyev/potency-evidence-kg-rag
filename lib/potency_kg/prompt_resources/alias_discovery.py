"""Versioned instruction for optional structured alias discovery."""

from __future__ import annotations

import json

from ..prompts import ALIAS_DISCOVERY_INSTRUCTIONS as INSTRUCTIONS


def assemble_prompt() -> str:
    """Return the stable instruction whose digest participates in run identity."""
    return f"{INSTRUCTIONS}\nResponse fields: {json.dumps(('alias_value', 'canonical_name', 'entity_type', 'passage_id', 'supporting_quote', 'occurrence_index', 'confidence'))}"
