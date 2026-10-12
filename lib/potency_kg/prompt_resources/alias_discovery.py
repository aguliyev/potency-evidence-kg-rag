"""Versioned instruction for optional structured alias discovery."""

from __future__ import annotations

import json


INSTRUCTIONS = (
    "Identify only explicit Product or Assay aliases in the supplied passages. "
    "Return every proposal with the exact passage_id and an exact supporting_quote copied "
    "from that passage. Do not infer aliases, normalize quotes, or approve a proposal."
)


def assemble_prompt() -> str:
    """Return the stable instruction whose digest participates in run identity."""
    return f"{INSTRUCTIONS}\nResponse fields: {json.dumps(('alias_value', 'canonical_name', 'entity_type', 'passage_id', 'supporting_quote', 'occurrence_index', 'confidence'))}"
