"""Deterministic assembly of the structured-extraction instruction (texts live in ``potency_kg.prompts``)."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from ..contracts import ALLOWED_RELATION_ENDPOINTS
from ..processing_policy import BUNDLE_ORDER
from ..prompts import (  # noqa: F401 (re-exported)
    ASSERTION_RULES,
    ID_RULES,
    QUOTE_RULES,
    REDACTION_RULES,
    RELATIONSHIP_RULES,
    RETRY_INSTRUCTION,
    SOURCE_REF_RULES,
    TYPE_GUIDANCE,
    TYPE_SCHEMAS,
    UNIVERSAL_INSTRUCTIONS,
)
from .jev_bundles import BUNDLE_TO_TYPES


def _relation_table(types: list[str]) -> str:
    """Allowed relation types whose endpoints can exist in this response (derived, so it cannot drift)."""
    available = set(types) | {"Assertion", "SourcePassage"}
    lines = []
    for relation, pairs in sorted(ALLOWED_RELATION_ENDPOINTS.items()):
        usable = sorted(pair for pair in pairs if set(pair) <= available)
        if usable:
            lines.append(f"- {relation}: " + " | ".join(f"{source} -> {target}" for source, target in usable))
    return "\n".join(lines)


def assemble_prompt(selected_bundles: tuple[str, ...]) -> str:
    unknown = set(selected_bundles) - set(BUNDLE_ORDER)
    if unknown:
        raise ValueError(f"unknown extraction bundles: {sorted(unknown)}")
    types: list[str] = []
    selected = set(selected_bundles)
    for bundle in BUNDLE_ORDER:
        if bundle not in selected:
            continue
        for kind in BUNDLE_TO_TYPES[bundle]:
            if kind not in types:
                types.append(kind)
    schemas = {kind: TYPE_SCHEMAS[kind] for kind in types}
    sections = [
        UNIVERSAL_INSTRUCTIONS,
        f"Allowed entity types (use no other type): {', '.join(types)}",
        "What to extract:\n" + "\n".join(f"- {kind}: {TYPE_GUIDANCE[kind]}" for kind in types),
        f"Selected type schemas: {json.dumps(schemas, sort_keys=True, separators=(',', ':'))}",
        QUOTE_RULES,
        ID_RULES,
        ASSERTION_RULES,
        RELATIONSHIP_RULES.format(relation_table=_relation_table(types)),
        SOURCE_REF_RULES,
    ]
    if "RedactionMention" in types:
        sections.append(REDACTION_RULES)
    sections.append("Universal records: Assertion, SourceRef, EvidenceRelationship.")
    return "\n\n".join(sections)


def resource_hash() -> str:
    parts = (Path(__file__), Path(__file__).with_name("jev_bundles.py"), Path(__file__).parent.parent / "prompts.py")
    return sha256(b"\0".join(path.read_bytes() for path in parts)).hexdigest()
