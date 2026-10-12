"""Deterministic assembly of the structured-extraction instruction."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from ..processing_policy import BUNDLE_ORDER
from .jev_bundles import BUNDLE_TO_TYPES

UNIVERSAL_INSTRUCTIONS = (
    "Extract only explicit passage-local claims. Every evidence relationship must reference "
    "one atomic assertion and one exact supporting quote. Use passage-local IDs for all "
    "references. Do not reconstruct, guess, or infer concealed redacted content."
)
RETRY_INSTRUCTION = (
    "The previous response used a type outside the selected bundles. Return a complete "
    "replacement response using only the emphasized allowed types."
)

TYPE_SCHEMAS: dict[str, dict[str, object]] = {
    "Product": {"required": ("local_id", "name", "supporting_quote")},
    "Application": {"required": ("local_id", "application_id", "supporting_quote")},
    "Indication": {"required": ("local_id", "name", "supporting_quote")},
    "AliasCandidate": {"required": ("local_id", "candidate", "target_local_id", "supporting_quote")},
    "PotencyStrategy": {"required": ("local_id", "description", "supporting_quote")},
    "Assay": {"required": ("local_id", "name", "supporting_quote")},
    "MeasuredAttribute": {"required": ("local_id", "name", "supporting_quote")},
    "RegulatoryInteraction": {"required": ("local_id", "chronology_state", "supporting_quote")},
    "ReviewConclusion": {"required": ("local_id", "conclusion", "supporting_quote")},
    "RedactionMention": {"required": ("local_id", "model_start", "model_end")},
}


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
    return (
        f"{UNIVERSAL_INSTRUCTIONS}\n"
        f"Allowed types: {', '.join(types)}\n"
        f"Selected type schemas: {json.dumps(schemas, sort_keys=True, separators=(',', ':'))}\n"
        "Universal records: Assertion, SourceRef, EvidenceRelationship."
    )


def resource_hash() -> str:
    bundle_path = Path(__file__).with_name("jev_bundles.py")
    return sha256(Path(__file__).read_bytes() + b"\0" + bundle_path.read_bytes()).hexdigest()
