"""Canonical, input-derived identities for immutable processing artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import Scope, ScopedPassageIdentity


def canonical_json(value: Any) -> str:
    def mapping_default(candidate: object) -> object:
        if isinstance(candidate, Mapping):
            return dict(candidate)
        raise TypeError(f"Object of type {type(candidate).__name__} is not JSON serializable")

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=mapping_default,
    )


def _version(kind: str, payload: Any) -> str:
    digest = hashlib.sha256(f"{kind}:".encode() + canonical_json(payload).encode()).hexdigest()
    return f"{kind}-{digest}"


def _identity_value(identity: ScopedPassageIdentity | tuple[str, ...]) -> tuple[str, ...]:
    return identity.as_tuple() if isinstance(identity, ScopedPassageIdentity) else tuple(identity)


def derive_scoped_id(kind: str, scope: Scope, *parts: str) -> str:
    """Derive a compact logical ID whose digest always includes tenant and project."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
        raise ValueError("scoped identity kind must be a lowercase internal identifier")
    payload = (scope.tenant_id, scope.project_id, *parts)
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:32]
    return f"{kind}:{digest}"


def derive_embedding_version(model: str, dimensions: int, encoding_format: str, normalization: str) -> str:
    return _version("embedding", {"model": model, "dimensions": dimensions, "encoding_format": encoding_format, "normalization": normalization})


def derive_index_version(embedding_version: str, members: Iterable[tuple[ScopedPassageIdentity | tuple[str, ...], str]]) -> str:
    canonical_members = sorted([{"identity": _identity_value(identity), "text_sha256": text_hash} for identity, text_hash in members], key=canonical_json)
    return _version("index", {"embedding_version": embedding_version, "members": canonical_members})


def derive_jev_response_version(
    *,
    model: str,
    questions: Mapping[str, object],
    redacted_input: str,
    normalization: str,
    prompt_hash: str,
    bundle_hash: str,
    profile: Mapping[str, object],
) -> str:
    return _version(
        "jev-response",
        {
            "model": model,
            "questions": questions,
            "normalization": normalization,
            "input_sha256": hashlib.sha256(redacted_input.encode()).hexdigest(),
            "prompt_hash": prompt_hash,
            "bundle_hash": bundle_hash,
            "profile": profile,
        },
    )


def derive_jev_selection_version(jev_response_version: str, thresholds: Mapping[str, float]) -> str:
    return _version("jev-selection", {"jev_response_version": jev_response_version, "thresholds": thresholds})


def derive_extraction_version(
    *,
    model: str,
    instructor_mode: str,
    selected_bundles: Iterable[str],
    prompt_hash: str,
    schema_hash: str,
    normalization: str,
    profile: Mapping[str, object],
) -> str:
    return _version(
        "extraction",
        {
            "model": model,
            "instructor_mode": instructor_mode,
            "selected_bundles": list(selected_bundles),
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "normalization": normalization,
            "profile": profile,
        },
    )


def derive_graph_version(
    scope: Scope,
    projection_schema: str,
    active_heads: Iterable[tuple[str, str]],
    active_extraction_ids: Iterable[str],
    mode: str,
    allow_partial: bool,
    omitted: Iterable[ScopedPassageIdentity | tuple[str, ...]],
    profile_version: str,
    index_version: str | None,
    embedding_version: str | None,
) -> str:
    return _version(
        "graph",
        {
            "scope": scope.as_tuple(),
            "projection_schema": projection_schema,
            "active_heads": sorted(active_heads),
            "active_extraction_ids": sorted(active_extraction_ids),
            "mode": mode,
            "allow_partial": allow_partial,
            "omitted": sorted(_identity_value(value) for value in omitted),
            "profile_version": profile_version,
            "index_version": index_version,
            "embedding_version": embedding_version,
        },
    )


def derive_graph_name(scope: Scope, graph_version: str) -> str:
    scope_digest = hashlib.sha256(f"{scope.tenant_id}/{scope.project_id}".encode()).hexdigest()[:16]
    version_digest = hashlib.sha256(graph_version.encode()).hexdigest()[:16]
    return f"kg_{scope_digest}_{version_digest}"
