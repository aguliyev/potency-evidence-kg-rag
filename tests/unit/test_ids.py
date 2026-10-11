from __future__ import annotations

import re
from types import MappingProxyType

from potency_kg.contracts import Scope, ScopedPassageIdentity
from potency_kg.ids import (
    canonical_json,
    derive_embedding_version,
    derive_extraction_version,
    derive_graph_name,
    derive_graph_version,
    derive_index_version,
    derive_jev_response_version,
    derive_jev_selection_version,
    derive_scoped_id,
)


def test_canonical_json_is_deterministic() -> None:
    """Dropping sorted-key canonicalization must break this test."""
    assert canonical_json({"b": 2, "a": ["x", "y"]}) == '{"a":["x","y"],"b":2}'
    assert canonical_json(MappingProxyType({"b": 2, "a": ["x", "y"]})) == '{"a":["x","y"],"b":2}'


def test_each_version_changes_with_its_declared_input() -> None:
    """Ignoring a declared version input must break this test."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "id")
    embedding = derive_embedding_version("text-embedding-3-large", 3072, "float", "provider_unit_length_v1")
    assert embedding != derive_embedding_version("text-embedding-3-large", 1, "float", "provider_unit_length_v1")
    assert derive_index_version(embedding, [(passage, "text-hash")]) != derive_index_version(embedding, [(passage, "other")])
    profile = {
        "name": "official",
        "aliases": {"Alias": "product"},
        "descriptor_sha256": "descriptor",
        "profile_version": "profile",
    }
    response_inputs = {
        "model": "jev",
        "questions": {"q": {"question": "q", "criterion": "c"}},
        "redacted_input": "text",
        "normalization": "redaction-v1",
        "prompt_hash": "prompt",
        "bundle_hash": "bundle",
        "profile": profile,
    }
    response = derive_jev_response_version(**response_inputs)
    for key, changed in {
        "model": "other-model",
        "questions": {"q": {"question": "other", "criterion": "c"}},
        "redacted_input": "other text",
        "normalization": "redaction-v2",
        "prompt_hash": "other-prompt",
        "bundle_hash": "other-bundle",
        "profile": {**profile, "aliases": {"Other": "product"}},
    }.items():
        assert response != derive_jev_response_version(**{**response_inputs, key: changed})
    selection = derive_jev_selection_version(response, {"product_context": 0.5})
    assert selection != derive_jev_selection_version(response, {"product_context": 0.6})
    extraction_inputs = {
        "model": "model",
        "instructor_mode": "openai",
        "selected_bundles": ["product_context"],
        "prompt_hash": "prompt",
        "schema_hash": "schema",
        "normalization": "redaction-v1",
        "profile": profile,
    }
    extraction = derive_extraction_version(**extraction_inputs)
    for key, changed in {
        "model": "other-model",
        "instructor_mode": "other-mode",
        "selected_bundles": ["assay_evidence"],
        "prompt_hash": "other-prompt",
        "schema_hash": "other-schema",
        "normalization": "redaction-v2",
        "profile": {**profile, "descriptor_sha256": "other-descriptor"},
    }.items():
        assert extraction != derive_extraction_version(**{**extraction_inputs, key: changed})
    graph = derive_graph_version("schema", [("s", "v")], ["run"], "structural", False, [], "profile")
    assert graph != derive_graph_version("schema", [("s", "v")], ["run"], "semantic", False, [], "profile")
    assert re.fullmatch(r"kg_[a-f0-9]{16}_[a-f0-9]{16}", derive_graph_name(Scope("tenant", "project"), graph))


def test_scoped_ids_include_tenant_and_project_even_when_local_ids_collide() -> None:
    """Dropping either scope component from a graph identifier must break this test."""
    first = derive_scoped_id("passage", Scope("tenant", "one"), "source", "version", "passage")
    second = derive_scoped_id("passage", Scope("tenant", "two"), "source", "version", "passage")
    assert first != second
    assert re.fullmatch(r"passage:[a-f0-9]{32}", first)
