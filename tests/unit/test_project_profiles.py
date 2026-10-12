from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from potency_kg.aliases import AliasCatalog, ApprovedAlias
from potency_kg.contracts import Scope
from potency_kg.project_profiles import load_project_profile


SCOPE = Scope("public-fda", "casgevy-potency")
CATALOG = AliasCatalog(
    scope=SCOPE,
    version="alias-catalog-hand-checked",
    entries=(
        ApprovedAlias(
            alias_id="alias-casgevy",
            alias_value="CASGEVY",
            normalized_alias="casgevy",
            entity_type="Product",
            canonical_key="casgevy",
            canonical_name="CASGEVY",
            normalization_version="nfkc-casefold-whitespace-v1",
        ),
    ),
)


def test_profile_uses_the_supplied_catalog_and_preserves_descriptor_provenance(tmp_path: Path) -> None:
    """Constructing aliases locally instead of retaining the approved snapshot breaks this test."""
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text('{"scope": {"tenant_id": "public-fda", "project_id": "casgevy-potency"}}', encoding="utf-8")

    profile = load_project_profile(SCOPE, descriptor, CATALOG)

    assert profile.name == "casgevy-v1"
    assert profile.alias_catalog is CATALOG
    assert profile.descriptor_sha256
    assert profile.applications == {"125785": "casgevy-tdt", "125787": "casgevy-scd"}
    with pytest.raises(FrozenInstanceError):
        profile.alias_catalog = CATALOG  # type: ignore[misc]


def test_profile_version_changes_for_descriptor_or_catalog_version(tmp_path: Path) -> None:
    """Omitting either the descriptor digest or catalog version would reuse stale semantic state."""
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("one", encoding="utf-8")
    original = load_project_profile(SCOPE, descriptor, CATALOG)

    descriptor.write_text("two", encoding="utf-8")
    changed_descriptor = load_project_profile(SCOPE, descriptor, CATALOG)
    changed_catalog = load_project_profile(
        SCOPE,
        descriptor,
        AliasCatalog(scope=SCOPE, version="different-approved-catalog", entries=CATALOG.entries),
    )

    assert original.profile_version != changed_descriptor.profile_version
    assert changed_descriptor.profile_version != changed_catalog.profile_version
