from __future__ import annotations

from pathlib import Path

import pytest

from potency_kg.project_profiles import load_project_profile


def test_official_profile_resolves_aliases_and_descriptor_provenance(tmp_path: Path) -> None:
    """Removing an official alias or descriptor hashing must break this test."""
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text('{"scope": {"tenant_id": "public-fda", "project_id": "casgevy-potency"}}', encoding="utf-8")
    profile = load_project_profile(("public-fda", "casgevy-potency"), descriptor)
    assert profile.name == "casgevy-v1"
    assert profile.aliases["CASGEVY"] == "casgevy"
    assert profile.aliases["exagamglogene autotemcel"] == "casgevy"
    assert profile.aliases["exa-cel"] == "casgevy"
    assert profile.descriptor_sha256
    with pytest.raises(TypeError):
        profile.aliases["new-alias"] = "casgevy"  # type: ignore[index]


def test_profile_version_changes_for_alias_and_descriptor_changes(tmp_path: Path) -> None:
    """Ignoring either alias or descriptor bytes must break this test."""
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("one", encoding="utf-8")
    scope = ("public-fda", "casgevy-potency")
    original = load_project_profile(scope, descriptor)
    descriptor.write_text("two", encoding="utf-8")
    changed_descriptor = load_project_profile(scope, descriptor)
    assert original.profile_version != changed_descriptor.profile_version
    unknown = load_project_profile(("other", "project"), descriptor)
    assert unknown.name == "empty-v1"
    assert unknown.aliases == {}
