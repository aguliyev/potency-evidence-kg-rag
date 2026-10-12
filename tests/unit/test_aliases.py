from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from potency_kg.aliases import (
    AliasCatalog,
    AliasCatalogStatus,
    AliasDecision,
    AliasRecord,
    AliasSeedReport,
    ApprovedAlias,
    normalize_alias,
)
from potency_kg.contracts import Scope


SCOPE = Scope("tenant", "project")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CASGEVY", "casgevy"),
        ("  exa\t\ncel  ", "exa cel"),
        ("Ｅｘａ－Ｃｅｌ", "exa-cel"),
        ("Straße", "strasse"),
    ],
)
def test_normalize_alias_uses_nfkc_casefold_and_collapsed_whitespace(value: str, expected: str) -> None:
    """Skipping Unicode normalization, case folding, or whitespace collapse breaks exact matching."""
    assert normalize_alias(value) == expected


def test_public_alias_contracts_are_frozen_snapshots() -> None:
    """Making a returned alias snapshot mutable would let callers change semantic identity after loading."""
    entry = ApprovedAlias(
        alias_id="alias-1",
        alias_value="CASGEVY",
        normalized_alias="casgevy",
        entity_type="Product",
        canonical_key="casgevy",
        canonical_name="CASGEVY",
        normalization_version="nfkc-casefold-whitespace-v1",
    )
    record = AliasRecord(
        alias_id="alias-1",
        alias_value="CASGEVY",
        normalized_alias="casgevy",
        entity_type="Product",
        canonical_key="casgevy",
        canonical_name="CASGEVY",
        status="approved",
        managed_by="static_config",
        confidence=None,
    )
    decision = AliasDecision("alias-1", "approved", "reviewer", "Product", "casgevy", "CASGEVY")
    report = AliasSeedReport(SCOPE, 3, 0, 0, 0, 3, "catalog-v1")
    status = AliasCatalogStatus(SCOPE, "catalog-v1", (), (), False)
    catalog = AliasCatalog(SCOPE, "catalog-v1", (entry,))

    for value in (entry, record, decision, report, status, catalog):
        with pytest.raises(FrozenInstanceError):
            value.scope = SCOPE  # type: ignore[misc]
    assert catalog.entries == (entry,)
    assert sorted((entry,)) == [entry]


def test_public_catalog_sequences_are_immutable_snapshots() -> None:
    """Retaining caller-owned lists would let callers change an already-returned semantic identity."""
    entry = ApprovedAlias(
        alias_id="alias-1",
        alias_value="CASGEVY",
        normalized_alias="casgevy",
        entity_type="Product",
        canonical_key="casgevy",
        canonical_name="CASGEVY",
        normalization_version="nfkc-casefold-whitespace-v1",
    )
    source_entries = [entry]
    source_versions = ["catalog-v1"]

    catalog = AliasCatalog(SCOPE, "catalog-v1", source_entries)
    status = AliasCatalogStatus(SCOPE, "catalog-v1", source_versions, source_versions, False)
    source_entries.clear()
    source_versions.append("catalog-v2")

    assert catalog.entries == (entry,)
    assert status.latest_jev_catalog_versions == ("catalog-v1",)
    assert status.latest_extraction_catalog_versions == ("catalog-v1",)
