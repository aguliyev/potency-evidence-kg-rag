"""Trusted project defaults and immutable supplied-input snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from .aliases import AliasCatalog
from .contracts import Scope
from .ids import canonical_json


CASGEVY_APPLICATIONS: Mapping[str, str] = MappingProxyType(
    {"125785": "casgevy-tdt", "125787": "casgevy-scd"}
)
TRUSTED_PROFILE_DESCRIPTOR_PATHS: Mapping[str, str] = MappingProxyType(
    {"casgevy-v1": "configs/casgevy-potency.json"}
)


def trusted_applications_for_profile(profile_name: str) -> Mapping[str, str]:
    return CASGEVY_APPLICATIONS if profile_name == "casgevy-v1" else MappingProxyType({})


def trusted_descriptor_path_for_profile(profile_name: str) -> str:
    return TRUSTED_PROFILE_DESCRIPTOR_PATHS.get(profile_name, "")


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def protected_input_hashes(root: Path) -> Mapping[str, str]:
    """Return a content snapshot without emitting supplied file contents."""
    hashes: dict[str, str] = {}
    for directory in ("configs", "corpus", "fixtures", "schema"):
        base = root / directory
        for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
            hashes[path.relative_to(root).as_posix()] = _digest(path.read_bytes())
    return hashes


def assert_protected_inputs_unchanged(root: Path, before: Mapping[str, str]) -> None:
    """Raise a path-only diagnostic for any protected additions, removals, or changes."""
    after = protected_input_hashes(root)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if added or removed or changed:
        raise RuntimeError(f"protected inputs changed: added={added}, removed={removed}, changed={changed}")


@dataclass(frozen=True)
class ProjectProfile:
    name: str
    descriptor_path: str
    descriptor_sha256: str
    profile_version: str
    alias_catalog: AliasCatalog
    scope: Scope | None = None
    applications: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def aliases(self) -> Mapping[str, str]:
        """Compatibility projection of the approved catalog for older semantic callers."""
        return MappingProxyType({entry.alias_value: entry.canonical_key for entry in self.alias_catalog.entries})


def load_project_profile(scope: Scope, descriptor: Path, alias_catalog: AliasCatalog) -> ProjectProfile:
    if alias_catalog.scope != scope:
        raise ValueError("project profile catalog must use the requested scope")
    descriptor_hash = _digest(descriptor.read_bytes())
    name: str
    if scope == Scope("public-fda", "casgevy-potency"):
        name = "casgevy-v1"
    else:
        name = "empty-v1"
    applications = dict(trusted_applications_for_profile(name))
    version_input = canonical_json({
        "name": name,
        "descriptor_sha256": descriptor_hash,
        "applications": applications,
        "alias_catalog_version": alias_catalog.version,
    })
    return ProjectProfile(
        name=name,
        descriptor_path=str(descriptor),
        descriptor_sha256=descriptor_hash,
        profile_version=f"profile-{_digest(version_input.encode())}",
        alias_catalog=alias_catalog,
        scope=scope,
        applications=MappingProxyType(applications),
    )
