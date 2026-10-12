"""Trusted project defaults and immutable supplied-input snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .contracts import Scope


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


def _scope_parts(scope: Any) -> tuple[str, str]:
    if isinstance(scope, tuple):
        return scope
    return scope.tenant_id, scope.project_id


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
    aliases: Mapping[str, str]
    descriptor_path: str
    descriptor_sha256: str
    profile_version: str
    scope: Scope | None = None
    applications: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


def load_project_profile(scope: Any, descriptor: Path) -> ProjectProfile:
    tenant_id, project_id = _scope_parts(scope)
    descriptor_hash = _digest(descriptor.read_bytes())
    aliases: dict[str, str]
    name: str
    if (tenant_id, project_id) == ("public-fda", "casgevy-potency"):
        name = "casgevy-v1"
        aliases = {
            "CASGEVY": "casgevy",
            "exagamglogene autotemcel": "casgevy",
            "exa-cel": "casgevy",
        }
    else:
        name, aliases = "empty-v1", {}
    applications = dict(trusted_applications_for_profile(name))
    version_input = json.dumps(
        {
            "name": name,
            "descriptor_sha256": descriptor_hash,
            "aliases": aliases,
            "applications": applications,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return ProjectProfile(
        name=name,
        aliases=MappingProxyType(aliases),
        descriptor_path=str(descriptor),
        descriptor_sha256=descriptor_hash,
        profile_version=f"profile-{_digest(version_input.encode())}",
        scope=Scope(tenant_id, project_id),
        applications=MappingProxyType(applications),
    )
