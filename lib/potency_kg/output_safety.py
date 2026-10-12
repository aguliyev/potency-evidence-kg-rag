"""Central, symlink-aware guards for every generated output destination."""

from __future__ import annotations

from pathlib import Path


PROTECTED_ROOT_NAMES = ("configs", "corpus", "fixtures", "schema")


def _contains(directory: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(directory)
    except ValueError:
        return False
    return True


def guard_output_path(
    path: Path,
    *,
    package_root: Path | None = None,
    boundary: Path | None = None,
    label: str = "output",
    follow_leaf: bool = True,
) -> Path:
    """Resolve existing symlink parents and reject protected or escaped destinations."""
    expanded = path.expanduser()
    resolved = (
        expanded.resolve(strict=False)
        if follow_leaf
        else expanded.parent.resolve(strict=False) / expanded.name
    )
    if package_root is not None:
        root = package_root.expanduser().resolve(strict=False)
        for name in PROTECTED_ROOT_NAMES:
            protected = (root / name).resolve(strict=False)
            if _contains(protected, resolved):
                raise ValueError(f"protected output destination is beneath {name}")
    if boundary is not None:
        allowed = boundary.expanduser().resolve(strict=False)
        if not _contains(allowed, resolved):
            raise ValueError(f"{label} escapes its allowed root")
    return resolved
