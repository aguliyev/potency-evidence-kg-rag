from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from potency_kg.project_profiles import protected_input_hashes


ROOT = Path(__file__).resolve().parents[1]


def test_runner_is_executable_and_protected_inputs_are_snapshotted() -> None:
    """Removing a protected root from the snapshot must break this test."""
    assert stat.S_IMODE((ROOT / "bin/runner").stat().st_mode) & 0o111
    hashes = protected_input_hashes(ROOT)
    expected = {
        path.relative_to(ROOT).as_posix()
        for directory in ("configs", "corpus", "fixtures", "schema")
        for path in (ROOT / directory).rglob("*")
        if path.is_file()
    }
    assert set(hashes) == expected


def test_compose_protects_every_supplied_input_directory() -> None:
    """Making one input mount writable must break this test."""
    compose = (ROOT / "docker/compose.yaml").read_text(encoding="utf-8")
    for directory in ("configs", "corpus", "fixtures", "schema"):
        assert compose.count(f"../{directory}:/workspace/{directory}:ro") == 3


def test_protected_mounts_reject_write_attempts_without_creating_a_probe() -> None:
    """Making a protected overlay writable must fail before the probe can be created."""
    for directory in ("configs", "corpus", "fixtures", "schema"):
        protected = ROOT / directory
        assert os.statvfs(protected).f_flag & os.ST_RDONLY
        probe = protected / ".stage-1-write-probe"
        with pytest.raises(OSError) as error:
            probe.write_bytes(b"probe")
        assert error.value.errno == errno.EROFS
        assert not probe.exists()


def test_structural_modules_import_without_model_environment() -> None:
    """Reading model credentials during structural imports must break this test."""
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "lib"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import potency_kg.contracts, potency_kg.db, potency_kg.graph_store, potency_kg.ingest",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
