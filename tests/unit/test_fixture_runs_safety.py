from __future__ import annotations

import json
from pathlib import Path

import pytest

from potency_kg.contracts import Scope
from potency_kg.fixture_catalog import FixtureRunConfiguration
from potency_kg.fixture_runs import _write_import_inputs


def _configuration(records_path: Path) -> FixtureRunConfiguration:
    return FixtureRunConfiguration(
        scenario_id="scenario",
        stage="stage",
        scope=Scope("tenant", "project"),
        corpus_snapshot="snapshot",
        input_records=(records_path,),
        replacements=(),
        operation="load",
        question=None,
    )


def test_fixture_raw_filenames_do_not_interpolate_malicious_source_identity(tmp_path: Path) -> None:
    """Source IDs and versions containing path syntax must remain inert manifest data."""
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "source_id": "../../escaped/source",
                "source_version": "/absolute/../../../version",
                "passage_id": "p1",
                "title": "title",
                "text": "fixture evidence",
                "status": "final",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    descriptor = _write_import_inputs(_configuration(records), output)

    manifest = json.loads((output / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    filename = manifest["sources"][0]["filename"]
    assert descriptor.is_relative_to(output)
    assert filename.count("/") == 3
    assert ".." not in filename
    assert len(list((output / "corpus" / "raw" / "fixture-inputs").glob("*.jsonl"))) == 1
    assert not (tmp_path / "escaped").exists()


def test_fixture_writer_rejects_symlink_parent_before_external_write(tmp_path: Path) -> None:
    """A symlinked generated-data parent must not redirect any fixture bytes outside its run root."""
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "source_id": "source",
                "source_version": "v1",
                "passage_id": "p1",
                "title": "title",
                "text": "fixture evidence",
                "status": "final",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "corpus").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="fixture output"):
        _write_import_inputs(_configuration(records), output)

    assert list(outside.iterdir()) == []
