# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``common/model_size.py`` — the one ``size_mb`` every event reports."""

import json

from common.model_size import exists_event, main, path_size_bytes, paths_size_bytes, paths_size_mb, size_mb

MIB = 1024 * 1024


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def test_size_mb_is_mib_rounded_to_two_decimals():
    assert size_mb(MIB) == 1.0
    assert size_mb(1536 * 1024) == 1.5
    assert size_mb(0) == 0.0
    assert size_mb(None) is None


def test_path_size_bytes_skips_bookkeeping_files(tmp_path):
    """A directory measures the same before and after its record is written."""
    _write(tmp_path / "model.eim", MIB)
    _write(tmp_path / "sub" / "part.bin", MIB)
    before = path_size_bytes(str(tmp_path))
    _write(tmp_path / ".arduino_metadata.yaml", 500)
    _write(tmp_path / ".arduino_metadata.yaml.tmp", 500)
    _write(tmp_path / ".download", 100)
    assert before == path_size_bytes(str(tmp_path)) == 2 * MIB


def test_path_size_bytes_of_a_file_and_of_nothing(tmp_path):
    _write(tmp_path / "a.gguf", 1234)
    assert path_size_bytes(str(tmp_path / "a.gguf")) == 1234
    assert path_size_bytes(str(tmp_path / "missing")) is None


def test_paths_are_summed_in_bytes_and_rounded_once(tmp_path):
    """Rounding per file would drift from the byte total: 3 x 0.004 MiB is 0.01, not 0."""
    paths = []
    for name in ("a", "b", "c"):
        _write(tmp_path / name, 4 * 1024)  # 0.0039 MiB each
        paths.append(str(tmp_path / name))
    assert paths_size_mb(paths) == 0.01


def test_paths_size_bytes_unknown_when_empty_or_missing(tmp_path):
    _write(tmp_path / "a", 10)
    assert paths_size_bytes([]) is None
    assert paths_size_bytes([str(tmp_path / "a"), str(tmp_path / "gone")]) is None


def test_exists_event_carries_size_mb(tmp_path):
    _write(tmp_path / "model.eim", MIB)
    assert exists_event("Model exists: m", str(tmp_path)) == {"event": "info", "description": "Model exists: m", "size_mb": 1.0}
    assert exists_event("Model exists: m", str(tmp_path), downloading=False)["downloading"] is False


def test_cli_prints_the_exists_event(tmp_path, capsys):
    """What the AI Hub and Edge Impulse shell scripts print for an installed model."""
    _write(tmp_path / "model.eim", 2 * MIB)
    main(["--description", "Model exists: model.eim", "--downloading", "false", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {
        "event": "info",
        "description": "Model exists: model.eim",
        "downloading": False,
        "size_mb": 2.0,
    }
