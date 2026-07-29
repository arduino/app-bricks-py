# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the JSON progress reporting of the Hugging Face downloader.

The downloader replaces huggingface_hub's terminal progress bars with a stream of JSON
events, which only works as long as ``JsonProgress`` keeps satisfying the hooks
huggingface_hub looks for. A library upgrade already broke this once, so the contract is
tested explicitly here.
"""

import json

import pytest

from hugging_face.hf_downloader import JsonProgress, matches_pattern


def read_events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]


def test_emits_start_and_completion_events(capsys):
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(100)
    bar.close()

    events = read_events(capsys)
    # No "update" in between: it is throttled to one event per EMIT_INTERVAL.
    assert [e["event"] for e in events] == ["start", "complete"]
    assert events[-1] == {
        "event": "complete",
        "description": "model.gguf",
        "current": 100,
        "total": 100,
        "unit": "B",
        "percentage": "100.0%",
    }


def test_emits_throttled_update_events(monkeypatch, capsys):
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(25)
    bar.update(25)

    events = read_events(capsys)
    assert [e["event"] for e in events] == ["start", "update", "update"]
    assert [e["percentage"] for e in events] == ["0.0%", "25.0%", "50.0%"]


def test_incomplete_transfer_does_not_report_completion(capsys):
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(40)
    bar.close()

    assert [e["event"] for e in read_events(capsys)] == ["start"]


@pytest.mark.parametrize(
    "desc",
    ["model.gguf: reconstructing file", "model.gguf: downloading bytes", "model.gguf: "],
)
def test_description_drops_progress_bar_decorations(desc, capsys):
    JsonProgress(desc=desc, total=100, unit="B")

    assert read_events(capsys)[0]["description"] == "model.gguf"


def test_network_bytes_drive_progress_while_disk_writes_lag(monkeypatch, capsys):
    """Xet flushes to disk in bursts, so progress must follow the network counter.

    Reporting only bytes written to disk leaves the percentage frozen at 0% for tens of MB.
    """
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    read_events(capsys)  # drop the "start" event

    bar.update_transfer(40)
    bar.set_transfer_postfix_str("1.00MB/s")

    assert bar.n == 0  # nothing written to disk yet
    assert read_events(capsys) == [
        {
            "event": "update",
            "description": "model.gguf",
            "current": 40,
            "total": 100,
            "unit": "B",
            "percentage": "40.0%",
        }
    ]


def test_progress_never_exceeds_the_file_size(monkeypatch, capsys):
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    read_events(capsys)

    bar.update_transfer(150)

    assert read_events(capsys)[-1]["percentage"] == "100.0%"


def test_completion_follows_disk_writes_not_the_network(monkeypatch, capsys):
    """Cached Xet chunks make network bytes end below the file size: they cannot signal the end."""
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update_transfer(60)
    bar.update(100)  # all bytes flushed to disk
    bar.close()

    events = read_events(capsys)
    assert events[-1]["event"] == "complete"
    assert events[-1]["current"] == 100


def test_xet_download_routes_both_counters_into_a_single_bar():
    """The Xet downloader must not fall back to its own bar for network transfer.

    huggingface_hub reports Xet downloads with two bars and honours ``tqdm_class`` for the
    reconstruction one only -- unless the class also implements ``update_transfer``, in
    which case both counters share one object. Without that hook a real progress bar is
    printed to the terminal alongside the JSON events.
    """
    reporting = pytest.importorskip("huggingface_hub.utils._xet_progress_reporting")
    if not hasattr(reporting, "XetDownloadProgressReporter"):
        pytest.skip("huggingface_hub predates the dual-bar Xet reporter")

    reporter = reporting.XetDownloadProgressReporter(
        reconstruction_desc="model.gguf: reconstructing file",
        transfer_desc="model.gguf: downloading bytes",
        total=100,
        log_level=20,
        tqdm_class=JsonProgress,
    )
    try:
        assert isinstance(reporter.reconstruction_bar, JsonProgress)
        assert reporter.transfer_bar is reporter.reconstruction_bar
    finally:
        reporter.close()


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("model-Q4_0.gguf", "*Q4_0*.gguf", True),
        ("UD-Q4_0/model-Q4_0.gguf", "*Q4_0*.gguf", True),  # nested per-quantization folder
        ("mmproj-F16.gguf", "*mmproj*", True),
        ("model-Q4_0.gguf", "*Q8_0*.gguf", False),
    ],
)
def test_matches_pattern(path, pattern, expected):
    assert matches_pattern(path, pattern) is expected
