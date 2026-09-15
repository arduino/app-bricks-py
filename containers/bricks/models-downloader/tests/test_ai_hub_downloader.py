# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the AI Hub downloader's failure reporting.

``qai_hub_models`` prints its own errors (unsupported version, unknown model, ...)
with a plain ``print(e)`` before exiting 1, so the explanation lands on *stdout*
while stderr stays empty. Reporting stderr alone used to hide it and leave the user
with nothing but the command repr, so both streams are asserted here.
"""

import json
import subprocess

import pytest

from ai_hub import download_ai_hub_model


VERSION_ERROR = (
    "Version 0.62.2 is newer than the installed version (0.59.0). "
    "Upgrade the package or use -v with an older version.\n"
    "Run `qai-hub-models versions` to see all supported versions."
)

ARGV = [
    "download_ai_hub_model.py",
    "--model_type",
    "genie",
    "--model_name",
    "qwen3_vl_8b_instruct",
    "--quantization",
    "w4a16",
    "--chipset",
    "qualcomm-qcs8275",
    "--version",
    "0.62.2",
]


def _run_main(monkeypatch, fake_run):
    monkeypatch.setattr(download_ai_hub_model.sys, "argv", ARGV)
    monkeypatch.setattr(download_ai_hub_model.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exit_info:
        download_ai_hub_model.main()
    return exit_info.value.code


def _error_events(capsys):
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    return [event for event in events if event.get("event") == "error"]


def test_cli_error_on_stdout_is_reported(monkeypatch, capsys):
    """The CLI's own message reaches the caller even when stderr is empty."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output=VERSION_ERROR + "\n", stderr="")

    assert _run_main(monkeypatch, fake_run) == 1

    (error,) = _error_events(capsys)
    description = error["description"]
    assert "Version 0.62.2 is newer than the installed version (0.59.0)." in description
    assert "Upgrade the package or use -v with an older version." in description
    assert "exit status 1" in description
    # Single-line JSON events: the CLI's multi-line text is collapsed, not split.
    assert "\n" not in description


def test_cli_error_on_stderr_is_reported(monkeypatch, capsys):
    """Messages written to stderr are still reported."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(2, cmd, output="", stderr="boom: traceback\n")

    assert _run_main(monkeypatch, fake_run) == 1

    (error,) = _error_events(capsys)
    assert "boom: traceback" in error["description"]
    assert "exit status 2" in error["description"]


def test_silent_cli_failure_falls_back_to_command(monkeypatch, capsys):
    """With both streams empty there is still the command and its exit status."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="")

    assert _run_main(monkeypatch, fake_run) == 1

    (error,) = _error_events(capsys)
    assert "qai_hub_models" in error["description"]
    assert "non-zero exit status 1" in error["description"]


def test_missing_cli_is_reported(monkeypatch, capsys):
    """A missing ``qai_hub_models`` binary is an error event, not a traceback."""

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    assert _run_main(monkeypatch, fake_run) == 1

    (error,) = _error_events(capsys)
    assert "qai_hub_models" in error["description"]


def test_non_url_output_is_reported(monkeypatch, capsys):
    """A zero exit status without a URL reports what the CLI printed instead."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="No assets for this chipset\n", stderr="")

    assert _run_main(monkeypatch, fake_run) == 1

    (error,) = _error_events(capsys)
    assert "No assets for this chipset" in error["description"]
