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


# --------------------------------------------------------------------------- #
# size_mb on the completion event and in the info stat event
# --------------------------------------------------------------------------- #
def test_completed_download_reports_size_mb(monkeypatch, capsys, tmp_path):
    """Sized like the listing sizes the model directory, bookkeeping files excluded."""
    model_directory = "qwen3_vl_8b_instruct-genie-w4a16-qualcomm_qcs8275"
    monkeypatch.setenv("model_directory", model_directory)
    monkeypatch.setattr(download_ai_hub_model.sys, "argv", [*ARGV, "--output-dir", str(tmp_path)])
    monkeypatch.setattr(
        download_ai_hub_model.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="https://example.com/model.zip\n", stderr=""),
    )

    def _extract(_url, output_dir, _json_progress):
        path = tmp_path / model_directory / "model.bin"
        path.write_bytes(b"\0" * (2 * 1024 * 1024))

    monkeypatch.setattr(download_ai_hub_model, "download_and_extract", _extract)

    download_ai_hub_model.main()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert events[-1]["description"].startswith("Downloaded to:")
    assert events[-1]["size_mb"] == 2.0


def test_info_reports_null_size_for_an_undeclared_model(monkeypatch, capsys, tmp_path):
    from ai_hub import ai_hub_model_info

    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text("models: []\n")
    monkeypatch.setattr(
        ai_hub_model_info.sys, "argv", ["ai_hub_model_info.py", "--model-type", "genie", "--model-name", "absent", "--model-list", str(yaml_path)]
    )
    ai_hub_model_info.main()
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "stat"
    assert event["size_mb"] is None
    assert event["size_bytes"] is None
