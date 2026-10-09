# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""run.sh of python-apps-base creates the app venv only when the app adds something to the image; `uv venv` costs 0.4 s on the board."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUN_SH = Path(__file__).parents[2] / "containers" / "bricks" / "python-apps-base" / "scripts" / "run.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None, reason="needs a POSIX sh")

# The tools run.sh calls, replaced by loggers: `uv venv` leaves an activate script behind like the real one,
# `uv pip show` answers "not installed" so that the streamlit install path is taken.
FAKE_TOOLS = {
    "uv": """#!/bin/sh
echo "uv $*" >> "$FAKE_LOG"
if [ "$1" = venv ]; then mkdir -p "$2/bin" && printf 'export VIRTUAL_ENV=%s\\n' "$2" > "$2/bin/activate"; fi
if [ "$1" = pip ] && [ "$2" = show ]; then exit 1; fi
""",
    "python": """#!/bin/sh
echo "python $* VIRTUAL_ENV=${VIRTUAL_ENV:-none}" >> "$FAKE_LOG"
""",
    "streamlit": """#!/bin/sh
echo "streamlit $*" >> "$FAKE_LOG"
""",
}

WEB_APP_YAML = "name: t\nbricks:\n- arduino:web_ui: {}\n"
STREAMLIT_APP_YAML = "name: t\nbricks:\n- arduino:streamlit_ui: {}\n"


def _run(app: Path, tmp_path: Path) -> list[str]:
    """Run run.sh against an app folder with the tools replaced by loggers; return the commands they saw."""
    # The script serves /app, where the container mounts the app: a copy pointed at the test folder stands in
    script = RUN_SH.read_text().replace("\r\n", "\n")
    assert script.count('BASE_DIR="/app"\n') == 1, "run.sh no longer sets BASE_DIR the way this test rewrites it"
    run_sh = tmp_path / "run.sh"
    run_sh.write_text(script.replace('BASE_DIR="/app"\n', f'BASE_DIR="{app}"\n'))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in FAKE_TOOLS.items():
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".asoundrc").write_text("# baked\n")  # As in the image: no ALSA provisioning at start
    log = tmp_path / "calls.log"
    log.write_text("")
    # Without the venv pytest itself runs in (`uv run` exports it): in the container nothing activates one
    env = {
        **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "FAKE_LOG": str(log),
    }
    result = subprocess.run(["sh", str(run_sh)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return log.read_text().splitlines()


def _app(tmp_path: Path, files: dict[str, str] | None = None, app_yaml: str = WEB_APP_YAML) -> Path:
    app = tmp_path / "app"
    (app / "python").mkdir(parents=True)
    (app / "python" / "main.py").write_text("print('hi')\n")
    (app / "app.yaml").write_text(app_yaml)
    for relative, content in (files or {}).items():
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return app


def _venv_created(app: Path, calls: list[str]) -> bool:
    return f"uv venv {app}/.cache/.venv --system-site-packages" in calls


def test_an_app_that_adds_nothing_runs_on_the_system_interpreter(tmp_path: Path):
    app = _app(tmp_path)
    calls = _run(app, tmp_path)
    assert not any(c.startswith("uv") for c in calls), calls
    assert calls[-1] == f"python {app}/python/main.py VIRTUAL_ENV=none"
    assert not (app / ".cache" / ".venv").exists()


def test_requirements_get_a_venv_and_are_installed_in_it(tmp_path: Path):
    app = _app(tmp_path, {"python/requirements.txt": "requests\n"})
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls
    assert any(c.startswith("uv pip install") and c.endswith("python/requirements.txt") for c in calls), calls
    assert calls[-1].endswith(f"VIRTUAL_ENV={app}/.cache/.venv")


def test_a_requirements_file_with_only_blank_lines_needs_no_venv(tmp_path: Path):
    app = _app(tmp_path, {"python/requirements.txt": "\n  \n"})
    calls = _run(app, tmp_path)
    assert not _venv_created(app, calls), calls
    assert (app / ".cache" / "installed_requirements.txt").exists(), "the empty file is still recorded as installed"


def test_custom_brick_requirements_get_a_venv(tmp_path: Path):
    app = _app(tmp_path, {"bricks/mine/requirements.txt": "numpy\n"})
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls


def test_python_libraries_get_a_venv(tmp_path: Path):
    app = _app(tmp_path)
    (app / "python-libraries").mkdir()
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls


def test_an_existing_venv_is_kept_and_used(tmp_path: Path):
    app = _app(tmp_path)
    venv = app / ".cache" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f"export VIRTUAL_ENV={venv}\n")
    calls = _run(app, tmp_path)
    assert not _venv_created(app, calls), calls
    assert calls[-1].endswith(f"VIRTUAL_ENV={venv}")


def test_streamlit_apps_get_a_venv(tmp_path: Path):
    app = _app(tmp_path, app_yaml=STREAMLIT_APP_YAML)
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls
    assert any("uv pip install" in c and "streamlit" in c for c in calls), calls
    assert calls[-1].startswith("streamlit run")
