# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""run.sh of python-apps-base creates the app venv only when the app adds something to the image; `uv venv` costs 0.4 s on the board.

It also serves the bricks installed from the bricks registry, which arduino-app-cli mounts in /opt/third-party-bricks:
their requirements are installed like the ones of the custom bricks and their folders are added to PYTHONPATH.
"""

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
case "$*" in *"$FAKE_UV_FAILS_ON"*) [ -n "$FAKE_UV_FAILS_ON" ] && exit 1 ;; esac
exit 0
""",
    "python": """#!/bin/sh
echo "python $* VIRTUAL_ENV=${VIRTUAL_ENV:-none}" >> "$FAKE_LOG"
printf '%s' "${PYTHONPATH:-}" > "$FAKE_LOG.pythonpath"
""",
    "streamlit": """#!/bin/sh
echo "streamlit $*" >> "$FAKE_LOG"
""",
}

WEB_APP_YAML = "name: t\nbricks:\n- arduino:web_ui: {}\n"
STREAMLIT_APP_YAML = "name: t\nbricks:\n- arduino:streamlit_ui: {}\n"


def _third_party_dir(tmp_path: Path) -> Path:
    """Where the test stands in for /opt/third-party-bricks; it exists only when a test creates it."""
    return tmp_path / "third-party-bricks"


def _run(app: Path, tmp_path: Path, command: str | None = None, uv_fails_on: str = "", expect_failure: bool = False) -> list[str]:
    """Run run.sh against an app folder with the tools replaced by loggers; return the commands they saw."""
    # The script serves /app, where the container mounts the app: a copy pointed at the test folder stands in
    script = RUN_SH.read_text().replace("\r\n", "\n")
    assert script.count('BASE_DIR="/app"\n') == 1, "run.sh no longer sets BASE_DIR the way this test rewrites it"
    run_sh = tmp_path / "run.sh"
    third_party = 'THIRD_PARTY_BRICKS_DIR="/opt/third-party-bricks"\n'
    assert script.count(third_party) == 1, "run.sh no longer sets THIRD_PARTY_BRICKS_DIR the way this test rewrites it"
    script = script.replace(third_party, f'THIRD_PARTY_BRICKS_DIR="{_third_party_dir(tmp_path)}"\n')
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
    # Without the venv pytest itself runs in (`uv run` exports it): in the container nothing activates one,
    # and PYTHONPATH is the one run.sh builds
    env = {
        **{k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")},
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "FAKE_LOG": str(log),
        "FAKE_UV_FAILS_ON": uv_fails_on,
    }
    result = subprocess.run(["sh", str(run_sh), *([command] if command else [])], env=env, capture_output=True, text=True, timeout=120)
    if expect_failure:
        assert result.returncode != 0, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    return log.read_text().splitlines()


def _python_path(tmp_path: Path) -> list[str]:
    """The PYTHONPATH the app was started with, as run.sh built it."""
    value = (tmp_path / "calls.log.pythonpath").read_text()
    return value.split(":") if value else []


def _third_party_brick(tmp_path: Path, mount: str, name: str, requirements: str | None = None) -> Path:
    """A brick as arduino-app-cli mounts it: <third-party dir>/<mount>/<brick name>/, the folder of the Python module."""
    brick = _third_party_dir(tmp_path) / mount / name
    brick.mkdir(parents=True)
    (brick / "__init__.py").write_text(f"NAME = '{name}'\n")
    (brick / "brick_config.yaml").write_text(f"id: acme:{name}\n")
    if requirements is not None:
        (brick / "requirements.txt").write_text(requirements)
    return brick


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


def test_without_third_party_bricks_the_python_path_has_only_the_custom_bricks(tmp_path: Path):
    app = _app(tmp_path, {"bricks/mine/__init__.py": ""})
    _run(app, tmp_path)
    assert _python_path(tmp_path) == [f"{app}/bricks"]


def test_third_party_bricks_are_importable_by_name_after_the_custom_ones(tmp_path: Path):
    app = _app(tmp_path, {"bricks/mine/__init__.py": ""})
    _third_party_brick(tmp_path, "a1b2c3", "camera")
    _third_party_brick(tmp_path, "d4e5f6", "soil_sensor")
    calls = _run(app, tmp_path)
    third_party = _third_party_dir(tmp_path)
    assert _python_path(tmp_path) == [f"{app}/bricks", f"{third_party}/a1b2c3", f"{third_party}/d4e5f6"]
    assert not any(c.startswith(("uv venv", "uv pip")) for c in calls), "bricks without requirements add nothing to install"
    # The path run.sh built resolves the bricks by their name alone, with no namespace
    imported = subprocess.run(
        [sys.executable, "-c", "import camera, soil_sensor; print(camera.NAME, soil_sensor.NAME)"],
        env={**os.environ, "PYTHONPATH": ":".join(_python_path(tmp_path))},
        capture_output=True,
        text=True,
    )
    assert imported.stdout.split() == ["camera", "soil_sensor"], imported.stderr


def test_third_party_bricks_alone_are_on_the_python_path(tmp_path: Path):
    app = _app(tmp_path)
    _third_party_brick(tmp_path, "a1b2c3", "camera")
    calls = _run(app, tmp_path)
    assert _python_path(tmp_path) == [f"{_third_party_dir(tmp_path)}/a1b2c3"]
    assert not any(c.startswith("uv") for c in calls), "a brick without requirements costs no uv call at start"


def test_third_party_brick_requirements_get_a_venv_and_are_installed_after_the_custom_ones(tmp_path: Path):
    app = _app(tmp_path, {"bricks/mine/requirements.txt": "numpy\n"})
    brick = _third_party_brick(tmp_path, "a1b2c3", "camera", "paho-mqtt==2.2.0\n")
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls
    installs = [c for c in calls if c.startswith("uv pip install")]
    assert installs == [
        f"uv pip install --compile-bytecode -r {app}/bricks/mine/requirements.txt",
        f"uv pip install --compile-bytecode -r {brick}/requirements.txt",
    ]
    assert calls[-1].endswith(f"VIRTUAL_ENV={app}/.cache/.venv")


def test_a_third_party_brick_with_requirements_alone_gets_a_venv(tmp_path: Path):
    app = _app(tmp_path)
    _third_party_brick(tmp_path, "a1b2c3", "camera", "paho-mqtt==2.2.0\n")
    calls = _run(app, tmp_path)
    assert _venv_created(app, calls), calls


def test_third_party_brick_requirements_are_installed_once_whatever_the_mount_folder(tmp_path: Path):
    app = _app(tmp_path)
    brick = _third_party_brick(tmp_path, "a1b2c3", "camera", "paho-mqtt==2.2.0\n")
    _run(app, tmp_path)
    recorded = app / ".cache" / "third-party-bricks" / "camera" / "installed_requirements.txt"
    assert recorded.read_text() == "paho-mqtt==2.2.0\n"
    # arduino-app-cli may mount the same brick under another folder name at the next start
    brick.parent.rename(brick.parent.with_name("z9y8x7"))
    calls = _run(app, tmp_path)
    assert not any(c.startswith("uv pip install") for c in calls), calls


def test_updated_third_party_brick_requirements_are_installed_again(tmp_path: Path):
    app = _app(tmp_path)
    brick = _third_party_brick(tmp_path, "a1b2c3", "camera", "paho-mqtt==2.2.0\n")
    _run(app, tmp_path)
    (brick / "requirements.txt").write_text("paho-mqtt==2.3.0\n")
    calls = _run(app, tmp_path)
    assert f"uv pip install --compile-bytecode -r {brick}/requirements.txt" in calls
    recorded = app / ".cache" / "third-party-bricks" / "camera" / "installed_requirements.txt"
    assert recorded.read_text() == "paho-mqtt==2.3.0\n"


def test_a_custom_and_a_third_party_brick_with_the_same_name_keep_separate_records(tmp_path: Path):
    app = _app(tmp_path, {"bricks/camera/requirements.txt": "numpy\n"})
    _third_party_brick(tmp_path, "a1b2c3", "camera", "paho-mqtt==2.2.0\n")
    _run(app, tmp_path)
    assert (app / ".cache" / "camera" / "installed_requirements.txt").read_text() == "numpy\n"
    assert (app / ".cache" / "third-party-bricks" / "camera" / "installed_requirements.txt").read_text() == "paho-mqtt==2.2.0\n"


def test_prepare_fails_when_third_party_brick_requirements_cannot_be_installed(tmp_path: Path):
    app = _app(tmp_path)
    brick = _third_party_brick(tmp_path, "a1b2c3", "camera", "no-such-package==9.9.9\n")
    _run(app, tmp_path, command="prepare", uv_fails_on=f"-r {brick}/requirements.txt", expect_failure=True)
    assert not (app / ".cache" / "third-party-bricks" / "camera" / "installed_requirements.txt").exists()
