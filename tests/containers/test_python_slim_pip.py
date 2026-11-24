# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-slim carries no pip: uv installs the packages of the images built on it, and a `pip install` run by hand
would replace what they ship, the OpenCV build of python-base included.

pip comes with the upstream Python image, so every stage that starts from it and hands /usr/local to an image has to
uninstall it: python-slim before compiling the bytecode, the builder of python-base before its directories are copied.
"""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

BASE = Path(__file__).parents[2] / "containers" / "base"
PYTHON_SLIM = (BASE / "python-slim" / "Dockerfile").read_text(encoding="utf-8")
PYTHON_BASE_BUILDER = (BASE / "python-base" / "Dockerfile").read_text(encoding="utf-8").split(" AS production\n")[0]

UNINSTALL = "python -m pip uninstall -y --root-user-action ignore pip"


@pytest.mark.parametrize("stage", [PYTHON_SLIM, PYTHON_BASE_BUILDER], ids=["python-slim", "python-base builder"])
def test_the_stage_uninstalls_pip(stage: str):
    assert stage.count(f"    {UNINSTALL}; \\\n") == 1


def test_python_slim_uninstalls_pip_before_compiling_its_bytecode():
    assert PYTHON_SLIM.index(UNINSTALL) < PYTHON_SLIM.index("python -m compileall")


def test_the_builder_uninstalls_pip_after_its_last_install():
    uninstall = PYTHON_BASE_BUILDER.index(UNINSTALL)
    assert uninstall > PYTHON_BASE_BUILDER.rindex("uv pip install")
    assert "pip" not in PYTHON_BASE_BUILDER[uninstall + len(UNINSTALL) :], "nothing puts it back before the stage ends"


@pytest.mark.skipif(sys.platform != "linux", reason="the layout of the image: bin/pip, bin/pip3, bin/pip3.x")
def test_the_uninstall_leaves_no_file_of_pip_and_fails_without_it(tmp_path: Path):
    venv.create(tmp_path, with_pip=True)
    python = str(tmp_path / "bin" / "python")
    command = [python, *UNINSTALL.split()[1:]]

    def files_of_pip() -> list[str]:
        return sorted(p.name for folder in ("bin", "lib/python*/site-packages") for p in tmp_path.glob(f"{folder}/pip*"))

    assert {"pip", "pip3", f"pip3.{sys.version_info.minor}"} < set(files_of_pip())
    assert subprocess.run(command, capture_output=True, text=True, timeout=120).returncode == 0
    assert files_of_pip() == [], "the scripts go too, the ones the record of the package does not list included"
    assert subprocess.run(command, capture_output=True, text=True, timeout=120).returncode != 0, "a silent pass would hide a pip that moved"
