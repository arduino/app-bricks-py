# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-base ships its own build of OpenCV, the only one with the GStreamer backend the CSI cameras use.

The installs of an app must leave cv2 to it: a distribution of OpenCV in the venv of the app is imported in place of the
one of the image. uv skips the distributions listed in conf/uv-excludes.txt, so the image carries neither the wheel
nor a pin to it.
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PYTHON_BASE = Path(__file__).parents[2] / "containers" / "base" / "python-base"
EXCLUDES = PYTHON_BASE / "conf" / "uv-excludes.txt"
BUILDER, PRODUCTION = (PYTHON_BASE / "Dockerfile").read_text(encoding="utf-8").split(" AS production\n")
UV_VERSION = re.findall(r"^FROM ghcr\.io/astral-sh/uv:([\d.]+)@", BUILDER, re.MULTILINE)[0]


def test_the_opencv_wheel_does_not_reach_the_image():
    assert "uv pip install --system --no-cache-dir /opt/wheels/*.whl" in BUILDER
    assert "/opt/wheels" not in PRODUCTION
    assert "CONSTRAINT" not in PRODUCTION and "FIND_LINKS" not in PRODUCTION, "nothing in the image serves a pinned OpenCV wheel"


def test_the_image_excludes_the_opencv_distributions_only_after_installing_its_own():
    assert "COPY conf/uv-excludes.txt /etc/uv-excludes.txt\nENV UV_EXCLUDE=/etc/uv-excludes.txt\n" in PRODUCTION
    assert "UV_EXCLUDE" not in BUILDER, "the builder stage would skip the OpenCV wheel it installs, without an error"


def test_every_distribution_that_provides_cv2_is_excluded():
    names = [line for line in EXCLUDES.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert sorted(names) == ["opencv-contrib-python", "opencv-contrib-python-headless", "opencv-python", "opencv-python-headless"]


@pytest.fixture(scope="module")
def uv() -> list[str]:
    """The uv release of the image, fetched when it is not the one at hand: older ones exclude a dependency, but not a
    requirement given directly."""
    if shutil.which("uv") is None:
        pytest.skip("needs uv")
    command = ["uv", "tool", "run", "--from", f"uv=={UV_VERSION}", "uv"]
    if subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=300).stdout.split()[1:2] != [UV_VERSION]:
        pytest.skip(f"uv {UV_VERSION} is not available")
    return command


def _install(uv: list[str], tmp_path: Path, requirement: str, excludes: Path | None) -> subprocess.CompletedProcess[str]:
    """What `uv pip install <requirement>` would do in an empty environment, with the packages of tmp_path/wheels as the
    only ones that exist."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    if excludes is not None:
        env["UV_EXCLUDE"] = str(excludes)
    (tmp_path / "wheels").mkdir(exist_ok=True)
    environment = ["--python", sys.executable, "--target", str(tmp_path / "site")]
    packages = ["--no-cache", "--no-index", "--find-links", str(tmp_path / "wheels")]
    return subprocess.run(
        [*uv, "pip", "install", "--dry-run", *environment, *packages, requirement], env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize(
    "requirement",
    ["opencv-python-headless", "opencv-python-headless==4.10.0.84", "OpenCV_Python>=4.5", "opencv-contrib-python", "opencv-contrib-python-headless"],
)
def test_an_app_requirement_on_opencv_installs_nothing(uv: list[str], tmp_path: Path, requirement: str):
    looked_up = _install(uv, tmp_path, requirement, excludes=None)
    assert looked_up.returncode != 0, "without the list uv looks the package up, and here there is none to find"
    excluded = _install(uv, tmp_path, requirement, EXCLUDES)
    assert excluded.returncode == 0, excluded.stderr
    assert "Would make no changes" in excluded.stderr


def test_a_package_that_depends_on_opencv_is_installed_without_it(uv: list[str], tmp_path: Path):
    (tmp_path / "wheels").mkdir()
    with zipfile.ZipFile(tmp_path / "wheels" / "needs_cv-1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("needs_cv-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: needs-cv\nVersion: 1.0\nRequires-Dist: opencv-python>=4.5\n")
        wheel.writestr("needs_cv-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr("needs_cv-1.0.dist-info/RECORD", "")
    assert _install(uv, tmp_path, "needs-cv", excludes=None).returncode != 0, "without the list the dependency is looked up too"
    excluded = _install(uv, tmp_path, "needs-cv", EXCLUDES)
    assert excluded.returncode == 0, excluded.stderr
    assert "Would install 1 package" in excluded.stderr and "+ needs-cv==1.0" in excluded.stderr
