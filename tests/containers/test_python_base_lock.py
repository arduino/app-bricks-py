# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-apps-base installs the lock of the library on top of the one of python-base: a package both pin must be on
one version, or the upper layer installs it again and the image ships both copies."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _locked_versions(lock: Path) -> dict[str, str]:
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    return {p["name"]: p["version"] for p in packages if "registry" in p["source"]}


BASE = _locked_versions(ROOT / "containers" / "base" / "python-base" / "uv.lock")
LIBRARY = _locked_versions(ROOT / "uv.lock")


def test_python_base_locks_the_numpy_its_opencv_wheel_depends_on():
    """The OpenCV wheel is installed outside the lock: without numpy in it, the latest release would come from PyPI unpinned."""
    assert "numpy" in BASE


def test_packages_locked_by_both_are_on_one_version():
    assert {name: (version, LIBRARY[name]) for name, version in BASE.items() if name in LIBRARY and LIBRARY[name] != version} == {}
