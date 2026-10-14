# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""`.licensed.yml` must stay in step with the container tree it scans.

The dependency license scan runs in a container (`scripts/licensed/`), so nothing in the
normal test run exercises `.licensed.yml`. That makes it easy for a container rename or
move to leave an app pointing at a requirements file that no longer exists, or at a cache
directory under `.licenses/` that no longer matches its name - and the failure only shows
up in the license workflow, long after the refactor.

These tests are the cheap half of `check_requirements_covered` in `scripts/licensed/run.py`:
every declared path resolves, and app names and cache directories are the same set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / ".licensed.yml").read_text(encoding="utf-8"))
APPS = CONFIG["apps"]
APP_IDS = [app["name"] for app in APPS]


@pytest.mark.parametrize("app", APPS, ids=APP_IDS)
def test_each_app_declares_exactly_one_venv_source(app: dict):
    """`run.py` builds the venv from either a requirements file or the project, not both."""
    venv = app["venv"]
    assert ("requirements" in venv) != ("project" in venv), (
        f"{app['name']}: venv needs exactly one of 'requirements' or 'project', got {sorted(venv)}"
    )


@pytest.mark.parametrize("app", APPS, ids=APP_IDS)
def test_each_declared_requirements_file_exists(app: dict):
    """A dangling path fails the scan at venv-build time, not at config-load time."""
    if "requirements" not in app["venv"]:
        pytest.skip("app builds its venv from the project, not a requirements file")
    requirements = REPO_ROOT / app["venv"]["requirements"]
    assert requirements.is_file(), f"{app['name']}: {app['venv']['requirements']} does not exist"


def test_app_names_and_license_caches_are_the_same_set():
    """`cache_path` is `.licenses/`, and licensed keys the cache by app name: a renamed app
    orphans its old directory and is reported as missing every record it already had."""
    declared = {app["name"] for app in APPS}
    cached = {path.name for path in (REPO_ROOT / CONFIG["cache_path"]).iterdir() if path.is_dir()}
    assert declared == cached, (
        f"apps without a license cache: {sorted(declared - cached)}; cache directories without an app: {sorted(cached - declared)}"
    )
