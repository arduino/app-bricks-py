# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The pyright rules file shipped in the wheel must stay readable by every consumer.

The file is maintained at the repository root and copied into
arduino/app_bricks/static/ by the build backend, next to the other generated
assets.

App Lab and the CI checks of both repositories load it with the constraints
below; a file they cannot parse makes them fall back to their defaults, so the
mistake would go unnoticed. Keep this test in sync with those loaders.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPO_ROOT / "pyright-rules.json"

SCHEMA_VERSION = 1
PROFILES = {"app-bricks-py", "api-user"}
TYPE_CHECKING_MODES = {"off", "basic", "standard", "strict"}
RULE_SEVERITIES = {"none", "information", "warning", "error"}


@pytest.fixture(scope="module")
def rules() -> dict:
    return json.loads(RULES_PATH.read_text())


def test_schema_version_is_the_one_consumers_understand(rules):
    assert rules["schemaVersion"] == SCHEMA_VERSION


def test_engine_settings_are_well_formed(rules):
    assert re.fullmatch(r"\d+\.\d+", rules["pythonVersion"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", rules["pyrightVersion"])
    assert rules["useLibraryCodeForTypes"] is True


def test_exactly_the_agreed_profiles_are_defined(rules):
    assert set(rules["profiles"]) == PROFILES


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_profile_holds_a_mode_and_report_rules_only(rules, profile):
    entry = rules["profiles"][profile]
    assert entry["typeCheckingMode"] in TYPE_CHECKING_MODES
    for name, value in entry["rules"].items():
        assert name.startswith("report"), f"{profile}: {name} is not a report* rule"
        assert isinstance(value, bool) or value in RULE_SEVERITIES, f"{profile}: {name} has an unsupported value {value!r}"


def test_the_file_is_shipped_in_the_wheel(tmp_path, monkeypatch):
    # arduino-bricks-release copies the file into the static assets, and the wheel
    # picks static files up through the package-data patterns.
    import tomllib

    from arduino.app_tools.module_listing import save_pyright_rules

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "*.json" in pyproject["tool"]["setuptools"]["package-data"]["*"]

    monkeypatch.chdir(REPO_ROOT)
    save_pyright_rules("pyright-rules.json", str(tmp_path))
    assert json.loads((tmp_path / "pyright-rules.json").read_text()) == json.loads(RULES_PATH.read_text())
