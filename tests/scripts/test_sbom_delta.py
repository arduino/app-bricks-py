# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the SBOM delta CLI argument handling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sbom_delta import SbomDeltaError, parse_container_spec  # noqa: E402


def test_spec_without_version_uses_default():
    assert parse_container_spec("python-slim", "1.2.3") == ("python-slim", "1.2.3")


def test_spec_with_version_overrides_default():
    assert parse_container_spec("python-slim:0.9.0", "1.2.3") == ("python-slim", "0.9.0")


def test_spec_with_empty_version_falls_back_to_default():
    assert parse_container_spec("python-slim:", "1.2.3") == ("python-slim", "1.2.3")


def test_spec_without_name_is_rejected():
    with pytest.raises(SbomDeltaError, match="name\\[:version\\]"):
        parse_container_spec(":1.2.3", "1.2.3")
