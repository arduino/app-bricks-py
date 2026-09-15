# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the SBOM delta base image resolution and CLI argument handling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container_deps import Containers  # noqa: E402
from scripts.sbom_delta import SbomDeltaError, extract_attested_sbom, parse_container_spec, resolve_runtime_base  # noqa: E402
from tests.scripts.test_container_deps import TREE, make_containers_dir  # noqa: E402


def test_runtime_base_of_a_derived_container_is_its_parent_at_the_scanned_version(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    assert resolve_runtime_base(containers, "python-apps-base", "ghcr.io/arduino", "1.2.3") == "ghcr.io/arduino/app-bricks/python-base:1.2.3"


def test_runtime_base_of_a_root_container_is_its_external_image(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    assert resolve_runtime_base(containers, "ei-models-runner", "ghcr.io/arduino/", "1.2.3") == "public.ecr.aws/g7a8t7v6/inference-container:v1.92.3"


def test_runtime_base_of_an_unknown_container_is_rejected(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    with pytest.raises(SbomDeltaError, match="Unknown container"):
        resolve_runtime_base(containers, "ghost", "ghcr.io/arduino/", "1.2.3")


def test_spec_without_version_uses_default():
    assert parse_container_spec("python-slim", "1.2.3") == ("python-slim", "1.2.3")


def test_spec_with_version_overrides_default():
    assert parse_container_spec("python-slim:0.9.0", "1.2.3") == ("python-slim", "0.9.0")


def test_spec_with_empty_version_falls_back_to_default():
    assert parse_container_spec("python-slim:", "1.2.3") == ("python-slim", "1.2.3")


def test_spec_without_name_is_rejected():
    with pytest.raises(SbomDeltaError, match="name\\[:version\\]"):
        parse_container_spec(":1.2.3", "1.2.3")


SPDX = {"SPDXID": "SPDXRef-DOCUMENT", "packages": []}


def test_attested_sbom_of_a_single_platform_image():
    assert extract_attested_sbom("img", {"SPDX": SPDX}) == SPDX


def test_attested_sbom_keyed_by_platform():
    assert extract_attested_sbom("img", {"linux/arm64": {"SPDX": SPDX}}) == SPDX


def test_attested_sbom_requires_exactly_one_platform():
    with pytest.raises(SbomDeltaError, match="expected one platform"):
        extract_attested_sbom("img", {"linux/arm64": {"SPDX": SPDX}, "linux/amd64": {"SPDX": SPDX}})


def test_image_without_attestation_is_rejected():
    with pytest.raises(SbomDeltaError, match="no SBOM attestation"):
        extract_attested_sbom("img", None)
