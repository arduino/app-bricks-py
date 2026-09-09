# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the distributed image set resolver."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distributed_images import DistributedImagesError, compose_image_versions, distributed_images  # noqa: E402
from tests.scripts.test_build_levels import make_graph  # noqa: E402


# Two release groups sharing one base image, plus a private base for the ai group:
#   slim -> apps-base (bricks)
#   slim -> runner (ai)
#   qairt -> npu-runner (ai)
TREE = {
    "python-slim": {"group": "base", "base_image": True, "downstream": ["python-apps-base", "llamacpp-runner"]},
    "qairt-common-base": {"group": "base", "base_image": True, "downstream": ["llamacpp-npu-runner"]},
    "python-apps-base": {"group": "bricks", "downstream": []},
    "llamacpp-runner": {"group": "ai", "downstream": []},
    "llamacpp-npu-runner": {"group": "ai", "downstream": []},
}


def write_compose(src_dir: Path, relative: str, *images: str) -> None:
    """Write a compose file listing the given image references."""
    path = src_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"    image: ${{DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}}app-bricks/{image}\n" for image in images), encoding="utf-8")


def test_compose_versions_ignore_generated_static_copies(tmp_path):
    """Only the source compose files count, not the copies bundled under static/."""
    src = tmp_path / "src"
    write_compose(src, "app_services/llamacpp/service_compose.yaml", "llamacpp-runner:1.0.0")
    write_compose(src, "app_bricks/static/services/llamacpp/service_compose.yaml", "llamacpp-runner:dev-latest")
    assert compose_image_versions(src) == {"llamacpp-runner": {"1.0.0"}}


def test_release_covers_own_group_and_referenced_groups_with_their_bases(tmp_path):
    """A bricks release lists its own build set at the new version and the ai set at the pinned one."""
    graph = make_graph(tmp_path, TREE)
    src = tmp_path / "src"
    write_compose(src, "app_services/llamacpp/service_compose.yaml", "llamacpp-runner:1.0.0")
    write_compose(src, "app_services/llamacpp/service_compose.unoq.yaml", "llamacpp-npu-runner:1.0.0")

    assert distributed_images(graph, "bricks", "2.0.0", src) == [
        ("llamacpp-npu-runner", "1.0.0"),
        ("llamacpp-runner", "1.0.0"),
        ("python-apps-base", "2.0.0"),
        ("python-slim", "1.0.0"),
        ("python-slim", "2.0.0"),
        ("qairt-common-base", "1.0.0"),
    ]


def test_shared_base_is_listed_once_when_versions_coincide(tmp_path):
    """The same image version is never scanned twice."""
    graph = make_graph(tmp_path, TREE)
    src = tmp_path / "src"
    write_compose(src, "app_services/llamacpp/service_compose.yaml", "llamacpp-runner:2.0.0", "llamacpp-npu-runner:2.0.0")

    images = distributed_images(graph, "bricks", "2.0.0", src)
    assert images.count(("python-slim", "2.0.0")) == 1
    assert len(images) == 5


def test_pinned_only_leaves_out_the_released_group(tmp_path):
    """The released group is scanned wave by wave; only the pinned images are left to list."""
    graph = make_graph(tmp_path, TREE)
    src = tmp_path / "src"
    write_compose(src, "app_services/llamacpp/service_compose.yaml", "llamacpp-runner:1.0.0", "llamacpp-npu-runner:1.0.0")

    assert distributed_images(graph, "bricks", "2.0.0", src, pinned_only=True) == [
        ("llamacpp-npu-runner", "1.0.0"),
        ("llamacpp-runner", "1.0.0"),
        ("python-slim", "1.0.0"),
        ("qairt-common-base", "1.0.0"),
    ]


def test_unreferenced_group_is_an_error(tmp_path):
    """A group whose version cannot be derived from the compose files fails loudly."""
    graph = make_graph(tmp_path, TREE)
    with pytest.raises(DistributedImagesError, match="No compose file references"):
        distributed_images(graph, "bricks", "2.0.0", tmp_path / "src")


def test_mixed_versions_within_a_group_are_an_error(tmp_path):
    """A group is released as a unit, so its containers must be pinned to one version."""
    graph = make_graph(tmp_path, TREE)
    src = tmp_path / "src"
    write_compose(src, "app_services/llamacpp/service_compose.yaml", "llamacpp-runner:1.0.0", "llamacpp-npu-runner:1.1.0")
    with pytest.raises(DistributedImagesError, match="several versions: 1.0.0, 1.1.0"):
        distributed_images(graph, "bricks", "2.0.0", src)
