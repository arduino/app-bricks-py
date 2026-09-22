# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The container dependency graph declared in ci.json must match the Dockerfiles.

CI orders the builds from the `downstream` lists in `containers/*/*/ci.json` alone; the
Dockerfiles are what actually pull the images. Nothing links the two, so a Dockerfile that
starts to build `FROM` (or `COPY --from`) another container in this repo without the parent
listing it under `downstream` builds fine locally and then picks up a stale base in CI,
because its wave runs before - or alongside - the image it depends on.

These tests close that gap: every in-repo image reference in a Dockerfile must be a
declared edge, every declared edge must correspond to a reference, `sbom.runtime_base` must
name the image the final stage is built from, and the resulting plan must fit the number of
build waves the workflows actually have.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_levels import MAX_LEVELS, Graph, build_plan, resolve_release_build_set  # noqa: E402

CONTAINERS_DIR = REPO_ROOT / "containers"

# `FROM ${REGISTRY}app-bricks/<name>:${BASE_IMAGE_VERSION} [AS stage]` and the
# `COPY --from=<stage>` that goes with it, plus a direct `COPY --from=<image>`.
IMAGE_REFERENCE = re.compile(r"app-bricks/([A-Za-z0-9._-]+):")
FROM_LINE = re.compile(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", re.MULTILINE | re.IGNORECASE)

CONTAINERS = sorted(path.parent for path in CONTAINERS_DIR.glob("*/*/ci.json"))
CONTAINER_IDS = [f"{path.parent.name}/{path.name}" for path in CONTAINERS]


def _ci_json(directory: Path) -> dict:
    return json.loads((directory / "ci.json").read_text(encoding="utf-8"))


def _dockerfile(directory: Path) -> str:
    return (directory / "Dockerfile").read_text(encoding="utf-8")


def _referenced_containers(directory: Path) -> set[str]:
    """In-repo container images this Dockerfile pulls, in any stage."""
    known = {path.name for path in CONTAINERS}
    return {name for name in IMAGE_REFERENCE.findall(_dockerfile(directory)) if name in known} - {directory.name}


def _final_from(directory: Path) -> str:
    """The image the last build stage starts from, with the build args left unexpanded.

    A last stage built `FROM <an earlier stage of this Dockerfile>` is followed back to the
    external image that stage came from, which is the one the delta SBOM is taken against.
    """
    stages = FROM_LINE.findall(_dockerfile(directory))
    assert stages, f"{directory.name}/Dockerfile has no FROM line"
    named = {alias: image for image, alias in stages if alias}
    image = stages[-1][0]
    seen: set[str] = set()
    while image in named and image not in seen:
        seen.add(image)
        image = named[image]
    return image


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_every_in_repo_image_a_dockerfile_pulls_is_a_declared_edge(directory: Path):
    """A parent must list every container whose Dockerfile references it, or CI builds them
    out of order and the child silently gets the previous release's base."""
    for parent in sorted(_referenced_containers(directory)):
        parent_dir = next(path for path in CONTAINERS if path.name == parent)
        downstream = _ci_json(parent_dir).get("downstream") or []
        assert directory.name in downstream, (
            f"{directory.name}/Dockerfile pulls app-bricks/{parent}, but {parent}/ci.json does not list "
            f"'{directory.name}' in downstream: CI would build them in the wrong order"
        )


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_every_declared_edge_is_actually_used(directory: Path):
    """The mirror of the above: a stale downstream entry drags a container into every
    rebuild of an image it no longer derives from."""
    for child in _ci_json(directory).get("downstream") or []:
        child_dir = next((path for path in CONTAINERS if path.name == child), None)
        assert child_dir is not None, f"{directory.name}/ci.json lists unknown downstream container '{child}'"
        assert directory.name in _referenced_containers(child_dir), (
            f"{directory.name}/ci.json lists '{child}' as downstream, but {child}/Dockerfile does not reference "
            f"app-bricks/{directory.name} any more; drop the stale edge"
        )


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_sbom_runtime_base_matches_the_final_from(directory: Path):
    """The delta SBOM is only meaningful against the image the runtime layers sit on, which
    is the *last* stage's FROM - not an earlier stage a multi-stage build only copies from."""
    declared = _ci_json(directory)["sbom"]["runtime_base"]
    assert declared == _final_from(directory), (
        f"{directory.name}: sbom.runtime_base is {declared!r} but the final stage is FROM {_final_from(directory)!r}"
    )


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_the_base_group_and_the_base_image_flag_agree(directory: Path):
    """`containers/base/` is the group that is never a release target, and
    `resolve_release_build_set` reads the flag, not the folder. A base filed under
    `containers/base/` without the flag would be seeded by a `base/X.Y.Z` tag - which is
    meant to build nothing - and a flagged container in `ai`/`bricks` would silently drop
    out of its own group's release."""
    flagged = bool(_ci_json(directory).get("base_image", False))
    in_base_group = directory.parent.name == "base"
    assert flagged == in_base_group, (
        f"{directory.name}: lives in containers/{directory.parent.name}/ but base_image is {flagged}; "
        f"the flag must be set for - and only for - the containers under containers/base/"
    )


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_watch_paths_cover_the_container_directory(directory: Path):
    watched = _ci_json(directory).get("watch_paths") or []
    own = f"{directory.relative_to(REPO_ROOT).as_posix()}/"
    assert own in watched, f"{directory.name}: watch_paths must include its own directory ({own}), got {watched}"


@pytest.mark.parametrize("directory", CONTAINERS, ids=CONTAINER_IDS)
def test_a_container_pulling_an_in_repo_image_parametrizes_its_base_version(directory: Path):
    """CI points a build at the freshly rebuilt upstream through these two build args; a
    Dockerfile that hardcodes a tag instead would pull a released image mid-release."""
    if not _referenced_containers(directory):
        pytest.skip("no in-repo base image")
    dockerfile = _dockerfile(directory)
    for arg in ("REGISTRY", "BASE_IMAGE_VERSION"):
        assert re.search(rf"^ARG\s+{arg}\b", dockerfile, re.MULTILINE), f"{directory.name}/Dockerfile must declare ARG {arg}"
    assert "${BASE_IMAGE_VERSION}" in dockerfile, f"{directory.name}/Dockerfile must tag its in-repo bases with ${{BASE_IMAGE_VERSION}}"


@pytest.mark.parametrize("group", sorted({path.parent.name for path in CONTAINERS}))
def test_every_release_group_fits_the_build_waves_the_workflows_have(group: str):
    """`build_plan` raises past MAX_LEVELS, and MAX_LEVELS is only honoured if the workflows
    really chain that many build-l* jobs."""
    graph = Graph(CONTAINERS_DIR)
    waves = build_plan(graph, resolve_release_build_set(graph, group))
    assert len(waves) == MAX_LEVELS

    for workflow, job_prefix in ((".github/workflows/docker-build.yml", "build-l"), (".github/workflows/docker-publish.yml", "build-l")):
        text = (REPO_ROOT / workflow).read_text(encoding="utf-8")
        for level in range(MAX_LEVELS):
            assert f"{job_prefix}{level}:" in text, f"{workflow} has no {job_prefix}{level} job, but build_levels emits level_{level}"
            assert f"outputs.level_{level} }}}}" in text, f"{workflow} never reads the level_{level} output"
