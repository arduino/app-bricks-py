# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Dockerfile-derived container graph."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container_deps import ContainerDepsError, Containers, parent_container, resolve_base_image  # noqa: E402


def parent_from(name: str) -> str:
    return f"FROM ${{REGISTRY}}app-bricks/{name}:${{BASE_IMAGE_VERSION}}\n"


def make_containers_dir(tmp_path: Path, spec: dict[str, tuple[str, str]]) -> Path:
    """Create ``containers/<group>/<name>/Dockerfile`` files from a ``{name: (group, dockerfile)}`` spec."""
    containers_dir = tmp_path / "containers"
    for name, (group, dockerfile) in spec.items():
        directory = containers_dir / group / name
        directory.mkdir(parents=True)
        (directory / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return containers_dir


# The real chains, spread over the groups as in the repo.
TREE = {
    "python-slim": ("base", "FROM python:3.13-slim-trixie@sha256:abc AS production\nRUN true\n"),
    "python-base": (
        "base",
        "ARG REGISTRY\nARG BASE_IMAGE_VERSION=latest\nFROM python:3.13-slim-trixie AS builder\n"
        + parent_from("python-slim")
        + "COPY --from=builder /x /x\n",
    ),
    "python-apps-base": ("bricks", "ARG REGISTRY\nARG BASE_IMAGE_VERSION=latest\n" + parent_from("python-base")),
    "qairt-common-base": (
        "base",
        "FROM python:3.13-slim-trixie@sha256:abc AS base\nFROM base AS native\nFROM base AS runtime\nCOPY --from=native /a /a\n",
    ),
    "aihub-models-runner": ("ai", "FROM ghcr.io/astral-sh/uv:0.10.3 AS uv\n" + parent_from("qairt-common-base")),
    "gesture-recognition-runner": ("ai", parent_from("aihub-models-runner")),
    "ei-models-runner": ("ai", "FROM public.ecr.aws/g7a8t7v6/inference-container:v1.92.3\n"),
}


def test_base_image_is_the_final_stage_base_through_aliases(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.13 AS base\nFROM other:1 AS tools\nFROM base AS runtime\nCOPY --from=tools /t /t\n")
    assert resolve_base_image(dockerfile) == "python:3.13"


def test_base_image_with_platform_flag_and_digest(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM --platform=linux/arm64 python:3.13@sha256:abc\n")
    assert resolve_base_image(dockerfile) == "python:3.13@sha256:abc"


def test_dockerfile_without_from_is_rejected(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("RUN true\n")
    with pytest.raises(ContainerDepsError, match="No FROM"):
        resolve_base_image(dockerfile)


def test_parent_is_recognised_only_from_the_repository_reference():
    assert parent_container("${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}") == "python-slim"
    assert parent_container("ghcr.io/arduino/app-bricks/python-slim:1.0.0") is None
    assert parent_container("python:3.13-slim-trixie") is None


def test_containers_are_identified_by_leaf_name_across_groups(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    assert containers.names == sorted(TREE)
    assert containers.parent["python-apps-base"] == "python-base"
    assert containers.parent["gesture-recognition-runner"] == "aihub-models-runner"
    assert containers.parent["qairt-common-base"] is None
    assert containers.base["qairt-common-base"] == "python:3.13-slim-trixie@sha256:abc"
    assert containers.parent["ei-models-runner"] is None


def test_duplicate_leaf_name_is_rejected(tmp_path):
    spec = {"twin": ("ai", "FROM a:1\n")}
    containers_dir = make_containers_dir(tmp_path, spec)
    (containers_dir / "bricks" / "twin").mkdir(parents=True)
    (containers_dir / "bricks" / "twin" / "Dockerfile").write_text("FROM b:1\n")
    with pytest.raises(ContainerDepsError, match="Duplicate container name"):
        Containers(containers_dir)


def test_unknown_parent_is_rejected(tmp_path):
    with pytest.raises(ContainerDepsError, match="unknown container 'ghost'"):
        Containers(make_containers_dir(tmp_path, {"orphan": ("ai", parent_from("ghost"))}))


def test_flat_layout_yields_no_containers(tmp_path):
    (tmp_path / "containers" / "python-slim").mkdir(parents=True)
    (tmp_path / "containers" / "python-slim" / "Dockerfile").write_text("FROM a:1\n")
    with pytest.raises(ContainerDepsError, match="No containers found"):
        Containers(tmp_path / "containers")


def test_closure_adds_children_then_parents(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    # A base pulls in everything derived from it.
    assert containers.closure(["python-slim"]) == ["python-apps-base", "python-base", "python-slim"]
    # A leaf pulls in its bases, not its siblings.
    assert containers.closure(["gesture-recognition-runner"]) == ["aihub-models-runner", "gesture-recognition-runner", "qairt-common-base"]
    # A middle node pulls in both directions.
    assert containers.closure(["python-base"]) == ["python-apps-base", "python-base", "python-slim"]
    assert containers.closure(["ei-models-runner"]) == ["ei-models-runner"]


def test_closure_rejects_unknown_containers(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    with pytest.raises(ContainerDepsError, match="Unknown container"):
        containers.closure(["does-not-exist"])


def test_tree_groups_roots_by_external_base(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    tree = containers.tree()
    assert tree.startswith("public.ecr.aws/g7a8t7v6/inference-container:v1.92.3\n└─ ei-models-runner\n")
    python_chains = tree.split("python:3.13-slim-trixie@sha256:abc\n", 1)[1]
    assert python_chains == (
        "├─ python-slim\n"
        "│  └─ python-base\n"
        "│     └─ python-apps-base\n"
        "└─ qairt-common-base\n"
        "   └─ aihub-models-runner\n"
        "      └─ gesture-recognition-runner"
    )


def bake_definition(**contexts: dict[str, str]) -> dict:
    """A ``bake --print`` definition with one target per TREE container and the given contexts."""
    return {"target": {name: {"contexts": contexts.get(name.replace("-", "_"), {})} for name in TREE}}


PARENT_LINKS = {
    "python_base": {"ghcr.io/arduino/app-bricks/python-slim:local": "target:python-slim"},
    "python_apps_base": {"wheel": "dist", "ghcr.io/arduino/app-bricks/python-base:local": "target:python-base"},
    "aihub_models_runner": {"ghcr.io/arduino/app-bricks/qairt-common-base:local": "target:qairt-common-base"},
    "gesture_recognition_runner": {"ghcr.io/arduino/app-bricks/aihub-models-runner:local": "target:aihub-models-runner"},
}


def test_check_bake_accepts_a_definition_matching_the_dockerfiles(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    assert containers.check_bake(bake_definition(**PARENT_LINKS)) == []


def test_check_bake_reports_containers_and_targets_that_do_not_match(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    definition = bake_definition(**PARENT_LINKS)
    del definition["target"]["ei-models-runner"]
    definition["target"]["ghost"] = {}
    assert containers.check_bake(definition) == [
        "'ei-models-runner' has a Dockerfile but no bake target in the default group",
        "bake target 'ghost' has no containers/ghost/Dockerfile",
    ]


def test_check_bake_reports_missing_wrong_and_spurious_parent_links(tmp_path):
    containers = Containers(make_containers_dir(tmp_path, TREE))
    links = dict(PARENT_LINKS)
    links["python_apps_base"] = {"wheel": "dist"}
    links["gesture_recognition_runner"] = {"x": "target:python-slim"}
    links["ei_models_runner"] = {"x": "target:python-slim"}
    assert containers.check_bake(bake_definition(**links)) == [
        "bake target 'ei-models-runner' links python-slim but its Dockerfile builds FROM an external image",
        "bake target 'gesture-recognition-runner' links python-slim but its Dockerfile builds FROM aihub-models-runner",
        "bake target 'python-apps-base' links no parent but its Dockerfile builds FROM python-base",
    ]
