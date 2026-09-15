# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the container graph and the container scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import ContainerError, Containers, main, parent_container, resolve_base_image, scaffold, target_blocks  # noqa: E402


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
    with pytest.raises(ContainerError, match="No FROM"):
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
    with pytest.raises(ContainerError, match="Duplicate container name"):
        Containers(containers_dir)


def test_unknown_parent_is_rejected(tmp_path):
    with pytest.raises(ContainerError, match="unknown container 'ghost'"):
        Containers(make_containers_dir(tmp_path, {"orphan": ("ai", parent_from("ghost"))}))


def test_flat_layout_yields_no_containers(tmp_path):
    (tmp_path / "containers" / "python-slim").mkdir(parents=True)
    (tmp_path / "containers" / "python-slim" / "Dockerfile").write_text("FROM a:1\n")
    with pytest.raises(ContainerError, match="No containers found"):
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
    with pytest.raises(ContainerError, match="Unknown container"):
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
        "bake target 'ghost' has no Dockerfile under containers/",
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


# --- scaffold

BAKE = """group "default" {
  targets = [
    "python-slim",
    "python-base",
    "python-apps-base",
    "qairt-common-base",
  ]
}

target "python-slim" {
  inherits   = ["_common"]
  context    = "containers/base/python-slim"
}

target "python-base" {
  inherits   = ["_downstream"]
  context    = "containers/base/python-base"
  contexts   = parent_context("python-slim")
}

target "python-apps-base" {
  inherits   = ["_downstream"]
  context    = "containers/bricks/python-apps-base"
  contexts   = parent_context("python-base")
}

target "qairt-common-base" {
  inherits   = ["_common"]
  context    = "containers/base/qairt-common-base"
}
"""

README = """# Containers

## Inventory

| Container | Group | Built `FROM` | Purpose |
|---|---|---|---|
| `python-slim` | base | `python:3.13-slim-trixie` | Minimal Python layer |
| `python-base` | base | `python-slim` | System deps |
| `python-apps-base` | bricks | `python-base` | App runtime |
| `qairt-common-base` | base | `python:3.13-slim-trixie` | Qualcomm runtime |

```mermaid
graph LR
  slim[python-slim] --> base[python-base] --> apps[python-apps-base]
  qairt[qairt-common-base]
```

## Anatomy
"""

LICENSED = """cache_path: .licenses

apps:
  - name: python-base
    source_path: .
    sources:
      pip: true
    python:
      virtual_env_dir: "/venvs/python-base"
    venv:
      project: containers/base/python-base

stale_records_action: error

allowed:
  - mit
"""

DEPENDABOT = """version: 2
updates:
  - package-ecosystem: uv
    directories:
      - /
      - /containers/base/python-base
    schedule:
      interval: weekly
  - package-ecosystem: docker
    directories:
      - /containers/*/*
    schedule:
      interval: weekly
"""

UV_STAGE = "FROM ghcr.io/astral-sh/uv:9.9.9@sha256:abc AS uv\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    containers = tmp_path / "containers"
    dockerfiles = {
        "base/python-slim": "FROM python:3.13-slim-trixie@sha256:abc\n",
        "base/python-base": UV_STAGE + parent_from("python-slim"),
        "bricks/python-apps-base": parent_from("python-base"),
        "base/qairt-common-base": "FROM python:3.13-slim-trixie@sha256:abc\n",
    }
    for path, dockerfile in dockerfiles.items():
        (containers / path).mkdir(parents=True)
        (containers / path / "Dockerfile").write_text(dockerfile)
    (containers / "README.md").write_text(README)
    (tmp_path / "docker-bake.hcl").write_text(BAKE)
    (tmp_path / ".licensed.yml").write_text(LICENSED)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "dependabot.yml").write_text(DEPENDABOT)
    return tmp_path


def target_order(hcl: str) -> list[str]:
    blocks = target_blocks(hcl)
    return sorted(blocks, key=lambda name: blocks[name][0])


def group_order(hcl: str) -> list[str]:
    group = hcl[hcl.index('group "default"') : hcl.index("]")]
    return [line.strip().strip('",') for line in group.splitlines() if line.strip().startswith('"')]


def test_derived_python_container_is_registered_everywhere(repo: Path) -> None:
    steps = scaffold(repo, "my-runner", "bricks", "python-slim", "Runs things", python=True)

    dockerfile = (repo / "containers/bricks/my-runner/Dockerfile").read_text()
    assert "ARG REGISTRY\nARG BASE_IMAGE_VERSION=latest" in dockerfile
    assert parent_from("python-slim").strip() in dockerfile
    assert "FROM ghcr.io/astral-sh/uv:9.9.9@sha256:abc AS uv" in dockerfile, "uv stage reuses the image the other Dockerfiles mount"
    assert "uv export --frozen --project /tmp/deps" in dockerfile
    assert (repo / "containers/bricks/my-runner/pyproject.toml").read_text().startswith("# Python packages this image installs")

    containers = Containers(repo / "containers")
    assert containers.parent["my-runner"] == "python-slim"

    hcl = (repo / "docker-bake.hcl").read_text()
    assert 'inherits   = ["_downstream"]' in target_blocks_text(hcl, "my-runner")
    assert 'contexts   = parent_context("python-slim")' in target_blocks_text(hcl, "my-runner")
    assert target_order(hcl) == ["python-slim", "python-base", "python-apps-base", "my-runner", "qairt-common-base"], "after the parent's subtree"
    assert group_order(hcl) == target_order(hcl)

    readme = (repo / "containers/README.md").read_text()
    assert (
        "| `python-slim` | base | `python:3.13-slim-trixie` | Minimal Python layer |\n| `my-runner` | bricks | `python-slim` | Runs things |\n"
        in readme
    )
    assert "  slim --> myrunner[my-runner]\n```" in readme

    assert "  - name: my-runner\n" in (repo / ".licensed.yml").read_text()
    assert "project: containers/bricks/my-runner\n\nstale_records_action: error" in (repo / ".licensed.yml").read_text()
    assert "      - /containers/base/python-base\n      - /containers/bricks/my-runner\n" in (repo / ".github/dependabot.yml").read_text()
    assert any("task deps:lock" in step for step in steps)


def test_external_base_container_uses_common_and_goes_last(repo: Path) -> None:
    steps = scaffold(repo, "ei-runner", "bricks", "docker.io/edgeimpulse/runner:1.0@sha256:def", "Edge Impulse", python=False)

    dockerfile = (repo / "containers/bricks/ei-runner/Dockerfile").read_text()
    assert "ARG REGISTRY" not in dockerfile
    assert "FROM docker.io/edgeimpulse/runner:1.0@sha256:def\n" in dockerfile
    assert "uv" not in dockerfile
    assert not (repo / "containers/bricks/ei-runner/pyproject.toml").exists()
    assert Containers(repo / "containers").parent["ei-runner"] is None

    hcl = (repo / "docker-bake.hcl").read_text()
    target = target_blocks_text(hcl, "ei-runner")
    assert 'inherits   = ["_common"]' in target
    assert "contexts" not in target
    assert target_order(hcl)[-1] == "ei-runner"
    assert group_order(hcl)[-1] == "ei-runner"

    readme = (repo / "containers/README.md").read_text()
    qairt_row = "| `qairt-common-base` | base | `python:3.13-slim-trixie` | Qualcomm runtime |\n"
    assert qairt_row + "| `ei-runner` | bricks | `docker.io/edgeimpulse/runner:1.0` | Edge Impulse |\n" in readme, "after the last row"
    assert "  eirunner[ei-runner]\n```" in readme
    assert "ei-runner" not in (repo / ".licensed.yml").read_text()
    assert "ei-runner" not in (repo / ".github/dependabot.yml").read_text()
    assert not any("digest" in step for step in steps)


def test_external_base_without_digest_is_warned(repo: Path) -> None:
    steps = scaffold(repo, "plain", "bricks", "python:3.13-slim", "Plain", python=False)
    assert "pin it with @sha256" in steps[0]


@pytest.mark.parametrize("name", ["Bad_Name", "-lead", "trail-", "python-base"])
def test_invalid_or_existing_names_are_rejected(repo: Path, name: str) -> None:
    with pytest.raises(ContainerError):
        scaffold(repo, name, "bricks", "python-slim", "x", python=False)
    assert not (repo / "containers/bricks" / name).exists()


def test_unknown_group_is_rejected(repo: Path) -> None:
    with pytest.raises(ContainerError, match="not a container group"):
        scaffold(repo, "my-runner", "tools", "python-slim", "x", python=True)


def test_nothing_is_written_when_a_registration_fails(repo: Path) -> None:
    (repo / "docker-bake.hcl").write_text('target "python-slim" {\n  context = "containers/base/python-slim"\n}\n')
    with pytest.raises(ContainerError, match="docker-bake.hcl"):
        scaffold(repo, "my-runner", "bricks", "python-slim", "x", python=True)
    assert not (repo / "containers/bricks/my-runner").exists()
    assert "my-runner" not in (repo / "containers/README.md").read_text()
    assert "my-runner" not in (repo / ".licensed.yml").read_text()


def test_cli_reports_errors_and_next_steps(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--repo-root", str(repo), "new", "my-runner", "--group", "bricks", "--from", "python-slim"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Scaffolded containers/bricks/my-runner. Next steps:")
    assert "task deps:lock" in out

    assert main(["--repo-root", str(repo), "new", "my-runner", "--group", "bricks", "--from", "python-slim", "--no-python"]) == 1
    assert "already exists" in capsys.readouterr().err


def target_blocks_text(hcl: str, name: str) -> str:
    start, end = target_blocks(hcl)[name]
    return hcl[start:end]
