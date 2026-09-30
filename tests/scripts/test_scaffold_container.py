# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the container scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container_deps import Containers  # noqa: E402
from scripts.scaffold_container import ScaffoldError, main, scaffold, target_blocks  # noqa: E402

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


def parent_from(name: str) -> str:
    return f"FROM ${{REGISTRY}}app-bricks/{name}:${{BASE_IMAGE_VERSION}}\n"


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
    with pytest.raises(ScaffoldError):
        scaffold(repo, name, "bricks", "python-slim", "x", python=False)
    assert not (repo / "containers/bricks" / name).exists()


def test_unknown_group_is_rejected(repo: Path) -> None:
    with pytest.raises(ScaffoldError, match="not a container group"):
        scaffold(repo, "my-runner", "tools", "python-slim", "x", python=True)


def test_nothing_is_written_when_a_registration_fails(repo: Path) -> None:
    (repo / "docker-bake.hcl").write_text('target "python-slim" {\n  context = "containers/base/python-slim"\n}\n')
    with pytest.raises(ScaffoldError, match="docker-bake.hcl"):
        scaffold(repo, "my-runner", "bricks", "python-slim", "x", python=True)
    assert not (repo / "containers/bricks/my-runner").exists()
    assert "my-runner" not in (repo / "containers/README.md").read_text()
    assert "my-runner" not in (repo / ".licensed.yml").read_text()


def test_cli_reports_errors_and_next_steps(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["my-runner", "--group", "bricks", "--from", "python-slim", "--repo-root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Scaffolded containers/bricks/my-runner. Next steps:")
    assert "task deps:lock" in out

    assert main(["my-runner", "--group", "bricks", "--from", "python-slim", "--no-python", "--repo-root", str(repo)]) == 1
    assert "already exists" in capsys.readouterr().err


def target_blocks_text(hcl: str, name: str) -> str:
    start, end = target_blocks(hcl)[name]
    return hcl[start:end]
