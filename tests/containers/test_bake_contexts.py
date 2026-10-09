# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Every named build context a Dockerfile reads must be declared by its docker-bake.hcl target.

A Dockerfile can ``COPY --from=<name>`` (or ``RUN --mount=from=<name>``) out of three things:
a stage of its own, an image reference, or a named context the bake target provides, like
``framework`` (containers/base/aihub-framework/, the `aihub` sources shared by the two runner
bases), ``models`` or ``wheel``. ``check-bake`` verifies only the parent links, so a target
missing one of its named contexts goes unnoticed until bake fails, or, worse, BuildKit resolves
the name as an image on Docker Hub. These tests read the Dockerfiles and docker-bake.hcl as
text, so they run without Docker.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import FROM_PATTERN, target_blocks  # noqa: E402

BAKE = (REPO_ROOT / "docker-bake.hcl").read_text(encoding="utf-8")
BLOCKS = target_blocks(BAKE)
DOCKERFILES = sorted((REPO_ROOT / "containers").glob("*/*/Dockerfile"))

FROM_REFERENCE = re.compile(r"(?:COPY\s+(?:--\S+\s+)*?--from=|--mount=(?:\S*,)?from=)([^\s,]+)", re.IGNORECASE)
NAMED_CONTEXT = re.compile(r'^\s*(?:\{\s*)?([A-Za-z0-9_-]+)\s*=\s*"([^"]+)"', re.MULTILINE)


def named_contexts_read(dockerfile: Path) -> set[str]:
    """The names a Dockerfile copies or mounts from that are neither its own stages nor image references."""
    text = dockerfile.read_text(encoding="utf-8")
    stages = {match.group(2).lower() for line in text.splitlines() if (match := FROM_PATTERN.match(line)) and match.group(2)}
    names = {match.group(1) for match in FROM_REFERENCE.finditer(text)}
    return {name for name in names if name.lower() not in stages and not re.search(r"[:/@$]", name)}


def declared_contexts(target: str) -> dict[str, str]:
    """The ``name = "path"`` named contexts of a bake target, parent links excluded."""
    start, end = BLOCKS[target]
    block = BAKE[start:end]
    contexts = block[block.index("contexts") :] if "contexts" in block else ""
    return {name: path for name, path in NAMED_CONTEXT.findall(contexts) if not path.startswith("target:")}


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=[path.parent.name for path in DOCKERFILES])
def test_every_named_context_a_dockerfile_reads_is_declared_by_its_target(dockerfile: Path):
    name = dockerfile.parent.name
    assert name in BLOCKS, f"{name} has no target in docker-bake.hcl"
    missing = named_contexts_read(dockerfile) - set(declared_contexts(name))
    assert not missing, f"{name}/Dockerfile reads the named context(s) {sorted(missing)}, but its docker-bake.hcl target does not declare them"


@pytest.mark.parametrize("target", sorted(t for t in BLOCKS if not t.startswith("_")))
def test_every_declared_named_context_exists(target: str):
    for name, path in declared_contexts(target).items():
        if name == "wheel":
            continue  # dist/ is filled by `task build:bricks` right before the build
        assert (REPO_ROOT / path).is_dir(), f"bake target '{target}' declares context '{name}' = {path}, which is not a directory"


@pytest.mark.parametrize("runner", ["aihub-litert-models-runner", "aihub-onnx-models-runner"])
def test_the_runner_bases_share_the_aihub_framework_sources(runner: str):
    """The framework is a source directory, not a container: both bases copy it from the same context."""
    assert declared_contexts(runner).get("framework") == "containers/base/aihub-framework"
    framework = REPO_ROOT / "containers" / "base" / "aihub-framework"
    assert not (framework / "Dockerfile").exists(), "aihub-framework is shared through a bake context, it is not an image"
    assert (framework / "aihub" / "__init__.py").is_file() and (framework / "app" / "main.py").is_file()


def test_named_contexts_are_told_apart_from_stages_and_images(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ghcr.io/astral-sh/uv:0.12.21 AS uv\n"
        "FROM python:3.13 AS builder\n"
        "FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}\n"
        "COPY --from=builder /a /a\n"
        "COPY --from=framework aihub/ /opt/aihub/\n"
        "COPY --from=models --chown=arduino:arduino models-list.yaml /app/\n"
        "COPY --from=ghcr.io/example/image:1 /b /b\n"
        "RUN --mount=from=uv,source=/uv,target=/bin/uv true\n",
        encoding="utf-8",
    )
    assert named_contexts_read(dockerfile) == {"framework", "models"}
