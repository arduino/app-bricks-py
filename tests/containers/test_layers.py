# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A layer is an image FROM scratch holding files only, never run and never published.

An image adds it through a stage of its own (``FROM ${REGISTRY}app-bricks/<layer>:${BASE_IMAGE_VERSION} AS <alias>``)
and ``COPY --link``, which reproduces the layer's own blobs rather than copying the files, and its
docker-bake.hcl target links the layer next to its parent, so bake builds it in-graph. The layer's own
target inherits ``_layer``: cache only, no tags. These tests read the Dockerfiles and docker-bake.hcl as
text, so they run without Docker.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import Containers, target_blocks  # noqa: E402

CONTAINERS = Containers(REPO_ROOT / "containers")
BAKE = (REPO_ROOT / "docker-bake.hcl").read_text(encoding="utf-8")
BLOCKS = target_blocks(BAKE)

LAYERS = [name for name in CONTAINERS.names if CONTAINERS.is_layer(name)]
IMPORTS = [(layer, importer) for layer in LAYERS for importer in CONTAINERS.names if layer in CONTAINERS.layers[importer]]
LAYER_STAGE = r"^FROM\s+\$\{{REGISTRY\}}app-bricks/{layer}:\$\{{BASE_IMAGE_VERSION\}}\s+AS\s+(\S+)\s*$"
COPY = re.compile(r"^COPY\s+((?:--\S+\s+)*)(.+?)\s+(\S+)\s*$", re.MULTILINE)


def dockerfile_of(name: str) -> str:
    return (CONTAINERS.directory[name] / "Dockerfile").read_text(encoding="utf-8")


def instructions(dockerfile: str) -> list[str]:
    """The instruction keywords of a Dockerfile, comments and continuations aside."""
    return [line.split()[0].upper() for line in dockerfile.splitlines() if line.strip() and not line.startswith((" ", "\t", "#"))]


def layer_paths(layer: str) -> dict[PurePosixPath, list[Path]]:
    """Map each path the layer holds to the files of its build context copied there."""
    held: dict[PurePosixPath, list[Path]] = {}
    for flags, sources, dest in COPY.findall(dockerfile_of(layer)):
        if not flags:
            held.setdefault(PurePosixPath(dest), []).extend(CONTAINERS.directory[layer] / source for source in sources.split())
    return held


def copies_from_layer(dockerfile: str, layer: str) -> list[tuple[list[str], PurePosixPath]]:
    """The ``COPY`` instructions reading from the layer's stage: their flags and source path."""
    aliases = re.findall(LAYER_STAGE.format(layer=re.escape(layer)), dockerfile, re.MULTILINE | re.IGNORECASE)
    assert len(aliases) == 1, f"{layer} must be imported through exactly one stage, found {aliases}"
    return [(flags.split(), PurePosixPath(src)) for flags, src, _ in COPY.findall(dockerfile) if f"--from={aliases[0]}" in flags.split()]


def test_every_layer_is_added_by_an_image():
    assert LAYERS, "no layer found: the scan is broken"
    for layer in LAYERS:
        assert [importer for candidate, importer in IMPORTS if candidate == layer], f"{layer} is a layer nothing adds"


@pytest.mark.parametrize("layer", LAYERS)
def test_a_layer_is_a_scratch_image_of_files_only(layer: str):
    assert CONTAINERS.parent[layer] is None and CONTAINERS.layers[layer] == []
    # scratch has no shell: a RUN would fail, and nothing but files belongs in a layer.
    assert set(instructions(dockerfile_of(layer))) == {"FROM", "COPY"}
    held = layer_paths(layer)
    assert held, f"{layer} copies nothing"
    for path, sources in held.items():
        for source in sources:
            assert source.exists(), f"{layer} copies {source} into {path}, which does not exist"


@pytest.mark.parametrize("layer", LAYERS)
def test_a_layer_target_is_built_in_graph_and_never_published(layer: str):
    start, end = BLOCKS[layer]
    block = BAKE[start:end]
    assert re.search(r'inherits\s*=\s*\["_layer"\]', block), f"bake target '{layer}' does not inherit _layer"
    assert "tags" not in block and "cache-to" not in block, f"bake target '{layer}' is tagged or cached:\n{block}"
    group = re.search(r'^group "default" \{\n.*?^\}\n', BAKE, re.MULTILINE | re.DOTALL)
    assert group and f'"{layer}",' in group.group(0), f"'{layer}' is not in the default group"


def test_the_layer_base_target_is_cache_only():
    start, end = BLOCKS["_layer"]
    block = BAKE[start:end]
    assert re.search(r'output\s*=\s*\["type=cacheonly"\]', block), block
    assert "tags" not in block and "cache-to" not in block, block


@pytest.mark.parametrize(("layer", "importer"), IMPORTS)
def test_an_importer_links_the_layer_and_adds_only_what_it_holds(layer: str, importer: str):
    start, end = BLOCKS[importer]
    assert f'parent_context("{layer}")' in BAKE[start:end], f"bake target '{importer}' does not link {layer}, bake would pull it"

    copied = [path for _, path in copies_from_layer(dockerfile_of(importer), layer)]
    assert copied, f"{importer} imports {layer} but adds nothing of it"
    held = layer_paths(layer)
    for path in copied:
        assert any(path == root or root in path.parents for root in held), f"{importer} copies {path}, which {layer} does not hold"


@pytest.mark.parametrize(("layer", "importer"), IMPORTS)
def test_an_importer_adds_the_layer_blobs_rather_than_copying_the_files(layer: str, importer: str):
    """Only --link reproduces the layer's own layers; a plain COPY --from writes the files into new ones."""
    plain = [str(path) for flags, path in copies_from_layer(dockerfile_of(importer), layer) if "--link" not in flags]
    assert not plain, f"{importer} copies {plain} out of {layer} without --link, as new layers instead of the layer's own"


def test_a_plain_copy_out_of_the_layer_is_told_apart_from_a_linked_one():
    dockerfile = (
        "FROM ${REGISTRY}app-bricks/some-layer:${BASE_IMAGE_VERSION} AS fw\n"
        "FROM python:3.13\n"
        "COPY --link --from=fw /opt/aihub/ /opt/aihub/\n"
        "COPY --from=fw /app/ ./\n"
        "COPY --link --from=other /x /x\n"
    )
    assert [("--link" in flags, str(path)) for flags, path in copies_from_layer(dockerfile, "some-layer")] == [(True, "/opt/aihub"), (False, "/app")]
