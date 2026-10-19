# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""aihub-framework is a layer: a minimal image whose layers the runner bases add, the `aihub` framework.

The layer holds only files, FROM scratch. An image imports it through a stage of its own
(``FROM ${REGISTRY}app-bricks/aihub-framework:${BASE_IMAGE_VERSION} AS <alias>``), adds its layers
with ``COPY --link`` and links it in docker-bake.hcl, and the runners deriving from those images
import the framework as ``aihub``. These tests read the Dockerfiles, docker-bake.hcl and the Python
sources as text, so they run without Docker.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import Containers, target_blocks  # noqa: E402

LAYER = "aihub-framework"
CONTAINERS = Containers(REPO_ROOT / "containers")
LAYER_DIR = CONTAINERS.directory[LAYER]
LAYER_DOCKERFILE = (LAYER_DIR / "Dockerfile").read_text(encoding="utf-8")
BAKE = (REPO_ROOT / "docker-bake.hcl").read_text(encoding="utf-8")
BLOCKS = target_blocks(BAKE)

IMPORTERS = [name for name in CONTAINERS.names if LAYER in CONTAINERS.layers[name]]
LAYER_STAGE = re.compile(rf"^FROM\s+\$\{{REGISTRY\}}app-bricks/{LAYER}:\$\{{BASE_IMAGE_VERSION\}}\s+AS\s+(\S+)\s*$", re.MULTILINE | re.IGNORECASE)
COPY = re.compile(r"^COPY\s+((?:--\S+\s+)*)(\S+)\s+(\S+)\s*$", re.MULTILINE)


def instructions(dockerfile: str) -> list[str]:
    """The instruction keywords of a Dockerfile, comments and continuations aside."""
    return [line.split()[0].upper() for line in dockerfile.splitlines() if line.strip() and not line.startswith((" ", "\t", "#"))]


def layer_paths() -> dict[PurePosixPath, Path]:
    """Map each path the layer holds to the directory of the layer's build context it comes from."""
    return {PurePosixPath(dest): LAYER_DIR / src for flags, src, dest in COPY.findall(LAYER_DOCKERFILE) if not flags}


def runners_of(importer: str) -> list[str]:
    """``importer`` and every container deriving from it."""
    names = [importer]
    for child in CONTAINERS.children(importer):
        names += runners_of(child)
    return names


def test_the_layer_is_a_scratch_image_of_files_only():
    assert CONTAINERS.base[LAYER] == "scratch"
    assert CONTAINERS.parent[LAYER] is None and CONTAINERS.layers[LAYER] == []
    # scratch has no shell: a RUN would fail, and nothing but files belongs in a layer.
    assert set(instructions(LAYER_DOCKERFILE)) == {"FROM", "COPY"}
    for path, source in layer_paths().items():
        assert source.is_dir(), f"the layer copies {source} into {path}, which is not a directory"
    assert (LAYER_DIR / "aihub" / "__init__.py").is_file() and (LAYER_DIR / "app" / "main.py").is_file()


def test_both_runner_bases_import_the_layer():
    assert IMPORTERS == ["aihub-litert-models-runner", "aihub-onnx-models-runner"]


def copies_from_layer(dockerfile: str) -> list[tuple[list[str], PurePosixPath]]:
    """The ``COPY`` instructions reading from the layer's stage: their flags and source path."""
    aliases = LAYER_STAGE.findall(dockerfile)
    assert len(aliases) == 1, f"the layer must be imported through exactly one stage, found {aliases}"
    return [(flags.split(), PurePosixPath(src)) for flags, src, _ in COPY.findall(dockerfile) if f"--from={aliases[0]}" in flags.split()]


@pytest.mark.parametrize("importer", IMPORTERS)
def test_an_importer_links_the_layer_and_adds_only_what_it_holds(importer: str):
    dockerfile = (CONTAINERS.directory[importer] / "Dockerfile").read_text(encoding="utf-8")
    start, end = BLOCKS[importer]
    assert f'parent_context("{LAYER}")' in BAKE[start:end], f"bake target '{importer}' does not link {LAYER}, bake would pull it"

    copies = copies_from_layer(dockerfile)
    assert copies, f"{importer} imports {LAYER} but adds nothing of it"
    copied = [path for _, path in copies]
    held = layer_paths()
    for path in copied:
        assert any(path == root or root in path.parents for root in held), f"{importer} copies {path}, which {LAYER} does not hold"
    # The framework lands in /opt/aihub and is imported as `aihub`.
    assert PurePosixPath("/opt/aihub") in copied
    assert 'ENV PYTHONPATH="/opt"' in dockerfile


@pytest.mark.parametrize("importer", IMPORTERS)
def test_an_importer_adds_the_layer_blobs_rather_than_copying_the_files(importer: str):
    """Only --link reproduces the layer's own layers; a plain COPY --from writes the files into new ones."""
    dockerfile = (CONTAINERS.directory[importer] / "Dockerfile").read_text(encoding="utf-8")
    plain = [str(path) for flags, path in copies_from_layer(dockerfile) if "--link" not in flags]
    assert not plain, f"{importer} copies {plain} out of {LAYER} without --link, as new layers instead of the layer's own"


def test_a_plain_copy_out_of_the_layer_is_told_apart_from_a_linked_one():
    dockerfile = (
        f"FROM ${{REGISTRY}}app-bricks/{LAYER}:${{BASE_IMAGE_VERSION}} AS fw\n"
        "FROM python:3.13\n"
        "COPY --link --from=fw /opt/aihub/ /opt/aihub/\n"
        "COPY --from=fw /app/ ./\n"
        "COPY --link --from=other /x /x\n"
    )
    assert [("--link" in flags, str(path)) for flags, path in copies_from_layer(dockerfile)] == [(True, "/opt/aihub"), (False, "/app")]


def defined_names(module: Path) -> set[str]:
    """The names a module binds at top level: definitions, assignments and imports."""
    names: set[str] = set()
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {target.id for target in targets if isinstance(target, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return names


def framework_module(dotted: str) -> Path | None:
    """The source of an ``aihub[.x]`` module inside the layer's build context."""
    base = LAYER_DIR.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def aihub_imports(source: Path) -> list[tuple[str, list[str]]]:
    """The ``aihub`` modules a Python file imports, with the names it takes from each."""
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] == "aihub":
            found.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names if alias.name.split(".")[0] == "aihub"]
    return found


RUNNER_SOURCES = sorted(
    {
        source
        for importer in IMPORTERS
        for runner in runners_of(importer)
        if runner != LAYER
        for source in CONTAINERS.directory[runner].rglob("*.py")
        if ".venv" not in source.parts and "tests" not in source.parts
    }
    | set((LAYER_DIR / "app").glob("*.py"))
)


def test_the_runners_import_the_framework():
    assert any(aihub_imports(source) for source in RUNNER_SOURCES), "no runner imports `aihub`: the scan finds no sources"


@pytest.mark.parametrize("source", RUNNER_SOURCES, ids=[str(path.relative_to(REPO_ROOT)) for path in RUNNER_SOURCES])
def test_every_aihub_import_of_a_runner_resolves_in_the_layer(source: Path):
    for dotted, names in aihub_imports(source):
        module = framework_module(dotted)
        assert module is not None, f"{source.relative_to(REPO_ROOT)} imports {dotted}, which {LAYER} does not hold"
        missing = [name for name in names if name not in defined_names(module) and not framework_module(f"{dotted}.{name}")]
        assert not missing, f"{source.relative_to(REPO_ROOT)} imports {missing} from {dotted}, which does not define them"
