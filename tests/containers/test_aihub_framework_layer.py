# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""aihub-framework is the layer holding the `aihub` runner framework and the app skeleton.

Both runner bases add it at /opt/aihub and /app and put /opt on PYTHONPATH, so their runners
``import aihub``, and every ``aihub`` import of a runner must resolve in the layer. The layer contract
itself is checked by test_layers.py. These tests read the Dockerfiles and the Python sources as text,
so they run without Docker.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import Containers  # noqa: E402
from tests.containers.test_layers import copies_from_layer  # noqa: E402

LAYER = "aihub-framework"
CONTAINERS = Containers(REPO_ROOT / "containers")
LAYER_DIR = CONTAINERS.directory[LAYER]
IMPORTERS = [name for name in CONTAINERS.names if LAYER in CONTAINERS.layers[name]]


def runners_of(importer: str) -> list[str]:
    """``importer`` and every container deriving from it."""
    names = [importer]
    for child in CONTAINERS.children(importer):
        names += runners_of(child)
    return names


def test_the_layer_holds_the_package_and_the_app_skeleton():
    assert (LAYER_DIR / "aihub" / "__init__.py").is_file() and (LAYER_DIR / "app" / "main.py").is_file()


def test_both_runner_bases_import_the_layer():
    assert IMPORTERS == ["aihub-litert-models-runner", "aihub-onnx-models-runner"]


@pytest.mark.parametrize("importer", IMPORTERS)
def test_an_importer_puts_the_framework_on_the_python_path(importer: str):
    dockerfile = (CONTAINERS.directory[importer] / "Dockerfile").read_text(encoding="utf-8")
    # The framework lands in /opt/aihub and is imported as `aihub`.
    assert PurePosixPath("/opt/aihub") in [path for _, path in copies_from_layer(dockerfile, LAYER)]
    assert 'ENV PYTHONPATH="/opt"' in dockerfile


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
