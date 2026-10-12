# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-apps-base applies the source patches of its patches/ directory at build time: each must apply to the version
the image installs, and the FastAPI one must keep FastAPI working."""

import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from importlib.metadata import packages_distributions, version
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
IMAGE_DIR = ROOT / "containers" / "bricks" / "python-apps-base"
PATCHES = sorted((IMAGE_DIR / "patches").glob("*.patch"))
FASTAPI_PATCH = IMAGE_DIR / "patches" / "fastapi-defer-openapi-model-build.patch"

PATCH_FLAGS = ["-p1", "--forward", "--fuzz=0", "--no-backup-if-mismatch"]
IMAGE_PATCH_COMMAND = " ".join(["patch", *PATCH_FLAGS, "-d", "/usr/local/lib/python3.13/site-packages", "-i", '"$p"'])

PATCH_TOOL = shutil.which("patch")
needs_patch = pytest.mark.skipif(PATCH_TOOL is None, reason="needs GNU patch, as the image build")


def _canonical(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"] if "version" in p}  # The project itself has none


def _apply_to_a_copy(patch: Path, site_packages: Path) -> None:
    """Copy the installed packages or modules the patch changes into site_packages and apply it there, as the image build does."""
    targets = sorted(set(re.findall(r"^\+\+\+ b/([^/\s]+)", patch.read_text(encoding="utf-8"), re.MULTILINE)))
    assert targets, f"{patch.name} changes no file"
    locked = _locked_versions()
    site_packages.mkdir(parents=True)
    for target in targets:
        name = target.removesuffix(".py")
        spec = find_spec(name)
        assert spec is not None and spec.origin is not None, f"{patch.name} changes {target}, which is not installed"
        for distribution in packages_distributions().get(name, []):
            assert version(distribution) == locked[_canonical(distribution)], f"the venv is not on the {distribution} of uv.lock, run `uv sync`"
        if spec.submodule_search_locations:
            shutil.copytree(spec.submodule_search_locations[0], site_packages / target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(spec.origin, site_packages / target)

    assert PATCH_TOOL is not None
    result = subprocess.run([PATCH_TOOL, *PATCH_FLAGS, "-d", str(site_packages), "-i", str(patch)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"{patch.name} no longer applies to the installed {', '.join(targets)}:\n{result.stdout}{result.stderr}"


@pytest.fixture
def patched_site_packages(tmp_path: Path) -> Path:
    """The installed FastAPI, patched as the image does, in a directory that shadows it on PYTHONPATH."""
    site_packages = tmp_path / "site-packages"
    _apply_to_a_copy(FASTAPI_PATCH, site_packages)
    return site_packages


def _run_with(site_packages: Path, code: str) -> None:
    """Run code in a fresh interpreter that imports the patched FastAPI, whose path is sys.argv[1]."""
    pythonpath = [str(site_packages), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(site_packages)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_image_applies_every_patch_with_the_tested_flags_before_compiling():
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    assert "COPY ./patches/ /tmp/patches/" in dockerfile
    assert f'for p in "$@"; do {IMAGE_PATCH_COMMAND}; done' in dockerfile
    assert dockerfile.index(IMAGE_PATCH_COMMAND) < dockerfile.index("python -m compileall /usr/local/lib")


@needs_patch
@pytest.mark.parametrize("patch", PATCHES, ids=[p.name for p in PATCHES])
def test_patch_applies_to_the_locked_version(patch: Path, tmp_path: Path):
    _apply_to_a_copy(patch, tmp_path / "site-packages")


@needs_patch
def test_webui_import_leaves_the_openapi_models_unbuilt(patched_site_packages: Path):
    _run_with(
        patched_site_packages,
        """
        import sys
        import arduino.app_bricks.web_ui
        import fastapi
        from fastapi.openapi import models
        from pydantic import BaseModel

        assert fastapi.__file__.startswith(sys.argv[1]), f"not the patched FastAPI: {fastapi.__file__}"
        openapi_models = [v for v in vars(models).values() if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == models.__name__]
        assert openapi_models
        built = [model.__name__ for model in openapi_models if model.__pydantic_complete__]
        # The one exception: fastapi.security.OAuth2.__init__ instantiates OAuthFlows as a default argument
        assert built == ["OAuthFlows"], f"OpenAPI models built at import: {built}"
        """,
    )


@needs_patch
def test_openapi_models_build_on_first_use_and_app_models_as_usual(patched_site_packages: Path):
    _run_with(
        patched_site_packages,
        """
        from fastapi.openapi import models
        from fastapi.testclient import TestClient
        from pydantic import BaseModel
        from arduino.app_bricks.web_ui import WebUI

        class Item(BaseModel):
            name: str
            tags: list[str] = []

        assert Item.__pydantic_complete__, "the app's own models must still be built at definition"

        def create(item: Item) -> Item:
            return item

        ui = WebUI()
        ui.expose_api("POST", "/items", create)
        client = TestClient(ui.app)
        assert client.post("/items", json={"name": "a"}).json() == {"name": "a", "tags": []}
        assert client.post("/items", json={"tags": "nope"}).status_code == 422

        # Generating the schema validates it with the deferred models, the self-referencing Schema included
        body = ui.app.openapi()["paths"]["/items"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert body == {"$ref": "#/components/schemas/Item"}

        # Without the model_rebuild() calls the patch removes, every model still resolves its forward references
        openapi_models = [v for v in vars(models).values() if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == models.__name__]
        for model in openapi_models:
            model.model_rebuild()
            assert model.__pydantic_complete__, model.__name__
        assert models.Contact.model_validate({"name": "x"}).name == "x"
        """,
    )
