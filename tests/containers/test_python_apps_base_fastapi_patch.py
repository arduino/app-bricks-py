# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""python-apps-base patches FastAPI at build time: the patch must apply to the version the image installs and keep FastAPI working."""

import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from importlib.metadata import version
from pathlib import Path

import fastapi
import pytest

ROOT = Path(__file__).parents[2]
IMAGE_DIR = ROOT / "containers" / "bricks" / "python-apps-base"
PATCH = IMAGE_DIR / "patches" / "fastapi-defer-openapi-model-build.patch"

PATCH_FLAGS = ["-p1", "--forward", "--fuzz=0", "--no-backup-if-mismatch"]
IMAGE_PATCH_COMMAND = " ".join(["patch", *PATCH_FLAGS, "-d", "/usr/local/lib/python3.13/site-packages", "-i", f"/tmp/patches/{PATCH.name}"])

PATCH_TOOL = shutil.which("patch")
needs_patch = pytest.mark.skipif(PATCH_TOOL is None, reason="needs GNU patch, as the image build")


def _locked_version(package: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return next(p["version"] for p in lock["package"] if p["name"] == package)


@pytest.fixture
def patched_site_packages(tmp_path: Path) -> Path:
    """The installed FastAPI, patched with the flags of the image build, in a directory that shadows it on PYTHONPATH."""
    assert version("fastapi") == _locked_version("fastapi"), "the venv is not on the FastAPI version of uv.lock, run `uv sync`"
    site_packages = tmp_path / "site-packages"
    shutil.copytree(Path(fastapi.__file__).parent, site_packages / "fastapi", ignore=shutil.ignore_patterns("__pycache__"))

    assert PATCH_TOOL is not None
    result = subprocess.run([PATCH_TOOL, *PATCH_FLAGS, "-d", str(site_packages), "-i", str(PATCH)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"the patch no longer applies to FastAPI {version('fastapi')}:\n{result.stdout}{result.stderr}"
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


def test_image_applies_the_patch_with_the_tested_flags_before_compiling():
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text()
    assert "COPY ./patches/*.patch /tmp/patches/" in dockerfile
    assert IMAGE_PATCH_COMMAND in dockerfile
    assert dockerfile.index(IMAGE_PATCH_COMMAND) < dockerfile.index("python -m compileall /usr/local/lib")


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
