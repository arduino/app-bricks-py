# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import sys
from setuptools.build_meta import build_wheel as _orig_build_wheel
from setuptools.build_meta import build_sdist as _orig_build_sdist
from setuptools.build_meta import build_editable as _orig_build_editable
from setuptools.build_meta import (
    get_requires_for_build_editable as _orig_get_requires_for_build_editable,
    prepare_metadata_for_build_editable as _orig_prepare_metadata_for_build_editable,
)
import subprocess
import shutil


def run_preprocessing(dev_mode: bool = False) -> None:
    if dev_mode:
        version = os.getenv("DEV_TAG_VERSION", "dev-latest")
    else:
        # Imported here: the build backend needs it, the unit tests importing this
        # module for its helpers do not have it installed.
        from setuptools_scm import get_version

        # Keep in sync with [tool.setuptools_scm] tag_regex in pyproject.toml: the explicit
        # kwargs here bypass the pyproject configuration entirely.
        version = get_version(
            version_scheme="only-version",
            local_scheme="no-local-version",
            tag_regex=r"^(?:ai|bricks|release)/(?P<version>v?\d+(?:\.\d+)*(?:rc\d+)?)$",
        )

    static_dir = "src/arduino/app_bricks/static"
    shutil.rmtree(static_dir, ignore_errors=True)

    print("################################### Docs generation #################################################################################")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    try:
        from docs_generator import runner

        runner.run_docs_generator()
    finally:
        sys.path.remove(project_root)

    print(f"################################## Provisioning static assets, version {version} - Dev Mode: {dev_mode} ##############################")
    subprocess.run(["arduino-bricks-release", "--static-dir", static_dir, "--version", version], check=True, cwd=os.getcwd())
    embed_pyright_rules(static_dir)


PYRIGHT_RULES_SOURCE = "pyright-rules.json"


def embed_pyright_rules(cache_folder_path: str) -> None:
    """Copy the pyright rules file into the static assets shipped in the wheel.

    The file describes how code using the library is type-checked (profiles for
    the library itself and for API users). It is maintained at the repository
    root and read from the wheel by App Lab and by the CI checks of both
    repositories, so it travels with every release.
    """
    print("################################## Embed pyright rules ##############################################################################")
    if not os.path.isfile(PYRIGHT_RULES_SOURCE):
        raise FileNotFoundError(f"{PYRIGHT_RULES_SOURCE} not found in the repository root")
    shutil.copy(PYRIGHT_RULES_SOURCE, os.path.join(cache_folder_path, os.path.basename(PYRIGHT_RULES_SOURCE)))


def build_wheel(wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None) -> str:
    return _orig_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict | None = None) -> str:
    dev_mode = False
    if config_settings and "build_type" in config_settings:
        dev_mode = config_settings["build_type"] == "dev"
    run_preprocessing(dev_mode)
    return _orig_build_sdist(sdist_directory, config_settings)


def build_editable(editable_build_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None) -> str:
    return _orig_build_editable(editable_build_directory, config_settings, metadata_directory)


def get_requires_for_build_editable(config_settings: dict | None = None) -> list[str]:
    return _orig_get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings: dict | None = None) -> str:
    return _orig_prepare_metadata_for_build_editable(metadata_directory, config_settings)
