# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import sys
from setuptools.build_meta import build_wheel as _orig_build_wheel
from setuptools.build_meta import build_sdist as _orig_build_sdist
from setuptools.build_meta import build_editable as _orig_build_editable
from setuptools_scm import get_version
import subprocess
import shutil


def run_preprocessing(dev_mode: bool = False) -> None:
    registry = os.getenv("PUBLIC_IMAGE_REGISTRY_BASE", None)
    if dev_mode:
        version = "dev-latest"
    else:
        version = get_version(
            version_scheme="only-version",
            local_scheme="no-local-version",
            tag_regex="^(ai|release)/(?P<version>[0-9.]+)$",
        )

    cache_folder_path = "src/arduino/app_bricks/static"
    if os.path.exists(cache_folder_path) and os.path.isdir(cache_folder_path):
        shutil.rmtree(cache_folder_path)
    os.makedirs(cache_folder_path, exist_ok=True)

    try:
        print(f"################################## Building bricks list Version: {version} - Dev Mode: {dev_mode} ##################################")
        cmd = ["arduino-bricks-release", "-o", f"{cache_folder_path}/bricks-list.yaml", "--version", f"{version}"]
        if registry:
            cmd.append("--registry")
            cmd.append(registry)
        if dev_mode:
            cmd.append("--dev")

        subprocess.run(cmd, check=True, cwd=os.getcwd())
    except Exception as e:
        print(f"Error: {e}.")
        raise

    try:
        print(f"################################## Pre-provision bricks list #######################################################################")
        cmd = ["arduino-bricks-list-modules", "-p", "-b", "-c", f"{cache_folder_path}"]
        subprocess.run(cmd, check=True, cwd=os.getcwd())
    except Exception as e:
        print(f"Error: {e}.")
        raise

    try:
        print(f"################################## Embed models list ###############################################################################")
        shutil.copyfile("models/models-list.yaml", f"{cache_folder_path}/models-list.yaml")
    except Exception as e:
        print(f"Error: {e}.")
        raise

    try:
        print("################################### Docs generation #################################################################################")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        print(f"Project root: {project_root}")
        if project_root not in sys.path:
            print(f"Adding project root to sys.path: {project_root}")
        sys.path.insert(0, project_root)
        from docs_generator import runner

        runner.run_docs_generator()
    except Exception as e:
        print(f"Error while generating docs: {e}.")
        raise
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _orig_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    dev_mode = False
    if config_settings and "build_type" in config_settings:
        dev_mode = config_settings["build_type"] == "dev"
    run_preprocessing(dev_mode)
    return _orig_build_sdist(sdist_directory, config_settings)


def build_editable(editable_build_directory, config_settings=None, metadata_directory=None):
    return _orig_build_editable(editable_build_directory, config_settings, metadata_directory)
