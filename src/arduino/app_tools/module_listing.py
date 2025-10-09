# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import site
import pathlib
import yaml
import json
import os
import sys
import argparse
import shutil
import time
from urllib.parse import urlparse
from typing import List, Dict, Optional
from arduino.app_internal.core.module import (
    load_module_supported_variables,
    ModuleVariable,
    _update_compose_release_version,
    EnvVariable,
)
from arduino.app_utils import Logger

logger = Logger(__name__)

editable_module_config = "direct_url.json"

config_file_name: str = "brick_config.yaml"
compose_config_file_name: str = "brick_compose.yaml"
main_readme_file_name: str = "README.md"
examples_folder_name: str = "examples"


class ArduinoBrick:
    def __init__(
        self,
        id: str,
        name: str,
        brick_description: str,
        ports: list[int],
        fs_path: str,
        model_name: str,
        category: str = "miscellaneous",
        require_devices: bool = False,
        requires_display: str = None,
        required_device_classes: List[str] = None,
        env_variables: Dict[str, str] = None,
    ):
        self.id = id
        self.name = name
        self.brick_description = brick_description
        self.ports = ports
        self.path = fs_path
        self.compose_file: Optional[str] = self.get_compose_file()
        self.readme_file: Optional[str] = self.get_readme_file()
        self.require_container: bool = self.compose_file is not None
        self.docker_compose_variables: Optional[List[ModuleVariable]] = (
            load_module_supported_variables(self.compose_file) if self.require_container else None
        )
        self.model_name: str = model_name
        self.require_model: bool = model_name != ""
        self.category = category
        self.require_devices: bool = require_devices
        self.requires_display: Optional[str] = requires_display
        self.required_device_classes: Optional[List[str]] = required_device_classes
        self.env_variables: Optional[Dict[str, str]] = env_variables

    def to_dict(self) -> dict:
        out_dict: dict = {
            "id": self.id,
            "name": self.name,
            "description": self.brick_description,
            "require_container": self.require_container,
            "require_model": self.require_model,
            "require_devices": self.require_devices,  # Keep for backward compatibility. To be removed in future versions.
            "mount_devices_into_container": self.require_devices,
            "ports": self.ports,
            "category": self.category,
        }
        if self.require_model:
            out_dict["model_name"] = self.model_name
        if self.requires_display:
            out_dict["requires_display"] = self.requires_display
        if self.required_device_classes:
            out_dict["required_devices"] = self.required_device_classes

        if self.require_container:
            var_to_add: List[ModuleVariable] = []
            for var in self.docker_compose_variables:
                if var.name != "APPSLAB_VERSION" and var.name != "DOCKER_REGISTRY_BASE" and var.name != "BIND_ADDRESS" and var.name != "BIND_PORT":
                    var_to_add.append(var)
            vars_list: List[Dict] = [var.to_dict() for var in var_to_add]
            out_dict["variables"] = vars_list

        if self.env_variables and len(self.env_variables) > 0:
            additional_vars: List[EnvVariable] = []
            for var in self.env_variables:
                name = var.get("name")
                description = var.get("description", "")
                default = var.get("default_value", "")
                additional_vars.append(EnvVariable(name, description, default))
            if "variables" in out_dict:
                out_dict["variables"].extend([var.to_dict() for var in additional_vars])
            else:
                out_dict["variables"] = [var.to_dict() for var in additional_vars]
        return out_dict

    def get_compose_file(self) -> Optional[str]:
        compose_file: pathlib.Path = pathlib.Path(self.path) / compose_config_file_name
        if compose_file.is_file():
            return str(compose_file)
        return None

    def get_readme_file(self) -> Optional[str]:
        readme_file: pathlib.Path = pathlib.Path(self.path) / main_readme_file_name
        if readme_file.is_file():
            return str(readme_file)
        return None

    def __str__(self):
        return f"Name: {self.name}\nDescription: {self.brick_description}\nPath: {self.path}\nCompose file: {self.get_compose_file()}\n"


def find_config_yaml(root_path: str) -> List[ArduinoBrick]:
    """Scans all subfolders within the given root_path to find 'config.yaml'.

    Args:
        root_path (str or pathlib.Path): The root directory to scan.

    Returns:
        list: A list of paths to directories that contain 'config.yaml'.
    """
    discovered_modules: List[ArduinoBrick] = []
    root_path_obj: pathlib.Path = pathlib.Path(root_path)

    if not root_path_obj.is_dir():
        return discovered_modules

    for item in root_path_obj.iterdir():
        if item.is_dir():
            config_file: pathlib.Path = item / config_file_name
            editable_module: pathlib.Path = item / editable_module_config
            if config_file.is_file():
                try:
                    config: dict = yaml.safe_load(config_file.read_text())
                    if "id" not in config or "name" not in config or "description" not in config:
                        continue

                    if "disabled" in config and config["disabled"]:
                        logger.debug(f"Module {config['id']} is disabled. Skipping it.")
                        continue

                    mod = ArduinoBrick(
                        config["id"],
                        config["name"],
                        config["description"],
                        config.get("ports", []),
                        str(config_file.parent),
                        config.get("model", ""),
                        config.get("category", None),
                        config.get("mount_devices_into_container", False),
                        config.get("requires_display", None),
                        required_device_classes=config.get("required_devices", None),
                        env_variables=config.get("variables", None),
                    )
                    discovered_modules.append(mod)
                except yaml.YAMLError:
                    logger.error(f"Error: {config_file} is not a valid YAML file.")
            elif editable_module.is_file():
                try:
                    with open(editable_module, "r") as editable_module_cfg:
                        content: dict = json.load(editable_module_cfg)
                        if "url" in content and "dir_info" in content:
                            editable_c: dict = content["dir_info"]
                            if "editable" in editable_c and editable_c["editable"]:
                                url: str = content["url"]
                                parsed_url = urlparse(url)
                                local_file_path: str = parsed_url.path
                                # For Windows paths, the path from urlparse will have a leading slash that needs to be removed.
                                if os.name == "nt" and local_file_path.startswith("/"):
                                    local_file_path = local_file_path[1:]

                                local_file_path = pathlib.Path(local_file_path) / "src"
                                discovered_modules.extend(find_config_yaml(local_file_path))

                except json.JSONDecodeError:
                    logger.error(f"Error: {editable_module} is not a valid JSON file.")
            else:
                discovered_modules.extend(find_config_yaml(item))  # add any config.yaml files found in subdirectories.

    return discovered_modules


def list_installed_packages_pkg_resources() -> Dict[str, List[ArduinoBrick]]:
    """List all installed packages and find those containing 'brick_config.yaml'.
    Returns a dictionary where keys are package paths and values are lists of ArduinoBrick instances.
    """
    start = time.time() * 1000
    checked_paths: Dict[str, List[ArduinoBrick]] = {}

    # Check standard site-packages and user site-packages directories
    paths = set(site.getsitepackages())
    paths.add(site.getusersitepackages())
    for local_path in paths:
        if local_path is None or local_path == "":
            continue
        logger.debug(f"Checking local path: {local_path}")
        local_installed_modules = find_config_yaml(local_path)
        checked_paths[local_path] = local_installed_modules

    # Check application python home directory
    app_home: Optional[str] = os.getenv("APP_HOME", "/app/python")
    if app_home and app_home != "":
        local_installed_modules: List[ArduinoBrick] = find_config_yaml(app_home)
        if local_installed_modules and len(local_installed_modules) > 0:
            checked_paths[app_home] = local_installed_modules
    else:
        main_module = sys.modules["__main__"]
        if hasattr(main_module, "__file__"):
            app_home = os.path.dirname(os.path.abspath(main_module.__file__))
            local_installed_modules = find_config_yaml(app_home)
            if local_installed_modules and len(local_installed_modules) > 0:
                checked_paths[app_home] = local_installed_modules

    end = time.time() * 1000
    logger.info(f"Module discovery took {end - start} ms")
    return checked_paths


def save_compose_file(module: ArduinoBrick, output_dir: str, appslab_version: str):
    """Save the compose file to the specified output directory."""
    if not module.require_container:
        return

    # We cannot save a folder containing the `:`, therefore we split and save it
    # with parent folder. Example: `arduino/object_detection` instead of `arduino:object_detection`
    module_name = "/".join(module.id.split(":"))
    output_folder: pathlib.Path = pathlib.Path(output_dir) / module_name
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file_name: pathlib.Path = output_folder / compose_config_file_name

    with open(module.compose_file, "rb") as f_source, open(output_file_name, "wb") as f_dest:
        while True:
            chunk = f_source.read(2048)
            if not chunk:
                break
            f_dest.write(chunk)

    _update_compose_release_version(compose_file_path=output_file_name, release_version=appslab_version)


def save_readme_file(module: ArduinoBrick, output_dir: str):
    """Save the readme file to the specified output directory."""
    if not module.readme_file:
        return

    # We cannot save a folder containing the `:`, therefore we split and save it
    # with parent folder. Example: `arduino/object_detection` instead of `arduino:object_detection`
    module_name = "/".join(module.id.split(":"))
    output_folder: pathlib.Path = pathlib.Path(output_dir) / module_name
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file_name: pathlib.Path = output_folder / main_readme_file_name

    with open(module.readme_file, "rb") as f_source, open(output_file_name, "wb") as f_dest:
        while True:
            chunk = f_source.read(2048)
            if not chunk:
                break
            f_dest.write(chunk)


def save_api_docs_files(output_dir: str):
    """Save the API docs files to the specified output directory."""
    shutil.copytree("docs/", output_dir, dirs_exist_ok=True)


def save_examples_files(module: ArduinoBrick, output_dir: str):
    """Save the examples files to the specified output directory."""
    if not module.readme_file:
        return

    # We cannot save a folder containing the `:`, therefore we split and save it
    # with parent folder. Example: `arduino/object_detection` instead of `arduino:object_detection`
    module_name = "/".join(module.id.split(":"))
    output_folder: pathlib.Path = pathlib.Path(output_dir) / module_name
    input_folder: pathlib.Path = pathlib.Path(module.path) / examples_folder_name
    if input_folder.is_dir():
        shutil.copytree(input_folder, output_folder, dirs_exist_ok=True)


def library_provisioning(out_path: str = None, modules: Dict[str, List[ArduinoBrick]] = None, buildtime: bool = False):
    print(f"Provisioning compose files for app execution and bricks documentation. File: {out_path}")
    try:
        from arduino._version import __version__ as arduino_bricks_version
    except ImportError:
        logger.error("Error: AppLab version not found. 'appslab._version' module is not available.")
        sys.exit(1)

    compose_output_dir = f"{out_path}/compose"
    docs_output_dir = f"{out_path}/docs"
    api_docs_output_dir = f"{out_path}/api-docs"
    examples_output_dir = f"{out_path}/examples"
    os.makedirs(compose_output_dir, exist_ok=True)
    os.makedirs(docs_output_dir, exist_ok=True)
    os.makedirs(api_docs_output_dir, exist_ok=True)
    os.makedirs(examples_output_dir, exist_ok=True)

    for path, module_list in modules.items():
        for module in module_list:
            save_compose_file(module, compose_output_dir, arduino_bricks_version)
            save_readme_file(module, docs_output_dir)
            save_examples_files(module, examples_output_dir)

    # Save API docs files
    if buildtime:
        print(f"Saving API docs files... buildtime: {buildtime}")
        save_api_docs_files(api_docs_output_dir)


def release():
    discovered_modules = list_installed_packages_pkg_resources()

    parser = argparse.ArgumentParser(description="Process AppLab modules release.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output file path list. If not provided, the output will be printed to the console.",
    )
    parser.add_argument("-v", "--version", type=str, default=None, help="Release version.")
    parser.add_argument("-d", "--dev", action="store_true", help="Development mode.")
    parser.add_argument("-r", "--registry", type=str, default=None, help="Docker registry override.")

    args = parser.parse_args()

    if args.version is None or args.version == "":
        logger.error("Error: Release version is required.")
        sys.exit(1)

    registry = None
    if args.registry is not None and args.registry != "":
        registry = args.registry

    arduino_bricks_version = args.version
    update_ei_containers = False
    if args.dev is not None and args.dev:
        logger.warning("Development mode enabled. Using 'dev-latest' as the version.")
        arduino_bricks_version = "dev-latest"
        update_ei_containers = True

    modules = []
    for path, module_list in discovered_modules.items():
        for module in module_list:
            modules.append(module.to_dict())
            # Update the compose file with the release version
            if module.require_container:
                print(f"Processing compose file {module.compose_file} for arduino bricks version {arduino_bricks_version}")
                _update_compose_release_version(
                    compose_file_path=module.compose_file,
                    release_version=arduino_bricks_version,
                    append_suffix=False,
                    only_ei_containers=update_ei_containers,
                    registry=registry,
                )

    mod_structure = {
        "bricks": modules,
    }
    mod_string = yaml.dump(mod_structure, indent=2, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output and args.output != "":
        for output_path in args.output.split(","):
            with open(output_path.strip(), "w") as f:
                f.write(mod_string)
    else:
        print(mod_string)


def update_ai_container_references():
    discovered_modules = list_installed_packages_pkg_resources()

    parser = argparse.ArgumentParser(description="Update AI container references.")
    parser.add_argument("-v", "--version", type=str, default=None, help="Release version.")

    parser.add_argument("-r", "--registry", type=str, default=None, help="Docker registry override.")

    args = parser.parse_args()

    if args.version is None or args.version == "":
        logger.error("Error: Release version is required.")
        sys.exit(1)

    registry = None
    if args.registry is not None and args.registry != "":
        registry = args.registry

    arduino_bricks_version = args.version

    modules = []
    for path, module_list in discovered_modules.items():
        for module in module_list:
            modules.append(module.to_dict())
            # Update the compose file with the release version
            if module.require_container:
                _update_compose_release_version(
                    compose_file_path=module.compose_file,
                    release_version=arduino_bricks_version,
                    append_suffix=False,
                    only_ei_containers=True,
                    registry=registry,
                )


def main():
    parser = argparse.ArgumentParser(description="Process AppLab modules.")

    parser.add_argument("-p", "--provision-compose", action="store_true", help="Provision compose files for app execution.")

    parser.add_argument("-o", "--output", type=str, help="Output path")

    parser.add_argument("-c", "--compose-output", type=str, help="Compose output path")

    parser.add_argument(
        "-m",
        "--model-output",
        type=str,
        default="/app/.cache/models-list.yaml",
        help="Optional models output file path.",
    )

    parser.add_argument("-b", "--buildtime", action="store_true", help="Buildtime execution.")

    args = parser.parse_args()

    discovered_modules = list_installed_packages_pkg_resources()

    modules = []
    imported_modules = []
    for path, module_list in discovered_modules.items():
        for module in module_list:
            if module.id in imported_modules:
                continue
            modules.append(module.to_dict())
            imported_modules.append(module.id)

    if args.provision_compose:
        composeout = args.output
        if args.compose_output is not None and args.compose_output != "":
            composeout = args.compose_output
        # Provision compose files for app execution and bricks documentation
        library_provisioning(composeout, discovered_modules, args.buildtime)
        if args.buildtime or len(args.output) > 0:
            print("Compose provisioning completed.")
            sys.exit(0)

    # List bricks and build the output structures
    print(f"Provisioning bricks and model lists...")
    mod_structure = {
        "bricks": modules,
    }

    mod_string = yaml.dump(mod_structure, indent=2, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output and args.output != "":
        for output_path in args.output.split(","):
            with open(output_path.strip(), "w") as f:
                f.write(mod_string)

    if args.model_output and args.model_output != "":
        import inspect

        logger_class = type(logger)
        logger_file_path = inspect.getfile(logger_class)
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(logger_file_path))),
            "app_bricks",
            "static",
            "models-list.yaml",
        )
        exists = os.path.exists(model_path)
        if exists:
            shutil.copy(model_path, args.model_output)
        else:
            print(f"Model path: {model_path} does not exist. Skipping model copy.")


if __name__ == "__main__":
    main()
    sys.exit(0)
