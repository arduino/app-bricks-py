# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import logging
import site
import pathlib
import yaml
import json
import os
import sys
import argparse
import glob
import shutil
import time
from urllib.parse import urlparse
from arduino.version import __version__

logger = logging.getLogger(__name__)

RELEASE_VERSION_PLACEHOLDER = "__BRICKS_RELEASE_VERSION__"

editable_module_config = "direct_url.json"

config_file_name: str = "brick_config.yaml"
compose_config_file_name: str = "brick_compose.yaml"
compose_config_file_name_prefix: str = "brick_compose"
service_config_file_name: str = "service_config.yaml"
service_compose_config_file_name: str = "service_compose.yaml"
service_compose_config_file_name_prefix: str = "service_compose"
main_readme_file_name: str = "README.md"
examples_folder_name: str = "examples"


class EnvVariable:
    def __init__(self, name: str, description: str, default_value: str = None, hidden: bool = False, secret: bool = False) -> None:
        """Represents a variable in brick_config file."""
        self.name = name
        self.default_value = default_value
        self.description = description
        self.hidden = hidden
        self.secret = secret

    def to_dict(self) -> dict:
        """Converts the EnvVariable object to a dictionary."""
        dict_out = {
            "name": self.name,
            "default_value": self.default_value,
            "description": self.description,
            "hidden": self.hidden,
            "secret": self.secret,
        }
        if self.default_value is None or self.default_value == "":
            del dict_out["default_value"]
        if self.description is None or self.description == "":
            del dict_out["description"]
        if not self.hidden:
            del dict_out["hidden"]
        if not self.secret:
            del dict_out["secret"]
        return dict_out

    def __str__(self) -> str:
        return f"Name: {self.name}, Default value: {self.default_value}, Description: {self.description}"


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
        mount_devices_into_container: bool = False,
        requires_display: str = None,
        required_device_classes: list[str] = None,
        env_variables: dict[str, str] = None,
        supported_boards: list[str] = None,
        requires_services: list[str] = None,
        ai_frameworks_compatibility: list[str] = None,
        model_by_boards: list[dict[str, str]] = None,
        model_configuration_variables: list[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.brick_description = brick_description
        self.ports = ports
        self.path = fs_path
        self.compose_file: str | None = self.get_compose_file()
        self.readme_file: str | None = self.get_readme_file()
        self.model_name: str = model_name
        self.category = category
        self.mount_devices_into_container: bool = mount_devices_into_container
        self.requires_display: str | None = requires_display
        self.required_device_classes: list[str] | None = required_device_classes
        self.env_variables: dict[str, str] | None = env_variables
        self.supported_boards: list[str] | None = supported_boards
        self.requires_services: list[str] | None = requires_services
        self.ai_frameworks_compatibility: list[str] | None = ai_frameworks_compatibility
        self.model_by_boards: list[dict[str, str]] | None = model_by_boards
        self.model_configuration_variables: list[str] | None = model_configuration_variables

    def to_dict(self) -> dict:
        out_dict: dict = {
            "id": self.id,
            "name": self.name,
            "description": self.brick_description,
            "mount_devices_into_container": self.mount_devices_into_container,
            "ports": self.ports,
            "category": self.category,
        }
        if self.model_name and self.model_name != "":
            out_dict["model_name"] = self.model_name
        if self.requires_display:
            out_dict["requires_display"] = self.requires_display
        if self.required_device_classes:
            out_dict["required_devices"] = self.required_device_classes
        if self.supported_boards:
            out_dict["supported_boards"] = self.supported_boards
        if self.requires_services:
            out_dict["requires_services"] = self.requires_services
        if self.model_by_boards:
            out_dict["model_by_boards"] = self.model_by_boards
        if self.ai_frameworks_compatibility:
            out_dict["ai_frameworks_compatibility"] = self.ai_frameworks_compatibility
        if self.model_configuration_variables:
            out_dict["model_configuration_variables"] = self.model_configuration_variables
        if self.env_variables and len(self.env_variables) > 0:
            additional_vars: list[EnvVariable] = []
            for var in self.env_variables:
                name = var.get("name")
                description = var.get("description", "")
                default = var.get("default_value", "")
                hidden = var.get("hidden", False)
                secret = var.get("secret", False)
                additional_vars.append(EnvVariable(name, description, default, hidden, secret))
            if "variables" in out_dict:
                out_dict["variables"].extend([var.to_dict() for var in additional_vars])
            else:
                out_dict["variables"] = [var.to_dict() for var in additional_vars]
        return out_dict

    def get_compose_file(self) -> str | None:
        compose_file: pathlib.Path = pathlib.Path(self.path) / compose_config_file_name
        if compose_file.is_file():
            return str(compose_file)
        return None

    def get_readme_file(self) -> str | None:
        readme_file: pathlib.Path = pathlib.Path(self.path) / main_readme_file_name
        if readme_file.is_file():
            return str(readme_file)
        return None

    def __str__(self) -> str:
        return f"Name: {self.name}\nDescription: {self.brick_description}\nPath: {self.path}\nCompose file: {self.get_compose_file()}\n"


class ArduinoService:
    def __init__(
        self,
        service_id: str,
        name: str,
        brick_description: str,
        fs_path: str,
        category: str = "miscellaneous",
        env_variables: dict[str, str] = None,
        supported_boards: list[str] = None,
        root_path: str = None,
    ) -> None:
        self.service_id = service_id
        self.name = name
        self.brick_description = brick_description
        self.path = fs_path
        self.compose_file: str | None = self.get_compose_file()
        self.category = category
        self.env_variables: dict[str, str] | None = env_variables
        self.supported_boards: list[str] | None = supported_boards
        self.root_path = root_path

    def to_dict(self) -> dict:
        out_dict: dict = {
            "service_id": self.service_id,
            "name": self.name,
            "description": self.brick_description,
            "category": self.category,
        }
        if self.supported_boards:
            out_dict["supported_boards"] = self.supported_boards
        if self.root_path:
            out_dict["root_path"] = self.root_path

        if self.env_variables and len(self.env_variables) > 0:
            additional_vars: list[EnvVariable] = []
            for var in self.env_variables:
                name = var.get("name")
                description = var.get("description", "")
                default = var.get("default_value", "")
                hidden = var.get("hidden", False)
                secret = var.get("secret", False)
                additional_vars.append(EnvVariable(name, description, default, hidden, secret))
            if "variables" in out_dict:
                out_dict["variables"].extend([var.to_dict() for var in additional_vars])
            else:
                out_dict["variables"] = [var.to_dict() for var in additional_vars]
        return out_dict

    def get_compose_file(self) -> str | None:
        compose_file: pathlib.Path = pathlib.Path(self.path) / compose_config_file_name
        if compose_file.is_file():
            return str(compose_file)
        return None

    def __str__(self) -> str:
        return f"Name: {self.name}\nDescription: {self.brick_description}\nPath: {self.path}\nCompose file: {self.get_compose_file()}\n"


def find_config_yaml(root_path: str) -> tuple[list[ArduinoBrick], list[ArduinoService]]:
    """Scans all subfolders within the given root_path to find 'config.yaml'.

    Args:
        root_path (str or pathlib.Path): The root directory to scan.

    Returns:
        list: A list of paths to directories that contain 'config.yaml'.
    """
    discovered_modules: list[ArduinoBrick] = []
    discovered_services: list[ArduinoService] = []
    root_path_obj: pathlib.Path = pathlib.Path(root_path)

    if not root_path_obj.is_dir():
        return discovered_modules, discovered_services

    for item in root_path_obj.iterdir():
        if item.is_dir():
            if item.name == examples_folder_name:
                # Example apps may embed app-local bricks (bricks/<id>/brick_config.yaml
                # with a namespace-less id): they belong to the example only and must
                # not be indexed as global bricks.
                continue
            config_file: pathlib.Path = item / config_file_name
            service_config_file: pathlib.Path = item / service_config_file_name
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
                        supported_boards=config.get("supported_boards", None),
                        requires_services=config.get("requires_services", None),
                        ai_frameworks_compatibility=config.get("ai_frameworks_compatibility", None),
                        model_by_boards=config.get("model_by_boards", None),
                        model_configuration_variables=config.get("model_configuration_variables", None),
                    )
                    discovered_modules.append(mod)
                except yaml.YAMLError:
                    logger.error(f"Error: {config_file} is not a valid YAML file.")
            elif service_config_file.is_file():
                try:
                    config: dict = yaml.safe_load(service_config_file.read_text())
                    if "service_id" not in config or "name" not in config or "description" not in config:
                        continue

                    if "disabled" in config and config["disabled"]:
                        logger.debug(f"Module {config['service_id']} is disabled. Skipping it.")
                        continue

                    mod = ArduinoService(
                        config["service_id"],
                        config["name"],
                        config["description"],
                        str(service_config_file.parent),
                        config.get("category", None),
                        env_variables=config.get("variables", None),
                        supported_boards=config.get("supported_boards", None),
                        root_path=root_path,
                    )
                    discovered_services.append(mod)
                except yaml.YAMLError:
                    logger.error(f"Error: {service_config_file} is not a valid YAML file.")
            elif editable_module.is_file():
                try:
                    with open(editable_module) as editable_module_cfg:
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
                                sub_bricks, sub_services = find_config_yaml(local_file_path)
                                discovered_modules.extend(sub_bricks)
                                discovered_services.extend(sub_services)

                except json.JSONDecodeError:
                    logger.error(f"Error: {editable_module} is not a valid JSON file.")
            else:
                sub_bricks, sub_services = find_config_yaml(item)  # add any config.yaml files found in subdirectories.
                discovered_modules.extend(sub_bricks)
                discovered_services.extend(sub_services)

    return discovered_modules, discovered_services


def list_installed_packages_pkg_resources() -> tuple[dict[str, list[ArduinoBrick]], str]:
    """List all installed packages and find those containing 'brick_config.yaml'.
    Returns a dictionary where keys are package paths and values are lists of ArduinoBrick instances.
    """
    start = time.time() * 1000
    checked_paths: dict[str, list[ArduinoBrick]] = {}
    checked_svc_paths: dict[str, list[ArduinoService]] = {}

    # Check standard site-packages and user site-packages directories
    paths = set(site.getsitepackages())
    paths.add(site.getusersitepackages())
    for local_path in paths:
        if local_path is None or local_path == "":
            continue
        logger.debug(f"Checking local path: {local_path}")
        local_bricks, local_svc = find_config_yaml(local_path)
        checked_paths[local_path] = local_bricks
        checked_svc_paths[local_path] = local_svc

    # Search for app_services folder (nested inside an 'arduino' subfolder)
    services_folder = None
    for key in checked_svc_paths.keys():
        for svs in checked_svc_paths[key]:
            local_path = svs.root_path
            logger.info(f"Searching for app_services folder in root path: {local_path}")
            if local_path is None or local_path == "":
                continue
            if "app_services" in str(local_path):
                logger.info(f"Found app_services folder directly in: {local_path}")
                services_folder = local_path
                break

        if services_folder:
            break

    if services_folder is None:
        logger.error("ERROR: app_services folder not found in site-packages directories.")

    # Check application python home directory
    app_home = "/app/python"
    local_bricks, local_svc = find_config_yaml(app_home)
    if local_bricks and len(local_bricks) > 0:
        checked_paths[app_home] = local_bricks
    if local_svc and len(local_svc) > 0:
        checked_svc_paths[app_home] = local_svc

    end = time.time() * 1000
    logger.info(f"Module discovery took {end - start} ms")
    return checked_paths, services_folder


def _stamp_release_version(content: str, release_version: str) -> str:
    return content.replace(RELEASE_VERSION_PLACEHOLDER, release_version)


def resolve_release_version(version: str | None = None) -> str:
    """Return the version stamped into compose and models files.

    Precedence: the explicit argument, the BRICKS_RELEASE_VERSION environment variable, the installed library version.
    """
    if version:
        return version
    env_version = os.environ.get("BRICKS_RELEASE_VERSION")
    if env_version:
        return env_version
    return __version__


def save_compose_file(module: ArduinoBrick, output_dir: str, release_version: str) -> None:
    """Copy every brick_compose*.yaml of the module to the output directory, stamping the release version."""
    if not module.compose_file:
        return

    # We cannot save a folder containing the `:`, therefore we split and save it
    # with parent folder. Example: `arduino/object_detection` instead of `arduino:object_detection`
    module_name = "/".join(module.id.split(":"))
    output_folder: pathlib.Path = pathlib.Path(output_dir) / module_name
    output_folder.mkdir(parents=True, exist_ok=True)

    for compose_file in pathlib.Path(module.path).glob(f"{compose_config_file_name_prefix}*.yaml"):
        logger.info(f"Copying compose file {compose_file} for module {module.id}")
        output_file: pathlib.Path = output_folder / compose_file.name
        output_file.write_text(_stamp_release_version(compose_file.read_text(), release_version))


def save_readme_file(module: ArduinoBrick, output_dir: str) -> None:
    """Save the readme file to the specified output directory."""
    if not module.readme_file:
        return

    # We cannot save a folder containing the `:`, therefore we split and save it
    # with parent folder. Example: `arduino/object_detection` instead of `arduino:object_detection`
    module_name = "/".join(module.id.split(":"))
    output_folder: pathlib.Path = pathlib.Path(output_dir) / module_name
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(module.readme_file, output_folder / main_readme_file_name)


def save_services_files(services_folder: str | None, output_dir: str, release_version: str) -> None:
    """Copy the services folder to the output directory, stamping the release version into its compose files."""
    if not services_folder:
        return
    print(f"Saving services files from {services_folder} to {output_dir}...")
    shutil.copytree(services_folder, output_dir, dirs_exist_ok=True)
    for compose_file in pathlib.Path(output_dir).rglob(f"{service_compose_config_file_name_prefix}*.yaml"):
        compose_file.write_text(_stamp_release_version(compose_file.read_text(), release_version))


def save_models_files(models_dir: str, output_dir: str, release_version: str) -> None:
    """Copy the models-*.yaml files to the output directory, stamping the release version."""
    model_files = glob.glob(os.path.join(models_dir, "models-*.yaml"))
    if not model_files:
        raise FileNotFoundError(f"No models-*.yaml files found in {models_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for model_file in model_files:
        content = pathlib.Path(model_file).read_text()
        pathlib.Path(output_dir, os.path.basename(model_file)).write_text(_stamp_release_version(content, release_version))


def save_pyright_rules(rules_file: str, output_dir: str) -> None:
    """Copy the pyright rules file into the static assets, App Lab and the CI checks read it from the wheel."""
    if not os.path.isfile(rules_file):
        raise FileNotFoundError(f"{rules_file} not found, it is maintained at the repository root")
    shutil.copy(rules_file, os.path.join(output_dir, os.path.basename(rules_file)))


def save_api_docs_files(api_docs_dir: str, output_dir: str) -> None:
    """Copy the generated API docs to the output directory."""
    if not os.path.isdir(api_docs_dir):
        raise FileNotFoundError(f"API docs directory {api_docs_dir} not found, generate it first")
    shutil.copytree(api_docs_dir, output_dir, dirs_exist_ok=True)


def save_bricks_list(modules: dict[str, list[ArduinoBrick]], output_paths: list[str]) -> None:
    """Write the bricks list to every output path."""
    bricks = [module.to_dict() for module_list in modules.values() for module in module_list]
    content = yaml.dump({"bricks": bricks}, indent=2, default_flow_style=False, sort_keys=False, allow_unicode=True)
    for output_path in output_paths:
        pathlib.Path(output_path).write_text(content)


def library_provisioning(out_path: str, modules: dict[str, list[ArduinoBrick]], services_folder: str | None, release_version: str) -> None:
    """Write the compose files, READMEs and services of the discovered bricks under out_path."""
    print(f"Provisioning compose files into {out_path} for release version {release_version}")
    compose_output_dir = f"{out_path}/compose"
    services_output_dir = f"{out_path}/services/arduino"
    docs_output_dir = f"{out_path}/docs"
    for output_dir in (compose_output_dir, services_output_dir, docs_output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for module_list in modules.values():
        for module in module_list:
            save_compose_file(module, compose_output_dir, release_version)
            save_readme_file(module, docs_output_dir)

    save_services_files(services_folder, services_output_dir, release_version)


def release() -> None:
    """Provision the static assets bundled into the wheel: bricks list, models files, compose files, READMEs, services, API docs and pyright rules."""
    parser = argparse.ArgumentParser(description="Provision the static assets bundled into the Arduino App Bricks wheel.")
    parser.add_argument("-d", "--static-dir", type=str, required=True, help="Static assets directory to populate.")
    parser.add_argument("-m", "--models-dir", type=str, default="models", help="Directory holding the models-*.yaml files.")
    parser.add_argument("-a", "--api-docs-dir", type=str, default="docs", help="Directory holding the generated API docs.")
    parser.add_argument("-r", "--pyright-rules", type=str, default="pyright-rules.json", help="Pyright rules file to ship in the wheel.")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Release version stamped into compose and models files. Defaults to BRICKS_RELEASE_VERSION, then to the installed library version.",
    )
    args = parser.parse_args()

    release_version = resolve_release_version(args.version)
    discovered_modules, services_folder = list_installed_packages_pkg_resources()

    os.makedirs(args.static_dir, exist_ok=True)
    save_bricks_list(discovered_modules, [os.path.join(args.static_dir, "bricks-list.yaml")])
    save_models_files(args.models_dir, args.static_dir, release_version)
    library_provisioning(args.static_dir, discovered_modules, services_folder, release_version)
    save_api_docs_files(args.api_docs_dir, os.path.join(args.static_dir, "api-docs"))
    save_pyright_rules(args.pyright_rules, args.static_dir)


def main() -> None:
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

    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Release version stamped into compose files. Defaults to BRICKS_RELEASE_VERSION, then to the installed library version.",
    )

    args = parser.parse_args()

    discovered_modules, services_folder = list_installed_packages_pkg_resources()

    if args.provision_compose:
        composeout = args.compose_output or args.output
        library_provisioning(composeout, discovered_modules, services_folder, resolve_release_version(args.version))
        if args.output:
            print("Compose provisioning completed.")
            sys.exit(0)

    # List bricks and build the output structures
    print(f"Provisioning bricks and model lists...")
    if args.output:
        save_bricks_list(discovered_modules, [output_path.strip() for output_path in args.output.split(",")])

    if args.model_output:
        static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_bricks", "static")
        model_files = glob.glob(os.path.join(static_path, "models-*.yaml"))
        output_dir = os.path.dirname(args.model_output)
        if model_files:
            for model_path in model_files:
                shutil.copy(model_path, os.path.join(output_dir, os.path.basename(model_path)))
            # Copy api-docs as well
            api_docs_source = os.path.join(static_path, "api-docs")
            api_docs_destination = os.path.join(output_dir, "api-docs")
            if os.path.exists(api_docs_source):
                shutil.copytree(api_docs_source, api_docs_destination, dirs_exist_ok=True)
        else:
            print(f"No models-*.yaml files found in {static_path}. Skipping model copy.")


if __name__ == "__main__":
    main()
    sys.exit(0)
