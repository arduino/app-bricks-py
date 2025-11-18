# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
import yaml
import sys
from typing import List, Dict, Optional

application_config_file_name: str = "app.yaml"
config_file_name: str = "brick_config.yaml"
compose_config_file_name: str = "brick_compose.yaml"


def get_app_config() -> Optional[Dict]:
    """Gets app.yaml application configuration."""
    config_path = None
    app_root_dir = os.getenv("APP_HOME")
    if app_root_dir and app_root_dir != "":
        config_path = os.path.join(app_root_dir, application_config_file_name)
        if not os.path.exists(config_path):
            config_path = None

    if config_path is None:
        main_module = sys.modules["__main__"]
        if hasattr(main_module, "__file__"):
            main_path = os.path.abspath(main_module.__file__)
            app_root_dir = os.path.dirname(os.path.dirname(main_path))
            config_path = os.path.join(app_root_dir, application_config_file_name)
            if not os.path.exists(config_path):
                return None

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_content = yaml.safe_load(f)
            return config_content

    return None


def get_brick_config_file(cls) -> Optional[str]:
    """Gets the full path of the brick_config.yaml file."""
    return get_brick_linked_resource_file(cls, config_file_name)


def get_brick_compose_file(cls) -> Optional[str]:
    """Gets the full path of the brick_compose.yaml file, if present."""
    return get_brick_linked_resource_file(cls, compose_config_file_name)


def load_brick_compose_file(cls) -> Optional[Dict]:
    """Loads the brick_compose.yaml file and returns its content."""
    pathfile = get_brick_compose_file(cls)
    if pathfile:
        with open(pathfile) as f:
            compose_content = yaml.safe_load(f)
            return compose_content
    else:
        return None


def get_brick_linked_resource_file(cls, resource_file_name) -> Optional[str]:
    """Gets the full path to a config file in the directory containing a class."""
    try:
        module = cls.__module__
        if module == "__main__":
            directory_path = os.path.dirname(os.path.abspath(__file__))
        else:
            module_obj = __import__(module, fromlist=["__file__"])
            file_path = os.path.abspath(module_obj.__file__)
            directory_path = os.path.dirname(file_path)

        requested_path = os.path.join(directory_path, resource_file_name)
        if os.path.exists(requested_path):
            return requested_path
        else:
            return None
    except AttributeError:
        # Handle built-in classes or other cases where __file__ is not available
        return None
    except ModuleNotFoundError:
        return None


def parse_docker_compose_variable(variable_string) -> List[tuple[str, str]] | str:
    """Parses a Docker Compose-style environment variable string, including nested variables.

    Args:
        variable_string: The string to parse (e.g., "${DATABASE_HOST:-db}",
            "${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8086}:8086"), "${DATABASE_PASSWORD}".

    Returns:
        A list of tuple containing the variable name and the default value (if present), or the original
        string if parsing fails.
    """
    matches = re.findall(r"\${([^:]+)(:\-)?([^}]+)?}", variable_string)
    if matches:
        results = []
        for match in matches:
            if len(match) == 3:
                var_name = match[0]
                default_value = match[2] if match[2] else None
                results.append((var_name, default_value))
            elif len(match) == 1:
                var_name = match[0]
                default_value = None
                results.append((var_name, default_value))
        return results
    else:
        return variable_string


def _accumulate_docker_compose_variables(discovered_vars, value):
    if isinstance(value, str):
        tp = parse_docker_compose_variable(value)
        if tp and isinstance(tp, list):
            for t in tp:
                discovered_vars.append(t)
    elif isinstance(value, dict):
        for k, val in value.items():
            tp = parse_docker_compose_variable(val)
            if tp and isinstance(tp, list):
                for t in tp:
                    discovered_vars.append(t)
    elif isinstance(value, list):
        for val in value:
            tp = parse_docker_compose_variable(val)
            if tp and isinstance(tp, list):
                for t in tp:
                    discovered_vars.append(t)


class ModuleVariable:
    def __init__(self, name: str, description: str, default_value: str = None):
        """Represents a variable in a Docker Compose file."""
        self.name = name
        self.default_value = default_value
        self.description = description

    def to_dict(self) -> dict:
        """Converts the ModuleVarable object to a dictionary."""
        dict_out = {"name": self.name, "default_value": self.default_value, "description": self.description}
        if self.default_value is None or self.default_value == "":
            del dict_out["default_value"]
        if self.description is None or self.description == "":
            del dict_out["description"]
        return dict_out

    def __str__(self):
        return f"Name: {self.name}, Default value: {self.default_value}, Description: {self.description}"


class EnvVariable:
    def __init__(self, name: str, description: str, default_value: str = None):
        """Represents a variable in brick_config file."""
        self.name = name
        self.default_value = default_value
        self.description = description

    def to_dict(self) -> dict:
        """Converts the EnvVariable object to a dictionary."""
        dict_out = {"name": self.name, "default_value": self.default_value, "description": self.description}
        if self.default_value is None or self.default_value == "":
            del dict_out["default_value"]
        if self.description is None or self.description == "":
            del dict_out["description"]
        return dict_out

    def __str__(self):
        return f"Name: {self.name}, Default value: {self.default_value}, Description: {self.description}"


def load_module_supported_variables(file_path: str) -> Optional[List[ModuleVariable]]:
    """Loads a Docker Compose file and returns all supported variables with its default values and description.

    Returns:
        A list of ModuleVarable objects representing the variables found in the Docker Compose file.
    """
    try:
        with open(file_path, "r") as file:
            # Read the file content to get headers
            descriptions: Dict[str, str] = {}
            while True:
                line = file.readline()
                if not line:  # End of file
                    break

                line = line.rstrip("\n")
                if not line.startswith("#"):
                    continue
                pieces = line[1:].strip().split("=")
                if len(pieces) < 2:
                    continue
                descriptions[pieces[0].strip()] = pieces[1].strip()

            file.seek(0)  # Reset file pointer to the beginning
            content = file.read()
            docker_c = yaml.safe_load(content)

            discovered_vars = []
            if "services" in docker_c:
                for service in docker_c["services"]:
                    for key, value in docker_c["services"][service].items():
                        if isinstance(value, str):
                            _accumulate_docker_compose_variables(discovered_vars, value)
                        elif isinstance(value, list):
                            for v in value:
                                _accumulate_docker_compose_variables(discovered_vars, v)
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                _accumulate_docker_compose_variables(discovered_vars, v)

            if len(discovered_vars) > 0:
                out_vars = []
                discovered_vars = list(set(discovered_vars))
                for name, default_value in sorted(discovered_vars):
                    out_vars.append(ModuleVariable(name, descriptions.get(name, None), default_value))
                return out_vars

            return None
    except FileNotFoundError:
        return None


def resolve_address(host: str) -> str:
    """Resolve address substituting it in case of local/remote development."""
    remote_dev = os.getenv("REMOTE_DEV")
    local_dev = os.getenv("LOCAL_DEV", "false").lower()
    if local_dev == "true":
        return "127.0.0.1"
    elif remote_dev and remote_dev != "":
        return remote_dev
    else:
        return host


def _update_compose_release_version(
    compose_file_path: str,
    release_version: str,
    append_suffix: bool = False,
    only_ei_containers: bool = False,
    registry: str = None,
) -> str:
    """Updates the release version in the Docker Compose file."""
    with open(compose_file_path, "r") as file:
        content = file.read()

    print("Updating compose file:", compose_file_path)
    if only_ei_containers and "ei-models-runner" not in content:
        return compose_file_path

    # Replace the release version in the content

    updated_content = content

    if only_ei_containers:
        substitution = "ei-models-runner:" + release_version
        updated_content = re.sub(r"ei-models-runner:[0-9]+\.[0-9]+\.[0-9]+", substitution, updated_content)

    substitution = release_version
    updated_content = re.sub(r"\${APPSLAB_VERSION:\-([^}]+)?}", substitution, updated_content)
    updated_content = re.sub(r"\${APPSLAB_VERSION}", substitution, updated_content)

    if registry and registry != "":
        substitution = "${DOCKER_REGISTRY_BASE:-" + registry + "}"
        updated_content = re.sub(r"\${DOCKER_REGISTRY_BASE:\-([^}]+)?}", substitution, updated_content)
        updated_content = re.sub(r"\${DOCKER_REGISTRY_BASE}", substitution, updated_content)

    if append_suffix:
        compose_file_path = compose_file_path + ".new"
    with open(compose_file_path, "w") as file:
        file.write(updated_content)

    return compose_file_path
