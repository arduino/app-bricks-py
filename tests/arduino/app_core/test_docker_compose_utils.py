# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import yaml
from arduino.app_internal.core.module import (
    parse_docker_compose_variable,
    load_module_supported_variables,
    get_brick_config_file,
    get_brick_compose_file,
    _update_compose_release_version,
)
from arduino.app_bricks.dbstorage_tsstore import _InfluxDBHandler


def test_parse_docker_compose_variable():
    """Test parsing a single Docker Compose variable."""
    assert parse_docker_compose_variable("${DATABASE_HOST:-db}") == [("DATABASE_HOST", "db")]
    assert parse_docker_compose_variable("volume1") == "volume1"


def test_parse_docker_compose_multi_variable():
    """Test parsing multiple Docker Compose variables."""
    assert parse_docker_compose_variable("${DATABASE_HOST:-db}:${DATABASE_PORT:-5432}") == [
        ("DATABASE_HOST", "db"),
        ("DATABASE_PORT", "5432"),
    ]


def test_docker_compose_load_all_vars():
    """Test loading all variables from a Docker Compose file."""
    discovered_vars = load_module_supported_variables("tests/arduino/app_core/brick_compose_test_data.yaml")
    assert len(discovered_vars) == 6
    for var in discovered_vars:
        if var.name == "DATABASE_HOST":
            assert var.default_value == "db"
        elif var.name == "DATABASE_PORT":
            assert var.default_value == "8086"
        elif var.name == "BIND_ADDRESS":
            assert var.default_value == "127.0.0.1"
        elif var.name == "ADMIN_TOKEN":
            assert var.default_value == "392edbf2-b8a2-481f-979d-3f188b2c05f0"
        elif var.name == "USERNAME":
            assert var.default_value == "admin"
        elif var.name == "APP_HOME":
            assert var.default_value == "."


def test_get_compose_file_dbstorage_tsstore():
    """Test getting the Docker Compose file for _InfluxDBHandler."""
    module_cfg = get_brick_config_file(_InfluxDBHandler)
    assert module_cfg is not None
    with open(module_cfg, "r") as file:
        content = file.read()
        cfg = yaml.safe_load(content)
        assert cfg["id"] == "arduino:dbstorage_tsstore"
    compose_file = get_brick_compose_file(_InfluxDBHandler)
    assert compose_file is not None
    discovered_vars = load_module_supported_variables(compose_file)
    assert len(discovered_vars) == 6
    for var in discovered_vars:
        if var.name == "DOCKER_INFLUXDB_INIT_MODE":
            assert var.default_value == "setup"
        if var.name == "DOCKER_INFLUXDB_INIT_USERNAME":
            assert var.default_value == "admin"
        if var.name == "DOCKER_INFLUXDB_INIT_PASSWORD":
            assert var.default_value == "Arduino15"
        if var.name == "DOCKER_INFLUXDB_INIT_ORG":
            assert var.default_value == "arduino"
        if var.name == "DOCKER_INFLUXDB_INIT_BUCKET":
            assert var.default_value == "arduinostorage"
        if var.name == "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN":
            assert var.default_value == "392edbf2-b8a2-481f-979d-3f188b2c05f0"


def test_release_devlatest_ai():
    """Test updating the release version in a Docker Compose file for AI container with dev-latest."""
    compose_file_path = "tests/arduino/app_core/brick_compose_ai.yaml"
    release_version = "dev-latest"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "1.5.22" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path,
        release_version=release_version,
        append_suffix=True,
        only_ei_containers=True,
    )
    with open(new_path, "r") as file:
        content = file.read()
        assert ":dev-latest" in content
    import os

    os.remove(new_path)


def test_release_upgrade_version():
    """Test updating the release version in a Docker Compose file."""
    compose_file_path = "tests/arduino/app_core/brick_compose_appslab.yaml"
    release_version = "0.2.4"
    registry = "arduino.io/"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "${APPSLAB_VERSION:-dev-latest}" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path, release_version=release_version, append_suffix=True, registry=registry
    )
    with open(new_path, "r") as file:
        content = file.read()
        print(f"Updated compose file: {content}")
        assert ":0.2.4" in content
        assert "${DOCKER_REGISTRY_BASE:-" + registry + "}app-bricks/ei-models-runner:" in content
    import os

    os.remove(new_path)


def test_release_upgrade_ai():
    """Test updating the release version in a Docker Compose file for AI container with new version."""
    compose_file_path = "tests/arduino/app_core/brick_compose_ai.yaml"
    release_version = "2.0.0"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "1.5.22" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path,
        release_version=release_version,
        append_suffix=True,
        only_ei_containers=True,
    )
    with open(new_path, "r") as file:
        content = file.read()
        assert ":2.0.0" in content
    import os

    os.remove(new_path)


def test_release_upgrade_to_dev_latest():
    """Test updating the release version to dev-latest in a Docker Compose file."""
    compose_file_path = "tests/arduino/app_core/brick_compose_applab_released.yaml"
    release_version = "dev-latest"
    registry = "ghcr.io/arduino/"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "${APPSLAB_VERSION:-dev-latest}" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path, release_version=release_version, append_suffix=True, registry=registry
    )
    with open(new_path, "r") as file:
        content = file.read()
        print(f"Updated compose file: {content}")
        assert ":dev-latest" in content
        assert "${DOCKER_REGISTRY_BASE:-" + registry + "}app-bricks/ei-models-runner:" in content
    import os

    os.remove(new_path)
