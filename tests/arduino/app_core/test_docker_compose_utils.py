# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import yaml
from arduino.app_internal.core.module import (
    parse_docker_compose_variable,
    load_module_supported_variables,
    get_brick_config_file,
    get_brick_compose_file,
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
    assert len(discovered_vars) == 5
    for var in discovered_vars:
        if var.name == "APP_HOME":
            assert var.default_value == "."
        if var.name == "BIND_ADDRESS":
            assert var.default_value == "127.0.0.1"
        if var.name == "DB_USERNAME":
            assert var.default_value == "admin"
        if var.name == "DB_PASSWORD":
            assert var.default_value == "Arduino15"
        if var.name == "INFLUXDB_ADMIN_TOKEN":
            assert var.default_value == "392edbf2-b8a2-481f-979d-3f188b2c05f0"


def test_get_compose_file_dbstorage_tsstore():
    """Test getting the Docker Compose file for _InfluxDBHandler."""
    module_cfg = get_brick_config_file(_InfluxDBHandler)
    assert module_cfg is not None
    with open(module_cfg) as file:
        content = file.read()
        cfg = yaml.safe_load(content)
        assert cfg["id"] == "arduino:dbstorage_tsstore"
    compose_file = get_brick_compose_file(_InfluxDBHandler)
    assert compose_file is not None
    discovered_vars = load_module_supported_variables(compose_file)
    assert len(discovered_vars) == 5
    for var in discovered_vars:
        if var.name == "APP_HOME":
            assert var.default_value == "."
        if var.name == "BIND_ADDRESS":
            assert var.default_value == "127.0.0.1"
        if var.name == "DB_USERNAME":
            assert var.default_value == "admin"
        if var.name == "DB_PASSWORD":
            assert var.default_value == "Arduino15"
        if var.name == "INFLUXDB_ADMIN_TOKEN":
            assert var.default_value == "392edbf2-b8a2-481f-979d-3f188b2c05f0"
