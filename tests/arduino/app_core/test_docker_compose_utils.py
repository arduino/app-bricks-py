# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import yaml
from arduino.app_internal.core.module import (
    parse_docker_compose_variable,
    load_module_supported_variables,
    get_brick_config_file,
    get_brick_compose_file,
    _update_compose_release_version,
    _update_compose_release_version_by_platform,
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
    with open(module_cfg, "r") as file:
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
        only_ai_containers=True,
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
        only_ai_containers=True,
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


def test_release_branch_tag_to_semver_ai():
    """Test updating from a branch-name tag (e.g. dev-next) to a semver in an AI compose file."""
    compose_file_path = "tests/arduino/app_core/brick_compose_ai_branch.yaml"
    release_version = "2.1.0"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "ei-models-runner:dev-next" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path,
        release_version=release_version,
        append_suffix=True,
        only_ai_containers=True,
    )
    with open(new_path, "r") as file:
        content = file.read()
        assert "ei-models-runner:2.1.0" in content
        assert "dev-next" not in content
    import os

    os.remove(new_path)


def test_release_branch_tag_to_branch_tag_ai():
    """Test updating from one branch-name tag (e.g. dev-next) to another (e.g. dev-latest) in an AI compose file."""
    compose_file_path = "tests/arduino/app_core/brick_compose_ai_branch.yaml"
    release_version = "dev-latest"
    with open(compose_file_path, "r") as file:
        content = file.read()
        assert "ei-models-runner:dev-next" in content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path,
        release_version=release_version,
        append_suffix=True,
        only_ai_containers=True,
    )
    with open(new_path, "r") as file:
        content = file.read()
        assert "ei-models-runner:dev-latest" in content
        assert "dev-next" not in content
    import os

    os.remove(new_path)


def test_release_no_runner_skipped():
    """Test that a compose file without any -runner image is left untouched when only_ai_containers=True."""
    compose_file_path = "tests/arduino/app_core/brick_compose_test_data.yaml"
    release_version = "3.0.0"
    with open(compose_file_path, "r") as file:
        original_content = file.read()
        assert "-runner" not in original_content
    new_path = _update_compose_release_version(
        compose_file_path=compose_file_path,
        release_version=release_version,
        append_suffix=True,
        only_ai_containers=True,
    )
    # Should return the original path unchanged (no .new file written)
    assert new_path == compose_file_path


# --- _update_compose_release_version_by_platform tests ---

BRICK_COMPOSE_CONTENT = """\
services:
  models-runner:
    image: ${DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}app-bricks/ei-models-runner:${APPSLAB_VERSION:-dev-latest}
    ports:
      - "${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"
"""

BRICK_COMPOSE_AI_CONTENT = """\
services:
  models-runner:
    image: ${DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}app-bricks/ei-models-runner:1.5.22
    ports:
      - "${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"
"""

SERVICE_COMPOSE_CONTENT = """\
services:
  my-service:
    image: ${DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}app-bricks/my-service-runner:${APPSLAB_VERSION:-dev-latest}
    ports:
      - "8080:8080"
"""

NO_RUNNER_COMPOSE_CONTENT = """\
services:
  dbstorage-influx:
    image: influxdb:2.7
    ports:
      - "8086:8086"
"""


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


def _read(path):
    with open(path, "r") as f:
        return f.read()


def test_by_platform_updates_all_brick_compose_variants(tmp_path):
    """Test that _update_compose_release_version_by_platform updates all brick_compose*.yaml files."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "brick_compose.ventunoq.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "brick_compose.unoq.yaml", BRICK_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="1.0.0",
    )

    for name in ["brick_compose.yaml", "brick_compose.ventunoq.yaml", "brick_compose.unoq.yaml"]:
        content = _read(tmp_path / name)
        assert ":1.0.0" in content
        assert "${APPSLAB_VERSION" not in content


def test_by_platform_updates_service_compose_files(tmp_path):
    """Test that _update_compose_release_version_by_platform updates service_compose*.yaml files."""
    _write(tmp_path / "service_compose.yaml", SERVICE_COMPOSE_CONTENT)
    _write(tmp_path / "service_compose.extra.yaml", SERVICE_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "service_compose.yaml"),
        release_version="2.0.0",
    )

    for name in ["service_compose.yaml", "service_compose.extra.yaml"]:
        content = _read(tmp_path / name)
        assert ":2.0.0" in content
        assert "${APPSLAB_VERSION" not in content


def test_by_platform_updates_mixed_brick_and_service_compose(tmp_path):
    """Test that both brick_compose and service_compose files are updated in the same directory."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "brick_compose.ventunoq.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "service_compose.yaml", SERVICE_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="3.0.0",
    )

    for name in ["brick_compose.yaml", "brick_compose.ventunoq.yaml", "service_compose.yaml"]:
        content = _read(tmp_path / name)
        assert ":3.0.0" in content


def test_by_platform_only_ai_containers(tmp_path):
    """Test only_ai_containers flag updates only -runner images across platform variants."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_AI_CONTENT)
    _write(tmp_path / "brick_compose.ventunoq.yaml", BRICK_COMPOSE_AI_CONTENT)
    # This file has no -runner image, should remain untouched
    _write(tmp_path / "brick_compose.unoq.yaml", NO_RUNNER_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="dev-latest",
        only_ai_containers=True,
    )

    for name in ["brick_compose.yaml", "brick_compose.ventunoq.yaml"]:
        content = _read(tmp_path / name)
        assert "ei-models-runner:dev-latest" in content
        assert "1.5.22" not in content

    # No runner file should be unchanged
    content = _read(tmp_path / "brick_compose.unoq.yaml")
    assert content == NO_RUNNER_COMPOSE_CONTENT


def test_by_platform_with_registry_override(tmp_path):
    """Test registry override is applied across all platform compose variants."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "brick_compose.ventunoq.yaml", BRICK_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="1.2.3",
        registry="arduino.io/",
    )

    for name in ["brick_compose.yaml", "brick_compose.ventunoq.yaml"]:
        content = _read(tmp_path / name)
        assert ":1.2.3" in content
        assert "${DOCKER_REGISTRY_BASE:-arduino.io/}" in content


def test_by_platform_ignores_non_compose_yaml_files(tmp_path):
    """Test that non-compose yaml files in the same directory are not touched."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "other_config.yaml", BRICK_COMPOSE_CONTENT)
    _write(tmp_path / "brick_config.yaml", NO_RUNNER_COMPOSE_CONTENT)

    original_other = _read(tmp_path / "other_config.yaml")
    original_config = _read(tmp_path / "brick_config.yaml")

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="4.0.0",
    )

    # brick_compose.yaml should be updated
    content = _read(tmp_path / "brick_compose.yaml")
    assert ":4.0.0" in content

    # other files should be untouched
    assert _read(tmp_path / "other_config.yaml") == original_other
    assert _read(tmp_path / "brick_config.yaml") == original_config


def test_by_platform_service_compose_only_ai_containers(tmp_path):
    """Test only_ai_containers flag with service_compose files containing -runner images."""
    _write(tmp_path / "service_compose.yaml", SERVICE_COMPOSE_CONTENT)
    _write(tmp_path / "service_compose.backup.yaml", NO_RUNNER_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "service_compose.yaml"),
        release_version="5.0.0",
        only_ai_containers=True,
    )

    # service_compose.yaml has a -runner image so it gets updated
    content = _read(tmp_path / "service_compose.yaml")
    assert "my-service-runner:5.0.0" in content

    # service_compose.backup.yaml has no -runner, should be untouched
    content = _read(tmp_path / "service_compose.backup.yaml")
    assert content == NO_RUNNER_COMPOSE_CONTENT


def test_by_platform_single_brick_compose_no_variants(tmp_path):
    """Test with only a single brick_compose.yaml and no platform variants."""
    _write(tmp_path / "brick_compose.yaml", BRICK_COMPOSE_CONTENT)

    _update_compose_release_version_by_platform(
        compose_file_path=str(tmp_path / "brick_compose.yaml"),
        release_version="6.0.0",
    )

    content = _read(tmp_path / "brick_compose.yaml")
    assert ":6.0.0" in content
    assert "${APPSLAB_VERSION" not in content
