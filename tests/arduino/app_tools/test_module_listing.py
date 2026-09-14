# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.version import __version__
from arduino.app_tools.module_listing import (
    RELEASE_VERSION_PLACEHOLDER,
    ArduinoBrick,
    resolve_release_version,
    save_compose_file,
    save_models_files,
    save_services_files,
)


def compose(image: str) -> str:
    return f"services:\n  runner:\n    image: ${{DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}}app-bricks/{image}\n"


def test_save_compose_file_stamps_release_version_in_every_variant(tmp_path):
    brick_dir = tmp_path / "foo"
    brick_dir.mkdir()
    (brick_dir / "brick_compose.yaml").write_text(compose(f"ei-models-runner:{RELEASE_VERSION_PLACEHOLDER}"))
    (brick_dir / "brick_compose.ventunoq.yaml").write_text(compose(f"ei-qnn-models-runner:{RELEASE_VERSION_PLACEHOLDER}"))
    brick = ArduinoBrick("arduino:foo", "Foo", "A brick", [], str(brick_dir), "")

    save_compose_file(brick, str(tmp_path / "out"), "1.2.3")

    out = tmp_path / "out" / "arduino" / "foo"
    assert (out / "brick_compose.yaml").read_text() == compose("ei-models-runner:1.2.3")
    assert (out / "brick_compose.ventunoq.yaml").read_text() == compose("ei-qnn-models-runner:1.2.3")


def test_save_compose_file_skips_bricks_without_compose(tmp_path):
    brick_dir = tmp_path / "bar"
    brick_dir.mkdir()
    brick = ArduinoBrick("arduino:bar", "Bar", "A brick", [], str(brick_dir), "")

    save_compose_file(brick, str(tmp_path / "out"), "1.2.3")

    assert not (tmp_path / "out").exists()


def test_save_services_files_stamps_only_compose_files(tmp_path):
    service_dir = tmp_path / "app_services" / "genie"
    service_dir.mkdir(parents=True)
    (service_dir / "service_compose.yaml").write_text(compose(f"genie-models-runner:{RELEASE_VERSION_PLACEHOLDER}"))
    (service_dir / "service_config.yaml").write_text(f"service_id: arduino:genie\nnote: {RELEASE_VERSION_PLACEHOLDER}\n")

    save_services_files(str(tmp_path / "app_services"), str(tmp_path / "out"), "1.2.3")

    out = tmp_path / "out" / "genie"
    assert (out / "service_compose.yaml").read_text() == compose("genie-models-runner:1.2.3")
    assert RELEASE_VERSION_PLACEHOLDER in (out / "service_config.yaml").read_text()


def test_save_models_files_stamps_release_version(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models-handlers.yaml").write_text(f"listing:\n  image: app-bricks/models-downloader:{RELEASE_VERSION_PLACEHOLDER}\n")
    (models_dir / "models-list.yaml").write_text("models: []\n")

    save_models_files(str(models_dir), str(tmp_path / "static"), "1.2.3")

    assert (tmp_path / "static" / "models-handlers.yaml").read_text() == "listing:\n  image: app-bricks/models-downloader:1.2.3\n"
    assert (tmp_path / "static" / "models-list.yaml").read_text() == "models: []\n"


def test_save_models_files_requires_models(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_models_files(str(tmp_path), str(tmp_path / "static"), "1.2.3")


def test_resolve_release_version_prefers_argument_then_environment_then_library_version(monkeypatch):
    monkeypatch.setenv("BRICKS_RELEASE_VERSION", "dev-branch")
    assert resolve_release_version("2.0.0") == "2.0.0"
    assert resolve_release_version() == "dev-branch"

    monkeypatch.delenv("BRICKS_RELEASE_VERSION")
    assert resolve_release_version() == __version__
