# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

from arduino.app_internal.core.module import (
    load_model_list,
    ModelEntry,
    ModelDeployment,
    ModelBrickConfig,
)

SAMPLE_MODELS_YAML = """
- "ei:efficientnet-b4":
    name: "General purpose object classification - EfficientNet-B4"
    description: "EfficientNetB4 is a machine learning model."
    metadata:
      requires_softmax_layer: true
      model_size_mb: 89
      source: "edgeimpulse"
      image-resolution: "380x380"
      hw_acceleration_backend: "qnn"
    supported_boards: ["ventunoq"]
    deployment:
      handler: "ei-handler"
      platforms:
        - ventunoq:
            variables:
              ei_project_id: 948887
              ei_impulse_id: 4
              models_repository: "models/edge-impulse"
              model_name: efficientnet-b4-qnn.eim
    bricks:
      - id: "arduino:image_classification"
        model_configuration:
          "EI_IMAGE_CLASSIFICATION_MODEL": "models/efficientnet-b4-qnn.eim"
- "genie:qwen3-4b":
    name: "Qwen3 4B"
    description: "A language model."
    metadata:
      model_size_mb: 2048
      source: "genie"
    supported_boards: ["ventunoq", "unoq"]
"""


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_returns_dict(mock_static_dir, tmp_path):
    """Test that load_model_list returns a dict keyed by model_id."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()

    assert result is not None
    assert isinstance(result, dict)
    assert "ei:efficientnet-b4" in result
    assert "genie:qwen3-4b" in result


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_model_entry_fields(mock_static_dir, tmp_path):
    """Test that ModelEntry fields are correctly populated."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    entry = result["ei:efficientnet-b4"]

    assert isinstance(entry, ModelEntry)
    assert entry.model_id == "ei:efficientnet-b4"
    assert entry.name == "General purpose object classification - EfficientNet-B4"
    assert entry.supported_boards == ["ventunoq"]
    assert entry.metadata["source"] == "edgeimpulse"
    assert entry.metadata["model_size_mb"] == 89


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_deployment(mock_static_dir, tmp_path):
    """Test that deployment info is correctly parsed."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    entry = result["ei:efficientnet-b4"]

    assert entry.deployment is not None
    assert isinstance(entry.deployment, ModelDeployment)
    assert entry.deployment.handler == "ei-handler"
    assert "ventunoq" in entry.deployment.platforms
    assert entry.deployment.platforms["ventunoq"]["variables"]["ei_project_id"] == 948887


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_bricks(mock_static_dir, tmp_path):
    """Test that bricks configuration is correctly parsed."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    entry = result["ei:efficientnet-b4"]

    assert len(entry.bricks) == 1
    brick = entry.bricks[0]
    assert isinstance(brick, ModelBrickConfig)
    assert brick.id == "arduino:image_classification"
    assert brick.model_configuration["EI_IMAGE_CLASSIFICATION_MODEL"] == "models/efficientnet-b4-qnn.eim"


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_no_deployment(mock_static_dir, tmp_path):
    """Test model entry without deployment section."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    entry = result["genie:qwen3-4b"]

    assert entry.deployment is None
    assert entry.bricks == []
    assert entry.supported_boards == ["ventunoq", "unoq"]


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_metadata_requires_softmax(mock_static_dir, tmp_path):
    """Test that metadata contains requires_softmax_layer for ei:efficientnet-b4."""
    model_file = tmp_path / "models-list.yaml"
    model_file.write_text(SAMPLE_MODELS_YAML, encoding="utf-8")
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    entry = result["ei:efficientnet-b4"]

    assert "requires_softmax_layer" in entry.metadata
    assert entry.metadata["requires_softmax_layer"] is True


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_returns_none_when_no_static_dir(mock_static_dir):
    """Test that None is returned when static assets directory is not found."""
    mock_static_dir.return_value = None

    result = load_model_list()
    assert result is None


@patch("arduino.app_internal.core.module.get_bricks_static_assets_directory")
def test_load_model_list_returns_none_when_file_missing(mock_static_dir, tmp_path):
    """Test that None is returned when models-list.yaml does not exist."""
    mock_static_dir.return_value = str(tmp_path)

    result = load_model_list()
    assert result is None
