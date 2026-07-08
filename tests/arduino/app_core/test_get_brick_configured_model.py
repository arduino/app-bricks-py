# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch

from arduino.app_internal.core.module import get_brick_configured_model


BRICK_ID = "arduino:llm"
MODEL_FROM_APP_YAML = "genie:qwen3-4b"
MODEL_FROM_BRICK_CONFIG = "genie:default-model"
MODEL_FROM_BOARD = "genie:board-specific-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app_cfg(brick_id: str, model: str) -> dict:
    """Minimal app.yaml structure that maps brick_id -> model."""
    return {"bricks": [{brick_id: {"model": model}}]}


def _brick_cfg_with_model(model: str) -> dict:
    return {"model": model}


def _brick_cfg_with_model_by_boards(board: str, model: str) -> dict:
    return {"model_by_boards": [{"platform": board, "model": model}]}


def _brick_cfg_with_model_and_boards(board: str, board_model: str, default_model: str) -> dict:
    return {
        "model_by_boards": [{"platform": board, "model": board_model}],
        "model": default_model,
    }


# ---------------------------------------------------------------------------
# Model from app.yaml
# ---------------------------------------------------------------------------


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_returned_from_app_yaml(mock_app_cfg):
    mock_app_cfg.return_value = _app_cfg(BRICK_ID, MODEL_FROM_APP_YAML)

    result = get_brick_configured_model(BRICK_ID)

    assert result == MODEL_FROM_APP_YAML


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_app_yaml_ignores_brick_config(mock_app_cfg):
    """app.yaml takes precedence over brick_config."""
    mock_app_cfg.return_value = _app_cfg(BRICK_ID, MODEL_FROM_APP_YAML)

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model("some-other-model"))

    assert result == MODEL_FROM_APP_YAML


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_app_yaml_wrong_brick_id_not_matched(mock_app_cfg):
    """If app.yaml contains a different brick_id, it should not match."""
    mock_app_cfg.return_value = _app_cfg("arduino:vlm", MODEL_FROM_APP_YAML)

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model(MODEL_FROM_BRICK_CONFIG))

    assert result == MODEL_FROM_BRICK_CONFIG


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_app_yaml_entry_missing_model_key(mock_app_cfg):
    """Brick entry present but no 'model' key → fall through to brick_config."""
    mock_app_cfg.return_value = {"bricks": [{BRICK_ID: {"other_key": "value"}}]}

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model(MODEL_FROM_BRICK_CONFIG))

    assert result == MODEL_FROM_BRICK_CONFIG


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_app_yaml_mixed_bricks_list(mock_app_cfg):
    """app.yaml with a mix of bare brick names and brick dicts with only 'variables'.

    Matches the pattern:
        bricks:
          - arduino:web_ui
          - arduino:llm:
              variables:
                BIND_ADDRESS: 0.0.0.0

    No 'model' key is present → should fall through to brick_config.
    """
    mock_app_cfg.return_value = {
        "name": "Edge AI Assistant",
        "icon": "💬",
        "description": "Chatbot powered by a local LLM",
        "bricks": [
            "arduino:web_ui",
            {BRICK_ID: {"variables": {"BIND_ADDRESS": "0.0.0.0"}}},
        ],
    }

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model(MODEL_FROM_BRICK_CONFIG))

    assert result == MODEL_FROM_BRICK_CONFIG


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_app_yaml_mixed_bricks_list_with_model(mock_app_cfg):
    """Same mixed list as above but the llm brick also declares a model."""
    mock_app_cfg.return_value = {
        "bricks": [
            "arduino:web_ui",
            {BRICK_ID: {"model": MODEL_FROM_APP_YAML, "variables": {"BIND_ADDRESS": "0.0.0.0"}}},
        ],
    }

    result = get_brick_configured_model(BRICK_ID)

    assert result == MODEL_FROM_APP_YAML


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_bare_string_brick_entries_are_skipped(mock_app_cfg):
    """Bare string entries (e.g. 'arduino:web_ui') must not cause errors and are ignored."""
    mock_app_cfg.return_value = {
        "bricks": [
            "arduino:web_ui",
            "arduino:audio",
        ],
    }

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model(MODEL_FROM_BRICK_CONFIG))

    assert result == MODEL_FROM_BRICK_CONFIG


# ---------------------------------------------------------------------------
# Model from brick_config – model_by_boards
# ---------------------------------------------------------------------------


@patch("arduino.app_internal.core.module.get_board_name")
@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_model_by_boards_matching_platform(mock_app_cfg, mock_board):
    mock_app_cfg.return_value = None
    mock_board.return_value = "ventunoq"

    brick_cfg = _brick_cfg_with_model_by_boards("ventunoq", MODEL_FROM_BOARD)
    result = get_brick_configured_model(BRICK_ID, brick_config=brick_cfg)

    assert result == MODEL_FROM_BOARD


@patch("arduino.app_internal.core.module.get_board_name")
@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_model_by_boards_no_matching_platform_falls_back_to_model(mock_app_cfg, mock_board):
    """When board doesn't match any entry in model_by_boards, fall back to 'model' key."""
    mock_app_cfg.return_value = None
    mock_board.return_value = "unoq"

    brick_cfg = _brick_cfg_with_model_and_boards("ventunoq", MODEL_FROM_BOARD, MODEL_FROM_BRICK_CONFIG)
    result = get_brick_configured_model(BRICK_ID, brick_config=brick_cfg)

    assert result == MODEL_FROM_BRICK_CONFIG


@patch("arduino.app_internal.core.module.get_board_name")
@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_model_by_boards_multiple_entries_selects_correct(mock_app_cfg, mock_board):
    mock_app_cfg.return_value = None
    mock_board.return_value = "unoq"

    brick_cfg = {
        "model_by_boards": [
            {"platform": "ventunoq", "model": "genie:ventunoq-model"},
            {"platform": "unoq", "model": "genie:unoq-model"},
        ]
    }
    result = get_brick_configured_model(BRICK_ID, brick_config=brick_cfg)

    assert result == "genie:unoq-model"


# ---------------------------------------------------------------------------
# Model from brick_config – flat 'model' key
# ---------------------------------------------------------------------------


@patch("arduino.app_internal.core.module.get_app_config")
def test_model_from_brick_config_flat_model(mock_app_cfg):
    mock_app_cfg.return_value = None

    result = get_brick_configured_model(BRICK_ID, brick_config=_brick_cfg_with_model(MODEL_FROM_BRICK_CONFIG))

    assert result == MODEL_FROM_BRICK_CONFIG


# ---------------------------------------------------------------------------
# No model found
# ---------------------------------------------------------------------------


@patch("arduino.app_internal.core.module.get_app_config")
def test_returns_none_when_no_app_config_and_no_brick_config(mock_app_cfg):
    mock_app_cfg.return_value = None

    result = get_brick_configured_model(BRICK_ID)

    assert result is None


@patch("arduino.app_internal.core.module.get_app_config")
def test_returns_none_when_app_config_has_no_bricks_key(mock_app_cfg):
    mock_app_cfg.return_value = {"other_key": "value"}

    result = get_brick_configured_model(BRICK_ID)

    assert result is None


@patch("arduino.app_internal.core.module.get_board_name")
@patch("arduino.app_internal.core.module.get_app_config")
def test_returns_none_when_model_by_boards_has_no_match_and_no_model_key(mock_app_cfg, mock_board):
    mock_app_cfg.return_value = None
    mock_board.return_value = "unknown-board"

    brick_cfg = {"model_by_boards": [{"platform": "ventunoq", "model": MODEL_FROM_BOARD}]}
    result = get_brick_configured_model(BRICK_ID, brick_config=brick_cfg)

    assert result is None


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_raises_value_error_for_none_brick_id():
    with pytest.raises(ValueError):
        get_brick_configured_model(None)


def test_raises_value_error_for_empty_brick_id():
    with pytest.raises(ValueError):
        get_brick_configured_model("")


def test_raises_value_error_for_blank_brick_id():
    with pytest.raises(ValueError):
        get_brick_configured_model("   ")
