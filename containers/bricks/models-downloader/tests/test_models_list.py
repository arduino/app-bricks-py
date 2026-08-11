# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``common/models_list.py``."""

import os

import pytest
import yaml

from common.models_list import find_matching_model, find_model_size_mb, load_models_list


# Path to the real models-list.yaml. It lives in the repository root, not in the
# container directory: CI copies it in right before ``docker build``.
REAL_MODELS_LIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "models-list.yaml"))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
AI_HUB_ENTRY = {
    "genie:qwen3_4b_instruct_2507": {
        "name": "Qwen 3-4B Instruct",
        "deployment": {
            "handler": "ai-hub-handler",
            "platforms": [
                {
                    "ventunoq": {
                        "variables": {
                            "model_type": "genie",
                            "model_name": "qwen3_4b_instruct_2507",
                            "models_repository": "genai",
                            "model_directory": "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs8275",
                            "quantization": "w4a16",
                            "chipset": "qualcomm-qcs8275",
                            "version": "0.51.0",
                        }
                    }
                }
            ],
        },
        "metadata": {"model_size_mb": 3039, "source": "qualcomm-ai-hub"},
    }
}

EI_ENTRY = {
    "ei:efficientnet-b4": {
        "name": "EfficientNet-B4",
        "deployment": {
            "handler": "ei-handler",
            "platforms": [
                {
                    "ventunoq": {
                        "variables": {
                            # ints in YAML, strings in the environment
                            "ei_project_id": 948887,
                            "ei_impulse_id": 10,
                            "models_repository": "edge-impulse",
                            "model_name": "efficientnet-b4-qnn.eim",
                            "target": "runner-linux-aarch64-qnn",
                        }
                    }
                }
            ],
        },
    }
}

# Same variables declared twice, once per board — as models-list.yaml does for
# llamacpp:gemma-3-1b-it-Q4_0.
MULTI_PLATFORM_ENTRY = {
    "llamacpp:gemma-3-1b-it-Q4_0": {
        "deployment": {
            "handler": "hf-handler",
            "platforms": [
                {"ventunoq": {"variables": {"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"}}},
                {"unoq": {"variables": {"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"}}},
            ],
        }
    }
}

PRE_LOADED_ENTRY = {"face-detection": {"deployment": {"handler": "ei-handler", "pre-loaded": True}}}


def _env(variables):
    """Environment as the container sees it: every value a string."""
    return {key: str(value) for key, value in variables.items()}


AI_HUB_ENV = _env(AI_HUB_ENTRY["genie:qwen3_4b_instruct_2507"]["deployment"]["platforms"][0]["ventunoq"]["variables"])
EI_ENV = _env(EI_ENTRY["ei:efficientnet-b4"]["deployment"]["platforms"][0]["ventunoq"]["variables"])


# --------------------------------------------------------------------------- #
# find_matching_model
# --------------------------------------------------------------------------- #
def test_match_single_platform():
    model_id, model_data, platform = find_matching_model([AI_HUB_ENTRY], AI_HUB_ENV)
    assert model_id == "genie:qwen3_4b_instruct_2507"
    assert model_data["name"] == "Qwen 3-4B Instruct"
    assert platform == "ventunoq"


def test_match_coerces_int_variables():
    assert EI_ENV["ei_project_id"] == "948887"
    model_id, _data, _platform = find_matching_model([EI_ENTRY], EI_ENV)
    assert model_id == "ei:efficientnet-b4"


def test_repeated_identical_platforms_are_not_ambiguous():
    env = _env({"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"})
    model_id, _data, _platform = find_matching_model([MULTI_PLATFORM_ENTRY], env)
    assert model_id == "llamacpp:gemma-3-1b-it-Q4_0"


def test_board_selects_platform():
    env = _env({"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"})
    _model_id, _data, platform = find_matching_model([MULTI_PLATFORM_ENTRY], env, board="unoq")
    assert platform == "unoq"


def test_unknown_board_falls_back_to_any_platform():
    env = _env({"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"})
    model_id, _data, _platform = find_matching_model([MULTI_PLATFORM_ENTRY], env, board="nonexistent")
    assert model_id == "llamacpp:gemma-3-1b-it-Q4_0"


def test_missing_variable_in_env_does_not_match():
    env = dict(AI_HUB_ENV)
    del env["chipset"]
    assert find_matching_model([AI_HUB_ENTRY], env) == (None, None, None)


def test_different_variable_value_does_not_match():
    env = dict(AI_HUB_ENV, version="0.57.1")
    assert find_matching_model([AI_HUB_ENTRY], env) == (None, None, None)


def test_extra_env_keys_are_ignored():
    env = dict(AI_HUB_ENV, PATH="/usr/bin", HOME="/home/arduino", BOARD_NAME="ventunoq")
    model_id, _data, _platform = find_matching_model([AI_HUB_ENTRY], env)
    assert model_id == "genie:qwen3_4b_instruct_2507"


def test_more_specific_match_wins():
    broad = {"llamacpp:any": {"deployment": {"platforms": [{"ventunoq": {"variables": {"models_repository": "llamacpp"}}}]}}}
    env = _env({"model_url": "https://hf/repo/g.gguf", "models_repository": "llamacpp"})
    model_id, _data, _platform = find_matching_model([broad, MULTI_PLATFORM_ENTRY], env)
    assert model_id == "llamacpp:gemma-3-1b-it-Q4_0"


def test_truly_ambiguous_returns_none():
    variables = {"models_repository": "llamacpp", "model_url": "https://hf/repo/g.gguf"}
    first = {"llamacpp:a": {"deployment": {"platforms": [{"ventunoq": {"variables": dict(variables)}}]}}}
    second = {"llamacpp:b": {"deployment": {"platforms": [{"ventunoq": {"variables": dict(variables)}}]}}}
    assert find_matching_model([first, second], _env(variables)) == (None, None, None)


def test_pre_loaded_entries_are_skipped():
    assert find_matching_model([PRE_LOADED_ENTRY], {"anything": "goes"}) == (None, None, None)


def test_empty_variables_never_match():
    entry = {"x:y": {"deployment": {"platforms": [{"ventunoq": {"variables": {}}}]}}}
    assert find_matching_model([entry], {"model_name": "whatever"}) == (None, None, None)


def test_empty_environment_never_matches():
    assert find_matching_model([AI_HUB_ENTRY, EI_ENTRY], {}) == (None, None, None)


def test_malformed_entries_are_skipped():
    models = ["not-a-dict", {"x:y": "not-a-dict"}, {"a:b": {"deployment": {"platforms": ["nope"]}}}, AI_HUB_ENTRY]
    model_id, _data, _platform = find_matching_model(models, AI_HUB_ENV)
    assert model_id == "genie:qwen3_4b_instruct_2507"


@pytest.mark.skipif(not os.path.isfile(REAL_MODELS_LIST), reason="models/models-list.yaml not available")
def test_every_downloadable_entry_round_trips():
    """Every real downloadable entry must be identifiable from its own variables.

    This is what makes the reverse lookup usable in production, and it fails loudly
    the day an entry is added whose variables are ambiguous with another's.
    """
    models = load_models_list(REAL_MODELS_LIST)
    checked = 0
    for entry in models:
        for model_id, model_data in entry.items():
            for platform_entry in (model_data.get("deployment") or {}).get("platforms") or []:
                for platform_name, platform_config in platform_entry.items():
                    variables = platform_config.get("variables") or {}
                    if not variables:
                        continue
                    found, _data, _platform = find_matching_model(models, _env(variables), board=platform_name)
                    assert found == model_id, f"{model_id} ({platform_name}) resolved to {found}"
                    checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# find_model_size_mb
# --------------------------------------------------------------------------- #
def test_find_model_size_mb_match():
    assert find_model_size_mb([AI_HUB_ENTRY], "genie", "qwen3_4b_instruct_2507") == 3039


def test_find_model_size_mb_no_match_returns_minus_one():
    assert find_model_size_mb([AI_HUB_ENTRY], "genie", "absent") == -1


def test_find_model_size_mb_matching_entry_without_metadata():
    entry = {"x:y": {"deployment": {"platforms": [{"ventunoq": {"variables": {"model_type": "genie", "model_name": "n"}}}]}}}
    assert find_model_size_mb([entry], "genie", "n") == -1


# --------------------------------------------------------------------------- #
# load_models_list
# --------------------------------------------------------------------------- #
def test_load_models_list(tmp_path):
    path = tmp_path / "models-list.yaml"
    path.write_text(yaml.safe_dump({"models": [AI_HUB_ENTRY]}))
    assert load_models_list(str(path)) == [AI_HUB_ENTRY]
