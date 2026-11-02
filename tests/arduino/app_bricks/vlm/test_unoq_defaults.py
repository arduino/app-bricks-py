# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the UNO Q defaults of the VLM brick: the image token cap of the llama.cpp runner and the model.

Measured on quattro (UNO Q, 2026-11-02) through VisionLanguageModel.chat() on a small fixed image set:
vision encoding runs on the CPU and dominates the latency. Without a cap an 18 MP photo took 32 min
with SmolVLM2-500M. Qwen3.5-0.8B encodes an image in one pass at ~0.28 s per image token, so the cap
sets its latency: ~25 s per camera frame at 96 tokens against ~73 s at 280, with the same answers.
SmolVLM2-500M never goes below two 512 px encodes (~42 s) whatever the cap.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
UNOQ_COMPOSE = REPO_ROOT / "src" / "arduino" / "app_services" / "llamacpp" / "service_compose.unoq.yaml"
VLM_BRICK_CONFIG = REPO_ROOT / "src" / "arduino" / "app_bricks" / "vlm" / "brick_config.yaml"
MODELS_LIST = REPO_ROOT / "models" / "models-list.yaml"


def _unoq_default_model() -> str:
    config = yaml.safe_load(VLM_BRICK_CONFIG.read_text())
    (model,) = [entry["model"] for entry in config["model_by_boards"] if entry["platform"] == "unoq"]
    return model


def test_unoq_runner_caps_image_tokens_at_96():
    compose = yaml.safe_load(UNOQ_COMPOSE.read_text())
    (service,) = compose["services"].values()
    environment = dict(entry.split("=", 1) for entry in service["environment"])
    assert environment["LLAMA_ARG_IMAGE_MAX_TOKENS"] == "${LLAMA_ARG_IMAGE_MAX_TOKENS:-96}"


def test_unoq_default_vlm_model_is_a_catalog_vision_model_for_unoq():
    """The default is Qwen3.5-0.8B, and the catalog serves it to arduino:vlm on unoq with its mmproj."""
    model = _unoq_default_model()
    assert model == "llamacpp:Qwen3.5-0.8B-Q4_0"

    catalog = {key: value for entry in yaml.safe_load(MODELS_LIST.read_text())["models"] for key, value in entry.items()}
    entry = catalog[model]
    assert "unoq" in entry["supported_boards"]
    assert "arduino:vlm" in [brick["id"] for brick in entry["bricks"]]
    platforms = {name: platform["variables"] for item in entry["deployment"]["platforms"] for name, platform in item.items()}
    assert platforms["unoq"]["model_mmproj_url"], "a VLM default without an mmproj cannot read images"
