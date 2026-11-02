# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import yaml

import arduino.app_bricks.vlm as vlm_module

UNOQ_COMPOSE = Path(vlm_module.__file__).parents[2] / "app_services" / "llamacpp" / "service_compose.unoq.yaml"


def _runner_environment() -> dict[str, str]:
    compose = yaml.safe_load(UNOQ_COMPOSE.read_text())
    (service,) = compose["services"].values()
    return dict(entry.split("=", 1) for entry in service["environment"])


def test_unoq_runner_caps_image_tokens_at_280():
    """The UNO Q llama.cpp runner must cap the image tokens, overridable, with a 280 default.

    The VLM vision encoder works on 512 px tiles and each one costs ~20 s on the UNO Q CPU. Without
    a cap llama.cpp tiles an image at its own resolution: on quattro (2026-11-02) an 18 MP photo
    became 70 tiles and took 32 min with SmolVLM2-500M, against 66 s and the same answer with the cap.
    """
    assert _runner_environment()["LLAMA_ARG_IMAGE_MAX_TOKENS"] == "${LLAMA_ARG_IMAGE_MAX_TOKENS:-280}"
