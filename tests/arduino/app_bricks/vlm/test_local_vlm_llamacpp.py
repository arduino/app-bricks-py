# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The VLM brick on the llama.cpp runner: where a llamacpp model is served."""

from unittest.mock import patch

from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.vlm import VisionLanguageModel

MODEL = "llamacpp:gemma-4-E2B_q4_0-it-multimodal"


def test_a_llamacpp_model_is_served_by_the_llamacpp_runner():
    with patch.object(LargeLanguageModel, "list_models", return_value=[MODEL.split(":", 1)[1]]):
        vlm = VisionLanguageModel(model=MODEL)

    assert vlm._model_name == MODEL
    assert vlm._base_model.openai_api_base == "http://llamacpp-models-runner:9999/v1"
