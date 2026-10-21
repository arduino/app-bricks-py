# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The VLM brick on the llama.cpp runner: where a llamacpp model is served, and the
thinking block llama.cpp models put before their answer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.vlm import VisionLanguageModel

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
MODEL = "llamacpp:gemma-4-E2B_q4_0-it-multimodal"


class RecordingModel:
    """Stands in for the LangChain client: records the input and answers with a scripted reply."""

    def __init__(self, reply: str = "", chunks: list[str] | None = None):
        self.reply = reply
        self.chunks = chunks or []
        self.inputs = []

    def invoke(self, input, config=None):
        self.inputs.append(input)
        return AIMessage(content=self.reply)

    def stream(self, input, config=None):
        self.inputs.append(input)
        for chunk in self.chunks:
            yield SimpleNamespace(content=chunk, tool_calls=[])


@pytest.fixture
def vlm():
    with patch.object(LargeLanguageModel, "list_models", return_value=[MODEL.split(":", 1)[1]]):
        yield VisionLanguageModel(model=MODEL)


def test_a_llamacpp_model_is_served_by_the_llamacpp_runner(vlm):
    assert vlm._model_name == MODEL
    assert vlm._base_model.openai_api_base == "http://llamacpp-models-runner:9999/v1"


def test_chat_sends_the_image_and_strips_the_thinking_block(vlm):
    vlm._model = RecordingModel(reply="<think>looking</think>A red chair.")

    answer = vlm.chat("What is it?", images=[JPEG])

    assert answer == "A red chair."
    assert [part["type"] for part in vlm._model.inputs[0][-1].content] == ["image_url", "text"]


def test_chat_stream_filters_the_thinking_block(vlm):
    vlm._model = RecordingModel(chunks=["<think>", "looking", "</think>A", " red chair."])

    answer = "".join(vlm.chat_stream("What is it?", images=[JPEG]))

    assert answer == "A red chair."
