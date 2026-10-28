# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""What MultimodalLanguageModel sends to the llama.cpp runner.

The brick is built through its real constructor (only the runner's model listing is
stubbed), so the checks cover the actual LangChain client: the default model and runner
address, the temperature in the request body, the media-first ordering of the content
parts and the ``input_audio`` part surviving LangChain's message conversion.
"""

import base64
import gc
import os
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from arduino.app_bricks.cloud_llm.memory import SQLMessagePersistence
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


@pytest.fixture
def brick():
    with patch.object(LargeLanguageModel, "list_models", return_value=["gemma-4-E2B_q4_0-it-multimodal"]):
        yield MultimodalLanguageModel()


def _pcm() -> np.ndarray:
    return (np.sin(np.linspace(0, 100, 1600)) * 10000).astype(np.int16)


def test_defaults_to_the_catalog_model_on_the_llamacpp_runner(brick):
    assert brick._model_name == "llamacpp:gemma-4-E2B_q4_0-it-multimodal"
    assert brick._base_model.openai_api_base == "http://llamacpp-models-runner:9999/v1"


def test_temperature_zero_is_sent_with_every_request(brick):
    payload = brick._base_model._get_request_payload([HumanMessage(content="hi")])

    assert payload["temperature"] == 0.0


def test_text_only_message_stays_a_plain_string(brick):
    messages = brick._get_message_with_history("How many legs does a spider have?")

    assert messages[-1].content == "How many legs does a spider have?"


def test_images_then_audio_then_text(brick):
    messages = brick._get_message_with_history("Describe both.", images=[JPEG], audio=[_pcm()])

    content = messages[-1].content
    assert [part["type"] for part in content] == ["image_url", "input_audio", "text"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[2]["text"] == "Describe both."


def test_single_audio_clip_without_a_list(brick):
    messages = brick._get_message_with_history("Transcribe.", audio=_pcm())

    assert [part["type"] for part in messages[-1].content] == ["input_audio", "text"]


def test_input_audio_part_survives_langchain_conversion(brick):
    """LangChain must forward the ``input_audio`` part to /v1/chat/completions unchanged."""
    messages = brick._get_message_with_history("Transcribe.", images=[JPEG], audio=[_pcm()])

    payload = brick._base_model._get_request_payload(messages)

    sent = payload["messages"][-1]["content"]
    assert sent == messages[-1].content


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


def test_chat_sends_audio_and_strips_thinking(brick):
    brick._model = RecordingModel(reply="<think>listening</think>Water is spilling over the levee.")

    answer = brick.chat("Transcribe.", audio=[_pcm()])

    assert answer == "Water is spilling over the levee."
    assert [part["type"] for part in brick._model.inputs[0][-1].content] == ["input_audio", "text"]


def test_chat_forwards_reasoning_effort_with_audio(brick):
    reasoning_model = RecordingModel(reply="<think>listening</think>Water is spilling over the levee.")

    with patch.object(brick, "_get_reasoning_model", return_value=reasoning_model) as get_reasoning_model:
        answer = brick.chat("Transcribe.", reasoning_effort="low", audio=[_pcm()])

    get_reasoning_model.assert_called_once_with("low")
    assert answer == "Water is spilling over the levee."
    assert [part["type"] for part in reasoning_model.inputs[0][-1].content] == ["input_audio", "text"]


def test_chat_stream_sends_images_and_audio_and_filters_thinking(brick):
    brick._model = RecordingModel(chunks=["<think>", "hmm", "</think>A", " car."])

    answer = "".join(brick.chat_stream("What is it?", images=[JPEG], audio=[_pcm()]))

    assert answer == "A car."
    assert [part["type"] for part in brick._model.inputs[0][-1].content] == ["image_url", "input_audio", "text"]
    assert not brick._keep_streaming.is_set()


def _server_error(message: str) -> openai.InternalServerError:
    response = httpx.Response(500, request=httpx.Request("POST", "http://llamacpp-models-runner:9999/v1/chat/completions"))
    return openai.InternalServerError(message, response=response, body={"code": 500, "message": message})


@pytest.mark.parametrize(
    ("server_message", "expected"),
    [
        pytest.param("failed to process mtmd chunk", "could not encode the image or audio input", id="encoder-failure"),
        pytest.param(
            "audio input is not supported - hint: you may need to provide the mmproj", "does not accept this kind of input", id="no-encoder"
        ),
    ],
)
def test_multimodal_runner_errors_are_explained(brick, server_message, expected):
    failing = MagicMock()
    failing.invoke.side_effect = _server_error(server_message)
    brick._model = failing

    with pytest.raises(RuntimeError, match=expected):
        brick.chat("Transcribe.", audio=[_pcm()])


def test_stream_error_is_explained(brick):
    failing = MagicMock()
    failing.stream.side_effect = _server_error("failed to process mtmd chunk")
    brick._model = failing

    with pytest.raises(RuntimeError, match="could not encode the image or audio input"):
        list(brick.chat_stream("Transcribe.", audio=[_pcm()]))
    assert not brick._keep_streaming.is_set()


def test_memory_is_off_by_default(brick):
    brick._model = RecordingModel(reply="ok")
    brick.chat("first", audio=[_pcm()])
    brick.chat("second")

    assert [m.content for m in brick._model.inputs[1]] == ["second"]


@pytest.fixture
def sql_store():
    with tempfile.TemporaryDirectory() as tmpdir, patch("os.makedirs"):
        db = SQLStore(database_name="multimodal_memory_test")
        db_path = os.path.join(tmpdir, "multimodal_memory_test.db")
        db.database_name = db_path
        db.start()
        yield db
        db.stop()
        gc.collect()
        for _ in range(30):
            try:
                os.remove(db_path)
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(0.1)


def test_sql_store_preserves_audio_message(sql_store, brick):
    content = brick._get_message_with_history("Transcribe.", images=[JPEG], audio=[_pcm()])[-1].content
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="multimodal-audio")

    store.append([HumanMessage(content=content), AIMessage(content="Water is spilling.")])

    loaded = store.load()
    assert loaded[0].content == content
    assert loaded[1].content == "Water is spilling."


def test_stop_stream_interrupts_generation(brick):
    brick._model = RecordingModel(chunks=["a", "b", "c"])
    chunks = []
    for chunk in brick.chat_stream("go"):
        chunks.append(chunk)
        brick.stop_stream()

    assert chunks == ["a"]


def _request_parts(request) -> list[list[str]]:
    """The content part types of each user message of a request."""
    return [[p["type"] for p in m.content] if isinstance(m.content, list) else ["str"] for m in request if isinstance(m, HumanMessage)]


def test_memory_resends_only_the_latest_turn_with_media(brick):
    """Text turns keep the latest image; a new image replaces the earlier one with a placeholder."""
    brick.with_memory(max_messages=10)
    brick._model = RecordingModel(reply="ok")

    brick.chat("What is it?", images=[JPEG])
    brick.chat("What colour is it?")
    brick.chat("And this one?", images=[JPEG], audio=[_pcm()])

    first, second, third = brick._model.inputs
    assert _request_parts(first) == [["image_url", "text"]]
    assert _request_parts(second) == [["image_url", "text"], ["str"]]
    assert _request_parts(third) == [["text", "text"], ["str"], ["image_url", "input_audio", "text"]]
    assert third[0].content[0] == {"type": "text", "text": "[image]"}
