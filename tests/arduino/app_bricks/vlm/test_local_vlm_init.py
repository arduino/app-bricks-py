# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the `init()` warm-up contract of the VisionLanguageModel brick.

A VLM asked a text-only question can misbehave or fail, so `init()` must always send a
real image. The image is synthesized in memory (never read from disk), placed before the
text as in `chat()`, and the call goes to the base model with `max_tokens=1` without
touching the conversation history.
"""

import base64
import io
import threading

import openai
import pytest
from PIL import Image
from langchain_core.messages import HumanMessage

import arduino.app_bricks.vlm.local_vlm as local_vlm_module
from arduino.app_bricks.vlm.local_vlm import VisionLanguageModel


class RecordingModel:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self._error = error

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self._error is not None:
            raise self._error
        return "YES"


class RecordingHistory:
    def __init__(self):
        self.messages = []

    def get_messages(self):
        return list(self.messages)

    def add_messages(self, messages):
        self.messages.extend(messages)


def _make_vlm(base_model) -> VisionLanguageModel:
    """Builds the brick without touching the network or the local runner discovery."""
    vlm = VisionLanguageModel.__new__(VisionLanguageModel)
    vlm._base_model = base_model
    vlm._model = RecordingModel()  # must stay untouched: init() warms up the base model only
    vlm._keep_streaming = threading.Event()
    vlm._history = RecordingHistory()
    return vlm


def _content_blocks(model: RecordingModel):
    assert len(model.calls) == 1
    messages, kwargs = model.calls[0]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    return messages[0].content, kwargs


def test_init_sends_black_image_before_yes_no_prompt():
    base_model = RecordingModel()
    vlm = _make_vlm(base_model)

    vlm.init()

    content, kwargs = _content_blocks(base_model)
    assert kwargs == {"max_tokens": 1}
    assert [block["type"] for block in content] == ["image_url", "text"], "image must precede the text, as in chat()"

    url = content[0]["image_url"]["url"]
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    image = Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :])))
    assert image.format == "JPEG"
    assert image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0)), "canary image must be solid black"

    text = content[1]["text"]
    assert "YES" in text and "NO" in text
    assert "black" in text.lower()


def test_init_does_not_touch_history_or_wrapped_model():
    base_model = RecordingModel()
    vlm = _make_vlm(base_model)

    vlm.init()

    assert vlm._history.messages == []
    assert vlm._model.calls == []


def test_init_builds_image_in_memory_without_filesystem(monkeypatch):
    local_vlm_module._canary_image_jpeg.cache_clear()
    monkeypatch.setattr(local_vlm_module.Image, "open", lambda *a, **k: pytest.fail("init() must not read images from disk"))
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("init() must not open files"))

    base_model = RecordingModel()
    _make_vlm(base_model).init()

    content, _ = _content_blocks(base_model)
    assert content[0]["type"] == "image_url"


def test_init_wraps_api_errors_into_runtime_error():
    error = openai.APIError("runner down", request=None, body=None)
    vlm = _make_vlm(RecordingModel(error=error))

    with pytest.raises(RuntimeError):
        vlm.init()
