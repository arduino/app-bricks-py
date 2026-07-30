# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the `chat_stream` contract on the CloudLLM subclass used for local models.

`LargeLanguageModel.chat_stream` wraps `CloudLLM.chat_stream` and filters `<think>`
tags out of the stream with plain string operations. That only works while the parent
yields flat text, so the same two guarantees are asserted here as for the base brick:
every yielded item is a bare `str`, and the stream never leaves chat completions for
the reasoning client (local runners only expose an OpenAI-compatible
`/v1/chat/completions` endpoint).
"""

import inspect
import threading
from types import SimpleNamespace

import pytest

import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_bricks.llm.local_llm import LargeLanguageModel


class FakeStreamModel:
    """Minimal stand-in for the LangChain model: replays scripted chunks on `stream`."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.stream_calls = 0

    def stream(self, input, config=None):
        self.stream_calls += 1
        yield from self._chunks


class RecordingHistory:
    def __init__(self):
        self.messages = []

    def get_messages(self):
        return list(self.messages)

    def add_messages(self, messages):
        self.messages.extend(messages)


def _chunk(content) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=[])


# The `content` shapes a chunk can carry. Local llama.cpp/genie runners answer with plain
# strings on chat completions, but the brick must stay robust to block lists as well.
CONTENT_SHAPES = [
    pytest.param(lambda text: text, id="string"),
    pytest.param(lambda text: [{"type": "text", "text": text, "index": 0}], id="text-blocks"),
    pytest.param(lambda text: [{"type": "text", "text": text, "annotations": [], "index": 0}], id="openai-responses-blocks"),
    pytest.param(lambda text: [text], id="bare-string-blocks"),
]


def _make_llm(chunks) -> LargeLanguageModel:
    """Builds the brick without touching the network or the local runner discovery."""
    llm = LargeLanguageModel.__new__(LargeLanguageModel)
    llm._model = FakeStreamModel(chunks)
    llm._keep_streaming = threading.Event()
    llm._reasoning_effort_default = None
    llm._callbacks = None
    llm._history = RecordingHistory()
    llm._get_message_with_history = lambda *_args, **_kwargs: []
    return llm


@pytest.mark.parametrize("as_content", CONTENT_SHAPES)
def test_chat_stream_yields_plain_text_for_every_content_shape(as_content):
    llm = _make_llm([_chunk(as_content("Hel")), _chunk(as_content("lo"))])

    out = list(llm.chat_stream("hi"))

    assert out == ["Hel", "lo"]
    assert all(type(c) is str for c in out), f"chat_stream must yield plain strings, got {out!r}"
    assert llm._history.messages[-1].content == "Hello"


@pytest.mark.parametrize("as_content", CONTENT_SHAPES)
def test_chat_stream_strips_think_tags_for_every_content_shape(as_content):
    # The `<think>` filter is string-based: raw block lists would slip through unfiltered.
    llm = _make_llm([
        _chunk(as_content("<think>")),
        _chunk(as_content("reasoning to hide")),
        _chunk(as_content("</think>Hello")),
    ])

    assert list(llm.chat_stream("hi")) == ["Hello"]


def test_chat_stream_drops_thinking_blocks():
    llm = _make_llm([
        _chunk([{"type": "thinking", "thinking": "reasoning to hide", "index": 0}]),
        _chunk("Hello"),
    ])

    assert list(llm.chat_stream("hi")) == ["Hello"]


@pytest.mark.parametrize("effort", ["high", "minimal", 1024, -1, 0])
def test_chat_stream_never_uses_the_reasoning_client(monkeypatch, effort):
    """Local runners only serve chat completions: the stream must never switch client."""
    monkeypatch.setattr(
        CloudLLM,
        "_get_reasoning_model",
        lambda self, reasoning_effort=None: pytest.fail("chat_stream must not route through the reasoning client"),
    )
    llm = _make_llm([_chunk("Hel"), _chunk("lo")])
    llm._reasoning_effort_default = effort

    assert list(llm.chat_stream("hi")) == ["Hel", "lo"]


def test_chat_stream_streams_from_the_configured_model():
    llm = _make_llm([_chunk("Hello")])

    list(llm.chat_stream("hi"))

    assert llm._model.stream_calls == 1


def test_chat_stream_has_no_reasoning_effort_parameter():
    assert "reasoning_effort" not in inspect.signature(LargeLanguageModel.chat_stream).parameters


# --- transport: local runners only serve /v1/chat/completions -----------------


@pytest.mark.parametrize("tools", [None, "with-tools"])
def test_local_model_stays_on_chat_completions(monkeypatch, tools):
    """Regression: binding tools used to switch the client to the Responses API, which the
    genie and llama.cpp runners do not expose, so every `chat()` failed with a 404.
    """

    def get_weather(location: str) -> str:
        """Get the weather."""
        return "sunny"

    monkeypatch.setattr(local_llm_module, "resolve_address", lambda host: host)
    monkeypatch.setattr(LargeLanguageModel, "list_models", lambda self: ["qwen3_4b_instruct_2507"])

    llm = LargeLanguageModel(
        model="genie:qwen3_4b_instruct_2507",
        tools=[get_weather] if tools else None,
    )

    # `chat()` and `chat_stream()` both invoke `self._model`; unwrap the tool binding to
    # reach the client that picks the endpoint.
    inner = getattr(llm._model, "bound", llm._model)
    assert inner._use_responses_api({}) is False, "local runners have no /v1/responses endpoint"
    assert inner.use_responses_api is None
    assert inner.output_version != "responses/v1"
    assert str(inner.openai_api_base) == "http://genie-models-runner:9001/v1"
