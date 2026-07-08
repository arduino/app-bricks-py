# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for reasoning-mode streaming.

These tests cover two layers without any network access:
- `ChatOpenAIReasoning`: the LangChain subclass that surfaces the standard
  OpenAI Responses API reasoning delta events (which stock langchain-openai
  ignores), fed with scripted fake stream events.
- `CloudLLM.chat_stream_reasoning`: the brick orchestration that separates
  reasoning from answer tokens and persists only the answer to memory, using a
  scripted fake reasoning model.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
from arduino.app_bricks.cloud_llm import CloudLLM, tool
from arduino.app_bricks.cloud_llm.cloud_llm import AlreadyGenerating
from arduino.app_bricks.cloud_llm.reasoning import ChatOpenAIReasoning


# --- Fakes & helpers ---------------------------------------------------------


class _FakeBaseModel:
    """Stand-in for the base chat model that accepts tool binding."""

    def bind_tools(self, tools):
        return self


class _FakeResponsesStream:
    """Context manager yielding scripted OpenAI Responses API stream events."""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *exc):
        return False


class _FakeAsyncResponsesStream:
    """Async context manager yielding scripted Responses API stream events."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        async def _gen():
            for event in self._events:
                yield event

        return _gen()

    async def __aexit__(self, *exc):
        return False


class FakeReasoningModel:
    """Scriptable stand-in for the reasoning-capable chat model.

    Each positional argument is a batch of chunks yielded by a single `stream`
    call, so tool-call round-trips (which re-stream) can be scripted.
    """

    def __init__(self, *batches):
        self._batches = list(batches)
        self.inputs: list = []

    def stream(self, input, config=None):
        self.inputs.append(input)
        batch = self._batches.pop(0)
        yield from batch


def _reasoning_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})


def _content_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=[{"type": "text", "text": text, "index": 0}])


def _tool_call_chunk(name: str, args: str, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(
        content=[{"type": "function_call", "arguments": args, "index": 0}],
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0, "type": "tool_call_chunk"}],
    )


@pytest.fixture
def make_llm(monkeypatch):
    """Build a real CloudLLM without constructing a real provider client."""

    def _make(**kwargs):
        monkeypatch.setattr(cloud_llm_module, "model_factory", lambda *a, **k: _FakeBaseModel())
        kwargs.setdefault("api_key", "test-key")
        kwargs.setdefault("model", "openai:gpt-test")
        return CloudLLM(**kwargs)

    return _make


# --- ChatOpenAIReasoning subclass --------------------------------------------


def test_reasoning_subclass_surfaces_reasoning_and_content_deltas():
    events = [
        SimpleNamespace(type="response.reasoning_text.delta", delta="R1 "),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="R2"),
        SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta="Hi"),
        SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta=" there"),
    ]
    model = ChatOpenAIReasoning(
        model="qwen3",
        api_key="sk-x",
        base_url="http://localhost:9999/v1",
        use_responses_api=True,
        output_version="responses/v1",
    )
    model.root_client.responses.create = lambda **kwargs: _FakeResponsesStream(events)

    results = []
    for chunk in model._stream_responses([HumanMessage("hi")]):
        reasoning = chunk.message.additional_kwargs.get("reasoning_content")
        results.append(("reasoning", reasoning) if reasoning else ("content", chunk.message.text))

    assert results == [
        ("reasoning", "R1 "),
        ("reasoning", "R2"),
        ("content", "Hi"),
        ("content", " there"),
    ]


def test_reasoning_subclass_ignores_empty_reasoning_delta():
    events = [
        SimpleNamespace(type="response.reasoning_text.delta", delta=""),
        SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta="ok"),
    ]
    model = ChatOpenAIReasoning(
        model="qwen3",
        api_key="sk-x",
        base_url="http://localhost:9999/v1",
        use_responses_api=True,
        output_version="responses/v1",
    )
    model.root_client.responses.create = lambda **kwargs: _FakeResponsesStream(events)

    results = [c.message.text for c in model._stream_responses([HumanMessage("hi")])]

    assert results == ["ok"]


# --- CloudLLM.chat_stream_reasoning ------------------------------------------


def test_chat_stream_reasoning_separates_reasoning_and_content(make_llm):
    llm = make_llm()
    llm._reasoning_model = FakeReasoningModel([
        _reasoning_chunk("Think A "),
        _reasoning_chunk("Think B"),
        _content_chunk("Ans"),
        _content_chunk("wer"),
    ])

    out = list(llm.chat_stream_reasoning("hi"))

    assert out == [
        {"type": "reasoning", "content": "Think A "},
        {"type": "reasoning", "content": "Think B"},
        {"type": "content", "content": "Ans"},
        {"type": "content", "content": "wer"},
    ]


def test_chat_stream_reasoning_persists_only_answer_to_memory(make_llm):
    llm = make_llm()
    llm._reasoning_model = FakeReasoningModel([
        _reasoning_chunk("secret thoughts"),
        _content_chunk("Ans"),
        _content_chunk("wer"),
    ])

    list(llm.chat_stream_reasoning("hi"))

    history = [(type(m).__name__, m.content) for m in llm._history.get_messages()]
    assert history == [
        ("HumanMessage", "hi"),
        ("AIMessage", "Answer"),
    ]


def test_chat_stream_reasoning_rejects_non_openai_model(make_llm):
    llm = make_llm()  # base model is a plain object(), not ChatOpenAIReasoning

    with pytest.raises(RuntimeError, match="OpenAI-compatible"):
        list(llm.chat_stream_reasoning("hi"))


def test_chat_stream_reasoning_raises_when_already_streaming(make_llm):
    llm = make_llm()
    llm._reasoning_model = FakeReasoningModel([_content_chunk("x")])
    llm._keep_streaming.set()

    with pytest.raises(AlreadyGenerating):
        list(llm.chat_stream_reasoning("hi"))


def test_chat_stream_reasoning_stop_halts_generation(make_llm):
    llm = make_llm()
    llm._reasoning_model = FakeReasoningModel([
        _content_chunk("first"),
        _content_chunk("second"),
    ])

    collected = []
    for chunk in llm.chat_stream_reasoning("hi"):
        collected.append(chunk)
        llm.stop_stream()

    assert collected == [{"type": "content", "content": "first"}]


def test_chat_stream_reasoning_handles_tool_calls(make_llm):
    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    llm = make_llm(tools=[get_weather])
    # First stream requests a tool call; after the tool runs, the second stream
    # produces the reasoning and the final answer.
    llm._reasoning_model = FakeReasoningModel(
        [_tool_call_chunk("get_weather", '{"city": "Rome"}', "call_1")],
        [_reasoning_chunk("Using the tool result "), _content_chunk("It is sunny in Rome.")],
    )

    out = list(llm.chat_stream_reasoning("weather in Rome?"))

    assert out == [
        {"type": "reasoning", "content": "Using the tool result "},
        {"type": "content", "content": "It is sunny in Rome."},
    ]
    # The tool result must have been fed back into the second stream call.
    second_call_messages = llm._reasoning_model.inputs[1]
    assert any(getattr(m, "content", None) == "sunny in Rome" for m in second_call_messages)


def test_async_reasoning_subclass_surfaces_reasoning_and_content_deltas():
    import asyncio

    events = [
        SimpleNamespace(type="response.reasoning_text.delta", delta="R1"),
        SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta="Hi"),
    ]
    model = ChatOpenAIReasoning(
        model="qwen3",
        api_key="sk-x",
        base_url="http://localhost:9999/v1",
        use_responses_api=True,
        output_version="responses/v1",
    )

    async def _fake_create(**kwargs):
        return _FakeAsyncResponsesStream(events)

    model.root_async_client.responses.create = _fake_create

    async def _collect():
        results = []
        async for chunk in model._astream_responses([HumanMessage("hi")]):
            reasoning = chunk.message.additional_kwargs.get("reasoning_content")
            results.append(("reasoning", reasoning) if reasoning else ("content", chunk.message.text))
        return results

    assert asyncio.run(_collect()) == [("reasoning", "R1"), ("content", "Hi")]
