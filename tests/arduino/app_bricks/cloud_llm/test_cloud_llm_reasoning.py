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
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
from arduino.app_bricks.cloud_llm import CloudLLM, ContentChunk, ReasoningChunk, tool
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


class FakeInvokeModel:
    """Scriptable stand-in for a chat model supporting non-streaming ``invoke``.

    Each positional argument is the message returned by a single ``invoke`` call.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.inputs: list = []

    def invoke(self, input, config=None):
        self.inputs.append(input)
        return self._responses.pop(0)


def _reasoning_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})


def _content_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=[{"type": "text", "text": text, "index": 0}])


def _gemini_thinking_chunk(text: str) -> AIMessageChunk:
    """Gemini surfaces reasoning as ``thinking`` content blocks."""
    return AIMessageChunk(content=[{"type": "thinking", "thinking": text}])


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
        ReasoningChunk("Think A "),
        ReasoningChunk("Think B"),
        ContentChunk("Ans"),
        ContentChunk("wer"),
    ]


def test_chat_stream_reasoning_separates_gemini_thinking_blocks(make_llm):
    llm = make_llm()
    llm._reasoning_model = FakeReasoningModel([
        _gemini_thinking_chunk("Think A "),
        _gemini_thinking_chunk("Think B"),
        _content_chunk("Ans"),
        _content_chunk("wer"),
    ])

    out = list(llm.chat_stream_reasoning("hi"))

    assert out == [
        ReasoningChunk("Think A "),
        ReasoningChunk("Think B"),
        ContentChunk("Ans"),
        ContentChunk("wer"),
    ]


def test_extract_reasoning_supports_openai_and_gemini_formats(make_llm):
    llm = make_llm()

    openai_token = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "R"})
    gemini_token = AIMessageChunk(content=[{"type": "thinking", "thinking": "T "}, {"type": "text", "text": "ans"}])
    plain_token = AIMessageChunk(content="just an answer")

    assert llm._extract_reasoning(openai_token) == "R"
    assert llm._extract_reasoning(gemini_token) == "T "
    assert llm._extract_reasoning(plain_token) == ""
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


def test_chat_without_reasoning_effort_uses_base_model(make_llm):
    llm = make_llm()
    llm._model = FakeInvokeModel(AIMessage(content="Hello"))

    out = llm.chat("hi")

    assert out == "Hello"
    # The default path must not build the reasoning model at all.
    assert llm._reasoning_model is None
    assert llm._model.inputs, "base model should have been invoked"


def test_chat_with_reasoning_effort_returns_only_answer(make_llm):
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    # Pre-seed the cached reasoning model so the effort cache hit returns our fake
    # (bypasses provider-specific model construction).
    llm._reasoning_effort = ReasoningEffort.HIGH
    llm._reasoning_model = FakeInvokeModel(
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "secret chain-of-thought"},
                {"type": "text", "text": "Final answer"},
            ]
        )
    )

    out = llm.chat("hi", reasoning_effort=ReasoningEffort.HIGH)

    # Only the final answer text is returned; the thinking block is excluded.
    assert out == "Final answer"
    assert llm._reasoning_model.inputs, "reasoning model should have been invoked"


def test_chat_with_reasoning_effort_persists_only_answer_to_history(make_llm):
    llm = make_llm()
    llm._reasoning_effort = 64
    llm._reasoning_model = FakeInvokeModel(
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "secret"},
                {"type": "text", "text": "Answer"},
            ]
        )
    )

    llm.chat("hi", reasoning_effort=64)

    history = [(type(m).__name__, llm._content_to_text(m.content)) for m in llm._history.get_messages()]
    assert history == [
        ("HumanMessage", "hi"),
        ("AIMessage", "Answer"),
    ]


def test_chat_invalid_reasoning_effort_raises_value_error(make_llm):
    llm = make_llm()

    with pytest.raises(ValueError):
        llm.chat("hi", reasoning_effort="nonsense")


def test_get_reasoning_model_enables_gemini_thoughts(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model()

    assert reasoning_model.include_thoughts is True


def test_reasoning_effort_openai_level_and_budget(make_llm):
    from arduino.app_bricks.cloud_llm import ReasoningEffort
    from arduino.app_bricks.cloud_llm.reasoning import ChatOpenAIReasoning

    llm = make_llm()
    llm._base_model = ChatOpenAIReasoning(model="gpt-5", api_key="x")

    llm._reasoning_model = None
    level_model = llm._get_reasoning_model(ReasoningEffort.HIGH)
    assert level_model.reasoning == {"effort": "high", "summary": "auto"}

    # An integer maps to llama.cpp's thinking_budget_tokens via extra_body,
    # with thinking enabled so gated templates honor the budget.
    llm._reasoning_model = None
    budget_model = llm._get_reasoning_model(-1)
    assert budget_model.extra_body == {
        "thinking_budget_tokens": -1,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_reasoning_effort_gemini3_uses_thinking_level(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.MEDIUM)

    assert reasoning_model.thinking_level == "medium"
    assert reasoning_model.include_thoughts is True


def test_reasoning_effort_gemini25_maps_level_to_budget(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from arduino.app_bricks.cloud_llm.models import EFFORT_TO_BUDGET, ReasoningEffort

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model("high")

    assert reasoning_model.thinking_budget == EFFORT_TO_BUDGET[ReasoningEffort.HIGH]
    assert reasoning_model.thinking_level is None


def test_reasoning_effort_gemini_int_budget(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(4096)

    assert reasoning_model.thinking_budget == 4096


def test_reasoning_effort_anthropic_level_maps_to_budget(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort
    from arduino.app_bricks.cloud_llm.models import EFFORT_TO_BUDGET

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.HIGH)

    assert reasoning_model.thinking == {"type": "enabled", "budget_tokens": EFFORT_TO_BUDGET[ReasoningEffort.HIGH]}
    # temperature is left unset (None) when not configured on the brick, so the
    # Anthropic default (1) applies while thinking is active.
    assert reasoning_model.temperature is None


def test_reasoning_effort_anthropic_forwards_configured_temperature(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    # A temperature explicitly configured on the brick is forwarded to the reasoning
    # model (legacy enabled-thinking path).
    llm = make_llm(temperature=0.5)
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.HIGH)

    assert reasoning_model.temperature == 0.5


def test_reasoning_effort_anthropic_adaptive_forwards_configured_temperature(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    # Same, on the adaptive-only path (Sonnet 5+).
    llm = make_llm(temperature=0.3)
    llm._base_model = ChatAnthropic(model="claude-sonnet-5", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.HIGH)

    assert reasoning_model.thinking == {"type": "adaptive", "display": "summarized"}
    assert reasoning_model.temperature == 0.3


def test_reasoning_effort_anthropic_minimal_level_clamped_to_minimum(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    # MINIMAL maps to 512, which is below Anthropic's 1024 minimum and must be clamped.
    reasoning_model = llm._get_reasoning_model(ReasoningEffort.MINIMAL)

    assert reasoning_model.thinking == {"type": "enabled", "budget_tokens": 1024}


def test_reasoning_effort_anthropic_int_budget_clamped(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")

    llm._reasoning_model = None
    big = llm._get_reasoning_model(4096)
    assert big.thinking == {"type": "enabled", "budget_tokens": 4096}
    assert big.temperature is None

    # Below the 1024 minimum is clamped up.
    llm._reasoning_model = None
    small = llm._get_reasoning_model(100)
    assert small.thinking == {"type": "enabled", "budget_tokens": 1024}


def test_reasoning_effort_anthropic_zero_disables_thinking(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(0)

    assert reasoning_model.thinking is None


def test_reasoning_effort_anthropic_negative_uses_adaptive(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    # Legacy model: -1 requests adaptive thinking without the newer display/effort keys.
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(-1)

    assert reasoning_model.thinking == {"type": "adaptive"}
    assert reasoning_model.temperature is None


def test_reasoning_effort_anthropic_none_uses_default_budget(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm.cloud_llm import ANTHROPIC_DEFAULT_THINKING_BUDGET

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model()

    assert reasoning_model.thinking == {"type": "enabled", "budget_tokens": ANTHROPIC_DEFAULT_THINKING_BUDGET}


def test_reasoning_effort_anthropic_raises_max_tokens_above_budget(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    # A small max_tokens must be raised above the thinking budget (budget < max_tokens).
    llm._base_model = ChatAnthropic(model="claude-sonnet-4-6", api_key="x", max_tokens=512)
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(4096)

    assert reasoning_model.thinking == {"type": "enabled", "budget_tokens": 4096}
    assert reasoning_model.max_tokens > 4096


def test_reasoning_effort_anthropic_new_model_uses_adaptive_effort(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    # Sonnet 5+ / Opus 4.7+ dropped budget_tokens and require adaptive thinking + effort.
    llm._base_model = ChatAnthropic(model="claude-sonnet-5", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.HIGH)

    assert reasoning_model.thinking == {"type": "adaptive", "display": "summarized"}
    # HIGH maps up to Anthropic's "xhigh" so adaptive thinking always reasons (its
    # default "high" skips thinking on simple prompts).
    assert reasoning_model.effort == "xhigh"
    assert reasoning_model.temperature is None


def test_reasoning_effort_anthropic_new_model_maps_minimal_to_low(make_llm):
    from langchain_anthropic import ChatAnthropic
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-opus-4-7", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(ReasoningEffort.MINIMAL)

    # Anthropic has no "minimal" effort; it is folded into "low".
    assert reasoning_model.effort == "low"
    assert reasoning_model.thinking == {"type": "adaptive", "display": "summarized"}


def test_reasoning_effort_anthropic_new_model_int_budget_uses_adaptive(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-5", api_key="x")
    llm._reasoning_model = None

    # Adaptive-only models do not accept an explicit budget, so no effort is set.
    reasoning_model = llm._get_reasoning_model(4096)

    assert reasoning_model.thinking == {"type": "adaptive", "display": "summarized"}
    assert reasoning_model.effort is None


def test_reasoning_effort_anthropic_new_model_none_uses_adaptive(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-5-20260101", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model()

    assert reasoning_model.thinking == {"type": "adaptive", "display": "summarized"}
    assert reasoning_model.effort is None


def test_reasoning_effort_anthropic_new_model_zero_disables_thinking(make_llm):
    from langchain_anthropic import ChatAnthropic

    llm = make_llm()
    llm._base_model = ChatAnthropic(model="claude-sonnet-5", api_key="x")
    llm._reasoning_model = None

    reasoning_model = llm._get_reasoning_model(0)

    assert reasoning_model.thinking is None


def test_anthropic_requires_adaptive_version_detection():
    from arduino.app_bricks.cloud_llm.cloud_llm import CloudLLM

    requires = CloudLLM._anthropic_requires_adaptive
    # Legacy (enabled + budget_tokens)
    assert requires("claude-sonnet-4-6") is False
    assert requires("claude-opus-4-5-20251101") is False
    assert requires("claude-opus-4-6") is False
    assert requires("claude-3-7-sonnet-20250219") is False
    # Adaptive-only (adaptive + effort)
    assert requires("claude-opus-4-7") is True
    assert requires("claude-sonnet-5") is True
    assert requires("claude-sonnet-5-20260101") is True


def test_reasoning_effort_invalid_level_raises(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        llm._get_reasoning_model("ultra")


def test_reasoning_effort_numeric_string_rejected(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    with pytest.raises(ValueError, match="numeric string"):
        llm._get_reasoning_model("64")


def test_reasoning_effort_bool_rejected(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    with pytest.raises(ValueError, match="not a bool"):
        llm._get_reasoning_model(True)


def test_reasoning_effort_wrong_type_rejected(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    with pytest.raises(TypeError, match="must be ReasoningEffort"):
        llm._get_reasoning_model(1.5)


def test_reasoning_effort_recomputes_on_change(make_llm):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from arduino.app_bricks.cloud_llm import ReasoningEffort

    llm = make_llm()
    llm._base_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="x")
    llm._reasoning_model = None

    low_model = llm._get_reasoning_model(ReasoningEffort.LOW)
    high_model = llm._get_reasoning_model(ReasoningEffort.HIGH)

    assert low_model.thinking_level == "low"
    assert high_model.thinking_level == "high"
    assert low_model is not high_model


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

    assert collected == [ContentChunk("first")]


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
        ReasoningChunk("Using the tool result "),
        ContentChunk("It is sunny in Rome."),
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
