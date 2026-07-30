# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the CloudLLM chat/stream orchestration.

The underlying LangChain chat model is replaced with a scripted fake, so these
tests exercise the brick's own logic (provider routing, the tool-call loop,
streaming, multimodal input and error handling) without any network access or
provider SDK behavior. The fake is the single mock seam: it implements just the
slice of the BaseChatModel surface the brick relies on (`invoke`, `stream`,
`bind_tools`).
"""

import base64
import inspect
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
from arduino.app_bricks.cloud_llm import CloudLLM, CloudModel, tool
from arduino.app_bricks.cloud_llm.cloud_llm import AlreadyGenerating, model_factory


# --- Fakes & helpers ---------------------------------------------------------


class FakeChatModel:
    """Scriptable stand-in for a LangChain BaseChatModel."""

    def __init__(self):
        self._invoke_queue: list = []
        self._stream_queue: list = []
        self.invoke_inputs: list = []
        self.bound_tools = None

    def queue_invoke(self, *messages):
        """Script the responses returned by successive `invoke` calls."""
        self._invoke_queue.extend(messages)
        return self

    def queue_stream(self, *chunk_batches):
        """Script the chunk batches yielded by successive `stream` calls."""
        self._stream_queue.extend(chunk_batches)
        return self

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, input, config=None):
        self.invoke_inputs.append(input)
        return self._invoke_queue.pop(0)

    def stream(self, input, config=None):
        for chunk in self._stream_queue.pop(0):
            yield chunk


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=[])


def _chunk(content) -> SimpleNamespace:
    """A streamed chunk carrying `content` verbatim (a string or a list of content blocks)."""
    return SimpleNamespace(content=content, tool_calls=[])


# The `content` shapes real providers emit while streaming. Anything but a bare string
# must be flattened to answer text before `chat_stream` yields it.
CONTENT_SHAPES = [
    pytest.param(lambda text: text, id="string"),
    pytest.param(lambda text: [{"type": "text", "text": text, "index": 0}], id="gemini-blocks"),
    pytest.param(lambda text: [{"type": "text", "text": text}], id="anthropic-blocks"),
    pytest.param(lambda text: [{"type": "text", "text": text, "annotations": [], "index": 0}], id="openai-responses-blocks"),
    pytest.param(lambda text: [text], id="bare-string-blocks"),
    pytest.param(lambda text: [{"type": "text", "text": text[:1], "index": 0}, {"type": "text", "text": text[1:], "index": 1}], id="split-blocks"),
]


def _assert_plain_text(chunks: list) -> None:
    """Fails when any yielded chunk is not a bare `str` (e.g. a raw block list/dict)."""
    assert all(type(c) is str for c in chunks), f"chat_stream must yield plain strings, got {chunks!r}"


def _tool_chunk(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _tool_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def fake_model():
    return FakeChatModel()


@pytest.fixture
def make_llm(fake_model, monkeypatch):
    """Build a real CloudLLM whose underlying model is the scripted fake."""

    def _make(**kwargs):
        monkeypatch.setattr(cloud_llm_module, "model_factory", lambda *a, **k: fake_model)
        kwargs.setdefault("api_key", "test-key")
        kwargs.setdefault("model", "openai:gpt-test")
        return CloudLLM(**kwargs)

    return _make


# --- model_factory routing ---------------------------------------------------


@pytest.mark.parametrize(
    "model_name, provider_path, expected_model",
    [
        (CloudModel.ANTHROPIC_CLAUDE, "langchain_anthropic.ChatAnthropic", str(CloudModel.ANTHROPIC_CLAUDE)),
        ("anthropic:claude-x", "langchain_anthropic.ChatAnthropic", "claude-x"),
        (CloudModel.OPENAI_GPT, "arduino.app_bricks.cloud_llm.reasoning.ChatOpenAIReasoning", str(CloudModel.OPENAI_GPT)),
        ("openai:gpt-x", "arduino.app_bricks.cloud_llm.reasoning.ChatOpenAIReasoning", "gpt-x"),
        (CloudModel.GOOGLE_GEMINI, "langchain_google_genai.ChatGoogleGenerativeAI", str(CloudModel.GOOGLE_GEMINI)),
        ("google:gemini-x", "langchain_google_genai.ChatGoogleGenerativeAI", "gemini-x"),
    ],
)
def test_model_factory_routes_to_provider(monkeypatch, model_name, provider_path, expected_model):
    captured = {}

    def fake_ctor(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "CHAT_MODEL"

    monkeypatch.setattr(provider_path, fake_ctor)

    result = model_factory(model_name, api_key="k", temperature=0.1)

    assert result == "CHAT_MODEL"
    assert captured["model"] == expected_model
    assert captured["kwargs"]["api_key"] == "k"


def test_model_factory_defaults_to_openai_when_base_url_and_no_prefix(monkeypatch):
    captured = {}

    def fake_ctor(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "OAI"

    monkeypatch.setattr("arduino.app_bricks.cloud_llm.reasoning.ChatOpenAIReasoning", fake_ctor)

    result = model_factory("local-model", base_url="http://localhost:1234", api_key="k")

    assert result == "OAI"
    assert captured["model"] == "local-model"
    assert captured["kwargs"]["base_url"] == "http://localhost:1234"


def test_model_factory_rejects_unknown_model_without_prefix():
    with pytest.raises(ValueError, match="Model not supported"):
        model_factory("mystery-model", api_key="k")


# --- construction ------------------------------------------------------------


def test_init_omits_temperature_when_none(monkeypatch):
    # The default temperature is None and must NOT be forwarded to the provider, so
    # each SDK uses its own default (and models that deprecated/rejected the field are
    # not sent it, e.g. Anthropic Sonnet 5+ or Gemini which rejects a None temperature).
    captured = {}

    def fake_factory(model, **kwargs):
        captured.update(kwargs)
        return FakeChatModel()

    monkeypatch.setattr(cloud_llm_module, "model_factory", fake_factory)

    CloudLLM(api_key="k", model="openai:gpt-x")

    assert "temperature" not in captured


def test_init_forwards_temperature_when_set(monkeypatch):
    captured = {}

    def fake_factory(model, **kwargs):
        captured.update(kwargs)
        return FakeChatModel()

    monkeypatch.setattr(cloud_llm_module, "model_factory", fake_factory)

    CloudLLM(api_key="k", model="openai:gpt-x", temperature=0.2)

    assert captured["temperature"] == 0.2


def test_init_does_not_forward_reasoning_effort_to_base_model(monkeypatch):
    # reasoning_effort must NEVER be forwarded to the base model: on OpenAI it is sent as a
    # raw chat-completions field and breaks tool calling. It is stored as a default instead.
    captured = {}

    def fake_factory(model, **kwargs):
        captured.update(kwargs)
        return FakeChatModel()

    monkeypatch.setattr(cloud_llm_module, "model_factory", fake_factory)

    llm = CloudLLM(api_key="k", model="openai:gpt-x", reasoning_effort="medium")

    assert "reasoning_effort" not in captured
    assert llm._reasoning_effort_default == "medium"


def test_chat_uses_constructor_reasoning_effort_default(make_llm, monkeypatch):
    # A default effort configured on the brick routes chat() through the reasoning model.
    llm = make_llm(reasoning_effort="low")
    used = {}
    reasoning_fake = FakeChatModel().queue_invoke(AIMessage(content="ok"))

    def fake_get(effort):
        used["effort"] = effort
        return reasoning_fake

    monkeypatch.setattr(llm, "_get_reasoning_model", fake_get)

    assert llm.chat("hi") == "ok"
    assert used["effort"] == "low"


def test_chat_per_call_reasoning_effort_overrides_default(make_llm, monkeypatch):
    llm = make_llm(reasoning_effort="low")
    used = {}
    reasoning_fake = FakeChatModel().queue_invoke(AIMessage(content="ok"))

    def fake_get(effort):
        used["effort"] = effort
        return reasoning_fake

    monkeypatch.setattr(llm, "_get_reasoning_model", fake_get)

    llm.chat("hi", reasoning_effort="high")

    assert used["effort"] == "high"


def test_chat_without_reasoning_effort_default_uses_base_model(make_llm, fake_model):
    # No default and no per-call effort -> base model is used and nothing is added.
    llm = make_llm()
    fake_model.queue_invoke(AIMessage(content="base"))

    assert llm.chat("hi") == "base"


def test_init_openai_with_tools_uses_responses_api():
    # OpenAI reasoning models reject function tools + reasoning on /v1/chat/completions;
    # binding tools must switch the client to the Responses API so tool calling works.
    @tool
    def get_weather(location: str) -> str:
        """Get the weather."""
        return "sunny"

    llm = CloudLLM(model="openai:gpt-5.6-terra", api_key="x", tools=[get_weather])

    inner = getattr(llm._model, "bound", llm._model)
    assert inner._use_responses_api({}) is True


def test_init_openai_without_tools_uses_chat_completions():
    # Without tools the base model is left on the default (chat completions) path.
    llm = CloudLLM(model="openai:gpt-5.6-terra", api_key="x")

    assert llm._model._use_responses_api({}) is False


@pytest.mark.parametrize("model", ["openai:gpt-x", "anthropic:claude-x", "google:gemini-x"])
def test_init_requires_api_key_for_provider_prefixed_models(model):
    with pytest.raises(ValueError, match="API key is required"):
        CloudLLM(api_key="", model=model)


def test_init_allows_empty_key_for_unprefixed_model(make_llm, fake_model):
    # Unprefixed identifiers (e.g. local/ollama-style) do not require a key.
    llm = make_llm(api_key="", model="local-model")
    assert llm.get_client() is fake_model


def test_init_binds_tools_to_model(make_llm, fake_model):
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    make_llm(tools=[ping])
    assert fake_model.bound_tools == [ping]


# --- chat (blocking) ---------------------------------------------------------


def test_chat_returns_text_response(make_llm, fake_model):
    fake_model.queue_invoke(AIMessage(content="Hello there."))
    llm = make_llm()
    assert llm.chat("hi") == "Hello there."


def test_chat_normalizes_block_content_to_text(make_llm, fake_model):
    fake_model.queue_invoke(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
    llm = make_llm()
    assert llm.chat("hi") == "ab"


def test_chat_runs_tool_calls_then_returns_final_answer(make_llm, fake_model):
    seen = []

    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        seen.append(city)
        return f"sunny in {city}"

    fake_model.queue_invoke(
        _tool_message("get_weather", {"city": "Turin"}, "call-1"),
        AIMessage(content="It's sunny in Turin."),
    )
    llm = make_llm(tools=[get_weather])

    assert llm.chat("weather in Turin?") == "It's sunny in Turin."
    assert seen == ["Turin"]


def test_chat_raises_when_tool_loop_limit_exceeded(make_llm, fake_model):
    @tool
    def loop_tool(x: int) -> str:
        """A tool that never lets the model settle."""
        return "again"

    fake_model.queue_invoke(*[_tool_message("loop_tool", {"x": 1}, f"c{i}") for i in range(5)])
    llm = make_llm(tools=[loop_tool], max_tool_loops=2)

    with pytest.raises(RuntimeError, match="Too many consecutive tool-call loops"):
        llm.chat("go")


def test_chat_raises_on_empty_response(make_llm, fake_model):
    fake_model.queue_invoke(None)
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Received empty response"):
        llm.chat("hi")


def test_chat_with_images_sends_multimodal_message(make_llm, fake_model):
    fake_model.queue_invoke(AIMessage(content="ok"))
    llm = make_llm()

    llm.chat("describe", images=[b"\x00\x01"])

    human = fake_model.invoke_inputs[-1][-1]
    assert isinstance(human, HumanMessage)
    assert human.content[0] == {"type": "text", "text": "describe"}
    assert human.content[1]["type"] == "image_url"
    assert human.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# --- chat_stream -------------------------------------------------------------


def test_chat_stream_yields_tokens_and_records_history(make_llm, fake_model):
    fake_model.queue_stream([_text_chunk("Hel"), _text_chunk("lo")])
    llm = make_llm()

    assert list(llm.chat_stream("hi")) == ["Hel", "lo"]

    history = llm._history.get_messages()
    assert isinstance(history[-1], AIMessage)
    assert history[-1].content == "Hello"


def test_chat_stream_stops_when_requested(make_llm, fake_model):
    fake_model.queue_stream([_text_chunk("a"), _text_chunk("b"), _text_chunk("c")])
    llm = make_llm()

    gen = llm.chat_stream("hi")
    assert next(gen) == "a"
    llm.stop_stream()
    assert list(gen) == []


def test_chat_stream_rejects_concurrent_generation(make_llm, fake_model):
    fake_model.queue_stream([_text_chunk("a"), _text_chunk("b")])
    llm = make_llm()

    gen = llm.chat_stream("hi")
    next(gen)  # begin streaming, marking a generation in progress

    with pytest.raises(AlreadyGenerating):
        list(llm.chat_stream("again"))


def test_chat_stream_processes_tool_calls(make_llm, fake_model):
    seen = []

    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        seen.append(city)
        return f"sunny in {city}"

    fake_model.queue_stream(
        [_tool_chunk("get_weather", {"city": "Rome"}, "c1")],
        [_text_chunk("Rome is sunny.")],
    )
    llm = make_llm(tools=[get_weather])

    assert "".join(llm.chat_stream("weather in Rome?")) == "Rome is sunny."
    assert seen == ["Rome"]


# --- chat_stream contract: plain text, chat completions -----------------------
#
# `chat_stream` is documented as an `Iterator[str]` of answer text. Two things must
# never happen again:
#   1. leaking a provider's raw content-block structure (list of dicts) to the caller,
#   2. routing the stream through the reasoning client / Responses API.


@pytest.mark.parametrize("as_content", CONTENT_SHAPES)
def test_chat_stream_yields_plain_text_for_every_provider_content_shape(make_llm, fake_model, as_content):
    fake_model.queue_stream([_chunk(as_content("Hel")), _chunk(as_content("lo"))])
    llm = make_llm()

    out = list(llm.chat_stream("hi"))

    assert out == ["Hel", "lo"]
    _assert_plain_text(out)


@pytest.mark.parametrize("as_content", CONTENT_SHAPES)
def test_chat_stream_records_plain_text_history_for_every_content_shape(make_llm, fake_model, as_content):
    # The `finally` block joins the accumulated chunks: non-flattened content would
    # either raise TypeError here or persist block dicts into the conversation history.
    fake_model.queue_stream([_chunk(as_content("Hel")), _chunk(as_content("lo"))])
    llm = make_llm()

    list(llm.chat_stream("hi"))

    recorded = llm._history.get_messages()[-1]
    assert isinstance(recorded, AIMessage)
    assert recorded.content == "Hello"
    assert type(recorded.content) is str


@pytest.mark.parametrize("as_content", CONTENT_SHAPES)
def test_chat_stream_yields_plain_text_after_tool_calls(make_llm, fake_model, as_content):
    # The second stream (after tool results) is a separate loop and must flatten too.
    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    fake_model.queue_stream(
        [_tool_chunk("get_weather", {"city": "Rome"}, "c1")],
        [_chunk(as_content("Rome is sunny."))],
    )
    llm = make_llm(tools=[get_weather])

    out = list(llm.chat_stream("weather in Rome?"))

    assert out == ["Rome is sunny."]
    _assert_plain_text(out)


@pytest.mark.parametrize(
    "blocks",
    [
        pytest.param([{"type": "thinking", "thinking": "hmm", "index": 0}], id="anthropic-gemini-thinking"),
        pytest.param([{"type": "reasoning", "summary": [{"type": "summary_text", "text": "hmm"}]}], id="openai-reasoning"),
    ],
)
def test_chat_stream_drops_reasoning_blocks(make_llm, fake_model, blocks):
    """Chain-of-thought is exclusive to `chat_stream_reasoning`; `chat_stream` yields answers only."""
    fake_model.queue_stream([_chunk(blocks), _chunk("Hello")])
    llm = make_llm()

    assert list(llm.chat_stream("hi")) == ["Hello"]


def test_chat_stream_keeps_only_text_from_mixed_blocks(make_llm, fake_model):
    # A single chunk can carry reasoning and answer blocks together.
    fake_model.queue_stream([
        _chunk([
            {"type": "thinking", "thinking": "hmm", "index": 0},
            {"type": "text", "text": "Hello", "index": 1},
        ])
    ])
    llm = make_llm()

    assert list(llm.chat_stream("hi")) == ["Hello"]


def test_chat_stream_skips_chunks_without_text(make_llm, fake_model):
    # Empty strings, empty block lists and text-less blocks must not surface as chunks.
    fake_model.queue_stream([
        _chunk(""),
        _chunk([]),
        _chunk([{"type": "text", "text": "", "index": 0}]),
        _chunk("Hello"),
    ])
    llm = make_llm()

    assert list(llm.chat_stream("hi")) == ["Hello"]


@pytest.mark.parametrize("effort", ["high", "minimal", 1024, -1, 0])
def test_chat_stream_never_uses_the_reasoning_client(make_llm, fake_model, monkeypatch, effort):
    """A `reasoning_effort` on the brick must not move `chat_stream` off chat completions."""
    monkeypatch.setattr(
        CloudLLM,
        "_get_reasoning_model",
        lambda self, reasoning_effort=None: pytest.fail("chat_stream must not route through the reasoning client"),
    )
    fake_model.queue_stream([_text_chunk("Hel"), _text_chunk("lo")])
    llm = make_llm(reasoning_effort=effort)

    assert list(llm.chat_stream("hi")) == ["Hel", "lo"]


def test_chat_stream_never_uses_the_reasoning_client_after_tool_calls(make_llm, fake_model, monkeypatch):
    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    monkeypatch.setattr(
        CloudLLM,
        "_get_reasoning_model",
        lambda self, reasoning_effort=None: pytest.fail("chat_stream must not route through the reasoning client"),
    )
    fake_model.queue_stream(
        [_tool_chunk("get_weather", {"city": "Rome"}, "c1")],
        [_text_chunk("Rome is sunny.")],
    )
    llm = make_llm(reasoning_effort="high", tools=[get_weather])

    assert list(llm.chat_stream("weather in Rome?")) == ["Rome is sunny."]


def test_chat_stream_has_no_reasoning_effort_parameter():
    # Guards the public signature: reasoning is opt-in through `chat_stream_reasoning`.
    assert "reasoning_effort" not in inspect.signature(CloudLLM.chat_stream).parameters


# --- image encoding ----------------------------------------------------------


def test_image_to_base64_from_bytes():
    llm = CloudLLM.__new__(CloudLLM)
    assert llm._image_to_base64(b"\x00\x01\x02") == base64.b64encode(b"\x00\x01\x02").decode()


def test_image_to_base64_reads_file(tmp_path):
    path = tmp_path / "img.bin"
    path.write_bytes(b"hello-bytes")
    llm = CloudLLM.__new__(CloudLLM)
    assert llm._image_to_base64(str(path)) == base64.b64encode(b"hello-bytes").decode()


def test_image_to_base64_missing_path_raises():
    llm = CloudLLM.__new__(CloudLLM)
    with pytest.raises(FileNotFoundError):
        llm._image_to_base64("/no/such/file.jpg")


# --- misc surface ------------------------------------------------------------


def test_get_client_returns_underlying_model(make_llm, fake_model):
    assert make_llm().get_client() is fake_model


def test_clear_memory_empties_history(make_llm, fake_model):
    fake_model.queue_invoke(AIMessage(content="hi"))
    llm = make_llm()

    llm.chat("hello")
    assert llm._history.get_messages()

    llm.clear_memory()
    assert llm._history.get_messages() == []
