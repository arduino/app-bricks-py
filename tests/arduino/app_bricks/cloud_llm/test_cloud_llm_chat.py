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
