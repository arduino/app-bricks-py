# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A request longer than the context window of a local model.

llama-server answers it with HTTP 400 and the error type `exceed_context_size_error`, in
streaming mode as well (the first error of a stream is sent as a plain response), with the
tokens of the request and the size of the context window. The memory then makes room as
`on_context_full` says and the request is sent again: by default the oldest whole turns are
dropped, down to about half of the context window. `ContextLengthExceededError` is raised
when that is not enough or with `on_context_full="raise"`, and a turn that failed is never
added to the memory: before, the user message was recorded ahead of the request, so every
following request resent it and failed again.
"""

from unittest.mock import patch

import httpx2
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import OpenAI

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.llm import ContextLengthExceededError, LargeLanguageModel, tool
from arduino.app_utils import AppError

N_PROMPT_TOKENS = 20000
N_CTX = 16384


def _context_size_error(n_prompt_tokens: int = N_PROMPT_TOKENS, n_ctx: int = N_CTX):
    """The error the OpenAI client raises for llama-server's answer to a request that does not fit."""
    body = {
        "error": {
            "code": 400,
            "message": f"request ({n_prompt_tokens} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": n_prompt_tokens,
            "n_ctx": n_ctx,
        }
    }
    request = httpx2.Request("POST", "http://llamacpp-models-runner:9999/v1/chat/completions")
    with OpenAI(base_url="http://llamacpp-models-runner:9999/v1", api_key="api_key") as client:
        return client._make_status_error_from_response(httpx2.Response(400, json=body, request=request))


class ScriptedModel:
    """Stand-in for the LangChain chat model: each call returns, streams or raises the next scripted item."""

    def __init__(self, *script):
        self._script = list(script)
        self.inputs: list[list[BaseMessage]] = []

    def bind_tools(self, tools):
        return self

    def _next(self, input):
        self.inputs.append(list(input))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def invoke(self, input, config=None, **_kwargs):
        return self._next(input)

    def stream(self, input, config=None):
        yield from self._next(input)


class RecordingStore:
    """A MessagePersistence that records what the brick persists."""

    def __init__(self):
        self.appended: list[BaseMessage] = []

    def load(self, limit=None):
        return []

    def append(self, messages):
        self.appended.extend(messages)

    def clear(self):
        self.appended.clear()


@pytest.fixture
def make_llm(monkeypatch):
    def _make(*script, **kwargs):
        model = ScriptedModel(*script)
        monkeypatch.setattr(cloud_llm_module, "model_factory", lambda *_a, **_k: model)
        monkeypatch.setattr(LargeLanguageModel, "list_models", lambda self: ["Qwen3.5-0.8B-Q4_0"])
        return LargeLanguageModel(model="llamacpp:Qwen3.5-0.8B-Q4_0", **kwargs), model

    return _make


def _fill(llm: LargeLanguageModel, turns: int, size: int = 1000) -> None:
    """Records answered turns of the same size in the memory: question q<i>, answer a<i>."""
    for i in range(turns):
        llm._history.add_messages([HumanMessage(content=f"q{i}".ljust(size)), AIMessage(content=f"a{i}".ljust(size))])


def _questions(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content).strip() for m in messages if isinstance(m, HumanMessage)]


# --- The error -----------------------------------------------------------------


def test_the_error_is_an_app_error_and_a_runtime_error():
    # Apps catching RuntimeError, the documented failure of chat(), keep working
    assert issubclass(ContextLengthExceededError, AppError)
    assert issubclass(ContextLengthExceededError, RuntimeError)


def test_raise_policy_raises_with_the_sizes_and_a_hint(make_llm):
    llm, _ = make_llm(_context_size_error())
    llm.with_memory(on_context_full="raise")

    with pytest.raises(ContextLengthExceededError) as excinfo:
        llm.chat("a very long message")

    assert excinfo.value.prompt_tokens == N_PROMPT_TOKENS
    assert excinfo.value.context_size == N_CTX
    assert f"{N_PROMPT_TOKENS} tokens" in str(excinfo.value)
    assert "clear_memory()" in excinfo.value.hint
    assert "LLAMA_ARG_CTX_SIZE" in excinfo.value.hint


def test_raise_policy_raises_from_chat_stream(make_llm):
    llm, _ = make_llm(_context_size_error())
    llm.with_memory(on_context_full="raise")

    with pytest.raises(ContextLengthExceededError) as excinfo:
        list(llm.chat_stream("a very long message"))

    assert excinfo.value.context_size == N_CTX


# --- A failed turn is never recorded ------------------------------------------


def test_a_failed_chat_turn_is_not_resent(make_llm):
    llm, model = make_llm(_context_size_error(), AIMessage(content="short answer"))
    llm.with_memory(on_context_full="raise")
    _fill(llm, 1)

    with pytest.raises(ContextLengthExceededError):
        llm.chat("a very long message")
    assert llm.chat("short") == "short answer"

    assert _questions(model.inputs[-1]) == ["q0", "short"]
    assert _questions(llm._history.get_messages()) == ["q0", "short"]


def test_a_failed_stream_turn_is_not_resent(make_llm):
    llm, model = make_llm(_context_size_error(), [AIMessage(content="short answer")])
    llm.with_memory(on_context_full="raise")

    with pytest.raises(ContextLengthExceededError):
        list(llm.chat_stream("a very long message"))
    assert list(llm.chat_stream("short")) == ["short answer"]

    assert _questions(model.inputs[-1]) == ["short"]
    assert [m.content for m in llm._history.get_messages()] == ["short", "short answer"]


def test_a_turn_failing_after_a_tool_exchange_leaves_memory_and_store_untouched(make_llm):
    # The tool result is what overflows the context: the second request of the turn fails
    @tool
    def read_log() -> str:
        """Return the device log."""
        return "log line\n" * 10000

    tool_call = AIMessage(content="", tool_calls=[{"name": "read_log", "args": {}, "id": "call-1", "type": "tool_call"}])
    llm, _ = make_llm(tool_call, _context_size_error(), tools=[read_log])
    store = RecordingStore()
    llm.with_memory(persistence=store, on_context_full="raise")

    with pytest.raises(ContextLengthExceededError):
        llm.chat("what does the log say?")

    assert llm._history.get_messages() == []
    assert store.appended == []


def test_an_answered_turn_is_recorded_whole_with_its_tool_exchange(make_llm):
    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    tool_call = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Turin"}, "id": "call-1", "type": "tool_call"}])
    llm, _ = make_llm(tool_call, AIMessage(content="It's sunny in Turin."), tools=[get_weather])
    store = RecordingStore()
    llm.with_memory(persistence=store)

    llm.chat("weather in Turin?")

    expected = ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert [type(m).__name__ for m in llm._history.get_messages()] == expected
    assert [type(m).__name__ for m in store.appended] == expected


def test_a_stream_the_app_stops_reading_keeps_the_partial_answer(make_llm):
    llm, _ = make_llm([AIMessage(content="Hel"), AIMessage(content="lo"), AIMessage(content=" there")])

    stream = llm.chat_stream("hi")
    assert next(stream) == "Hel"
    stream.close()

    assert [m.content for m in llm._history.get_messages()] == ["hi", "Hel"]


# --- The memory makes room: drop_oldest (default) ----------------------------------


def test_drop_oldest_brings_the_request_down_to_half_of_the_context(make_llm):
    # Four turns of 2000 characters and a short question: the request takes 20000 tokens of
    # 16384, so about 59% of its text goes (down to 8192 tokens): the three oldest turns.
    llm, model = make_llm(_context_size_error(), AIMessage(content="answer"))
    _fill(llm, 4)

    with patch.object(local_llm_module.logger, "warning") as log_warning:
        assert llm.chat("new") == "answer"

    assert _questions(model.inputs[0]) == ["q0", "q1", "q2", "q3", "new"]
    assert _questions(model.inputs[1]) == ["q3", "new"]
    assert _questions(llm._history.get_messages()) == ["q3", "new"]
    log_warning.assert_called_once()
    assert "went down by 6 messages" in log_warning.call_args.args[0]


def test_drop_oldest_retries_a_stream(make_llm):
    llm, model = make_llm(_context_size_error(), [AIMessage(content="ans"), AIMessage(content="wer")])
    _fill(llm, 4)

    assert list(llm.chat_stream("new")) == ["ans", "wer"]

    assert _questions(model.inputs[1]) == ["q3", "new"]
    assert [m.content for m in llm._history.get_messages()[-2:]] == ["new", "answer"]


def test_drop_oldest_keeps_the_tool_exchange_of_the_current_turn(make_llm):
    @tool
    def read_log() -> str:
        """Return the device log."""
        return "log line"

    tool_call = AIMessage(content="", tool_calls=[{"name": "read_log", "args": {}, "id": "call-1", "type": "tool_call"}])
    llm, model = make_llm(tool_call, _context_size_error(), AIMessage(content="all good"), tools=[read_log])
    _fill(llm, 2)

    assert llm.chat("what does the log say?") == "all good"

    retried = model.inputs[-1]
    assert _questions(retried) == ["what does the log say?"]
    assert [type(m).__name__ for m in retried] == ["HumanMessage", "AIMessage", "ToolMessage"]


def test_the_store_keeps_the_turns_the_memory_drops(make_llm):
    llm, _ = make_llm(_context_size_error(), AIMessage(content="answer"))
    store = RecordingStore()
    llm.with_memory(persistence=store)
    _fill(llm, 4)

    llm.chat("new")

    assert _questions(store.appended) == ["q0", "q1", "q2", "q3", "new"]


def test_a_message_that_does_not_fit_alone_raises_and_leaves_the_memory_as_it_was(make_llm):
    # The only turn of the memory is dropped, the request is sent again and still does not fit:
    # dropping did not help, so the turn is put back
    llm, model = make_llm(_context_size_error(), _context_size_error(), AIMessage(content="answer"))
    _fill(llm, 1)

    with pytest.raises(ContextLengthExceededError):
        llm.chat("a very long message")

    assert len(model.inputs) == 2
    assert _questions(model.inputs[1]) == ["a very long message"]
    assert _questions(llm._history.get_messages()) == ["q0"]
    assert llm.chat("short") == "answer"
    assert _questions(model.inputs[2]) == ["q0", "short"]


def test_the_retries_are_bounded(make_llm):
    def drop_one(window: list[BaseMessage]) -> list[BaseMessage]:
        return window[1:]

    llm, model = make_llm(*[_context_size_error() for _ in range(10)])
    llm.with_memory(on_context_full=drop_one)
    _fill(llm, 5)

    with pytest.raises(ContextLengthExceededError):
        llm.chat("new")

    assert len(model.inputs) == 1 + cloud_llm_module.CONTEXT_FULL_RETRIES
    assert len(llm._history.get_messages()) == 10


# --- The memory makes room: a callable ---------------------------------------------


def test_a_callable_policy_replaces_the_memory_window(make_llm):
    def summarize(window: list[BaseMessage]) -> list[BaseMessage]:
        return [HumanMessage(content="summary of the oldest turns"), AIMessage(content="ok")] + window[-2:]

    llm, model = make_llm(_context_size_error(), AIMessage(content="answer"))
    llm.with_memory(on_context_full=summarize)
    _fill(llm, 4)

    assert llm.chat("new") == "answer"

    assert _questions(model.inputs[1]) == ["summary of the oldest turns", "q3", "new"]


def test_a_callable_policy_that_keeps_everything_raises(make_llm):
    llm, model = make_llm(_context_size_error())
    llm.with_memory(on_context_full=lambda window: window)
    _fill(llm, 2)

    with pytest.raises(ContextLengthExceededError):
        llm.chat("new")

    assert len(model.inputs) == 1


def test_drop_oldest_empties_the_memory_on_its_last_retry(make_llm):
    # The request keeps being rejected while turns are dropped: the last retry goes out with an empty memory
    retries = cloud_llm_module.CONTEXT_FULL_RETRIES
    llm, model = make_llm(*[_context_size_error() for _ in range(retries)], AIMessage(content="answer"))
    llm.with_memory(max_messages=40)
    _fill(llm, 10)

    assert llm.chat("new") == "answer"

    assert len(model.inputs) == retries + 1
    assert _questions(model.inputs[-1]) == ["new"]


def test_drop_oldest_retries_a_reasoning_stream(make_llm, monkeypatch):
    llm, model = make_llm(_context_size_error(), [AIMessage(content="answer")])
    monkeypatch.setattr(llm, "_get_reasoning_model", lambda _effort=None: model)
    _fill(llm, 4)

    chunks = list(llm.chat_stream_reasoning("new"))

    assert [c.content for c in chunks] == ["answer"]
    assert _questions(model.inputs[1]) == ["q3", "new"]
    assert _questions(llm._history.get_messages()) == ["q3", "new"]


def test_an_unknown_policy_is_rejected(make_llm):
    llm, _ = make_llm()

    with pytest.raises(ValueError, match="on_context_full"):
        llm.with_memory(on_context_full="summarize")
