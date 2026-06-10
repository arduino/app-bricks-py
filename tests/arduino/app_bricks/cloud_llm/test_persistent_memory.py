# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import gc
import os
import tempfile
import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from arduino.app_bricks.cloud_llm.memory import (
    DEFAULT_DATABASE,
    DEFAULT_HISTORY_TABLE,
    DEFAULT_THREAD_ID,
    SQLMessagePersistence,
    WindowedChatMessageHistory,
)
from arduino.app_bricks.dbstorage_sqlstore import SQLStore


@pytest.fixture
def sql_store():
    """Provide an open SQLStore backed by a temporary file."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("os.makedirs"):
        db = SQLStore(database_name="chat_memory_test")
        db_path = os.path.join(tmpdir, "chat_memory_test.db")
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


# --- SQLMessagePersistence -----------------------------------------------------


def test_store_append_and_load_round_trip(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="thread-A")
    store.append([HumanMessage(content="hi"), AIMessage(content="hello")])

    loaded = store.load()
    assert [type(m).__name__ for m in loaded] == ["HumanMessage", "AIMessage"]
    assert loaded[0].content == "hi"
    assert loaded[1].content == "hello"


def test_store_default_uses_dedicated_database_and_thread():
    with patch("arduino.app_bricks.dbstorage_sqlstore.SQLStore") as mock_sql_store:
        sql_store = mock_sql_store.return_value

        store = SQLMessagePersistence()

        mock_sql_store.assert_called_once_with(DEFAULT_DATABASE)
        sql_store.start.assert_called_once_with()
        sql_store.create_table.assert_called_once()
        assert store._thread_id == DEFAULT_THREAD_ID


def test_store_thread_id_isolates_conversations(sql_store):
    SQLMessagePersistence(sql_store=sql_store, thread_id="alice").append([HumanMessage(content="alice msg")])
    SQLMessagePersistence(sql_store=sql_store, thread_id="bob").append([HumanMessage(content="bob msg")])

    alice = SQLMessagePersistence(sql_store=sql_store, thread_id="alice").load()
    bob = SQLMessagePersistence(sql_store=sql_store, thread_id="bob").load()

    assert [m.content for m in alice] == ["alice msg"]
    assert [m.content for m in bob] == ["bob msg"]


def test_store_preserves_tool_calls_round_trip(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="tools")
    store.append([
        HumanMessage(content="weather in Turin?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "Turin"}, "id": "call-1", "type": "tool_call"}],
        ),
        ToolMessage(content="sunny, 22C", tool_call_id="call-1"),
        AIMessage(content="It's sunny and 22C in Turin."),
    ])

    loaded = store.load()
    assert len(loaded) == 4

    ai = loaded[1]
    assert isinstance(ai, AIMessage)
    assert ai.tool_calls and ai.tool_calls[0]["name"] == "get_weather"
    assert ai.tool_calls[0]["args"] == {"city": "Turin"}

    tool_msg = loaded[2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call-1"
    assert tool_msg.content == "sunny, 22C"


def test_store_load_with_limit(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="lim")
    for i in range(10):
        store.append([HumanMessage(content=f"q{i}")])

    last3 = store.load(limit=3)
    assert [m.content for m in last3] == ["q7", "q8", "q9"]


def test_store_clear_only_affects_thread(sql_store):
    SQLMessagePersistence(sql_store=sql_store, thread_id="keep").append([HumanMessage(content="keep me")])
    target = SQLMessagePersistence(sql_store=sql_store, thread_id="wipe")
    target.append([HumanMessage(content="to be wiped")])

    target.clear()

    assert target.load() == []
    assert [m.content for m in SQLMessagePersistence(sql_store=sql_store, thread_id="keep").load()] == ["keep me"]


def test_store_custom_table_name(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="custom", table_name="alt_chat")
    store.append([HumanMessage(content="hi")])

    rows = sql_store.execute_sql("SELECT COUNT(*) AS c FROM alt_chat WHERE thread_id = ?", ("custom",))
    assert rows[0]["c"] == 1


# --- WindowedChatMessageHistory with a store --------------------------------


def test_history_persists_via_store(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="hist")
    h1 = WindowedChatMessageHistory(k=10, store=store)
    h1.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])

    h2 = WindowedChatMessageHistory(k=10, store=SQLMessagePersistence(sql_store=sql_store, thread_id="hist"))
    msgs = h2.get_messages()

    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert [m.content for m in msgs] == ["hi", "hello"]


def test_history_window_limits_in_memory_but_store_keeps_all(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="window")
    history = WindowedChatMessageHistory(k=4, store=store)
    for i in range(8):
        history.add_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])

    in_memory = history.get_messages()
    assert len(in_memory) == 4
    assert [m.content for m in in_memory] == ["q6", "a6", "q7", "a7"]

    rows = sql_store.execute_sql(
        f"SELECT COUNT(*) AS c FROM {DEFAULT_HISTORY_TABLE} WHERE thread_id = ?",
        ("window",),
    )
    assert rows[0]["c"] == 16


def test_history_reload_respects_window_size(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="reload")
    seed = WindowedChatMessageHistory(k=100, store=store)
    for i in range(20):
        seed.add_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])

    reloaded = WindowedChatMessageHistory(k=4, store=SQLMessagePersistence(sql_store=sql_store, thread_id="reload"))
    msgs = reloaded.get_messages()
    assert len(msgs) == 4
    assert [m.content for m in msgs] == ["q18", "a18", "q19", "a19"]


def test_history_clear_wipes_store(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="wipeme")
    history = WindowedChatMessageHistory(k=10, store=store)
    history.add_messages([HumanMessage(content="A"), AIMessage(content="B")])

    history.clear()

    assert history.get_messages() == []
    assert store.load() == []


def test_history_system_prompt_is_not_persisted(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="sysprompt")
    h = WindowedChatMessageHistory(k=10, system_message="You are an assistant.", store=store)
    h.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])

    reloaded = WindowedChatMessageHistory(
        k=10,
        system_message="You are an assistant.",
        store=SQLMessagePersistence(sql_store=sql_store, thread_id="sysprompt"),
    )
    msgs = reloaded.get_messages()
    assert isinstance(msgs[0], SystemMessage)
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1
    assert [type(m).__name__ for m in msgs[1:]] == ["HumanMessage", "AIMessage"]

    rows = sql_store.execute_sql(
        f"SELECT message_data FROM {DEFAULT_HISTORY_TABLE} WHERE thread_id = ?",
        ("sysprompt",),
    )
    assert all('"type": "system"' not in row["message_data"] for row in rows)


def test_history_k_zero_disables_memory_and_persistence(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="off")
    history = WindowedChatMessageHistory(k=0, store=store)
    history.add_messages([HumanMessage(content="should not be saved")])

    assert history.get_messages() == []
    rows = sql_store.execute_sql(
        f"SELECT COUNT(*) AS c FROM {DEFAULT_HISTORY_TABLE} WHERE thread_id = ?",
        ("off",),
    )
    assert rows[0]["c"] == 0


def test_history_without_store_is_in_memory_only():
    history = WindowedChatMessageHistory(k=10)
    history.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    assert [m.content for m in history.get_messages()] == ["hi", "hello"]


# --- CloudLLM.with_memory wiring --------------------------------------------


def _bare_llm(system_prompt: str = ""):
    """Build a CloudLLM without going through __init__ (which needs a real API key)."""
    from arduino.app_bricks.cloud_llm import CloudLLM

    llm = CloudLLM.__new__(CloudLLM)
    llm._system_prompt = system_prompt
    llm._max_messages = 0
    return llm


def test_with_memory_default_is_in_memory_only():
    llm = _bare_llm()
    llm.with_memory(max_messages=5)
    assert isinstance(llm._history, WindowedChatMessageHistory)
    assert llm._history._store is None


def test_with_memory_persistence_true_uses_default_sql_backend(sql_store):
    """persistence=True must instantiate SQLMessagePersistence with default args."""
    llm = _bare_llm()
    with patch("arduino.app_bricks.cloud_llm.cloud_llm.SQLMessagePersistence") as mock_store:
        mock_store.return_value = SQLMessagePersistence(sql_store=sql_store, thread_id=DEFAULT_THREAD_ID)
        llm.with_memory(max_messages=5, persistence=True)
        mock_store.assert_called_once_with()
    assert isinstance(llm._history._store, SQLMessagePersistence)


def test_with_memory_accepts_store_instance(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="user-42")
    llm = _bare_llm(system_prompt="You are helpful.")
    llm.with_memory(max_messages=10, persistence=store)

    llm._history.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])

    # New brick session: should reload prior messages from the same thread.
    llm2 = _bare_llm(system_prompt="You are helpful.")
    llm2.with_memory(max_messages=10, persistence=SQLMessagePersistence(sql_store=sql_store, thread_id="user-42"))

    msgs: list[BaseMessage] = llm2._history.get_messages()
    contents = [m.content for m in msgs if not isinstance(m, SystemMessage)]
    assert contents == ["hi", "hello"]
    assert any(isinstance(m, SystemMessage) and m.content == "You are helpful." for m in msgs)


def test_with_memory_persistence_false_is_treated_as_none():
    llm = _bare_llm()
    llm.with_memory(max_messages=5, persistence=False)
    assert llm._history._store is None
