# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import gc
import os
import tempfile
import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from arduino.app_bricks.cloud_llm.memory import DEFAULT_THREAD_ID, SQLMessagePersistence, WindowedChatMessageHistory
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from arduino.app_bricks.vlm import VisionLanguageModel


@pytest.fixture
def sql_store():
    """Provide an open SQLStore backed by a temporary file."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("os.makedirs"):
        db = SQLStore(database_name="vlm_memory_test")
        db_path = os.path.join(tmpdir, "vlm_memory_test.db")
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


def _bare_vlm(system_prompt: str = ""):
    """Build a VisionLanguageModel without going through __init__."""
    vlm = VisionLanguageModel.__new__(VisionLanguageModel)
    vlm._system_prompt = system_prompt
    vlm._max_messages = 0
    return vlm


def test_vlm_with_memory_default_is_in_memory_only():
    vlm = _bare_vlm()
    vlm.with_memory(max_messages=5)

    assert isinstance(vlm._history, WindowedChatMessageHistory)
    assert vlm._history._store is None


def test_vlm_with_memory_persistence_true_uses_default_sql_backend(sql_store):
    vlm = _bare_vlm()
    with patch("arduino.app_bricks.cloud_llm.cloud_llm.SQLMessagePersistence") as mock_store:
        mock_store.return_value = SQLMessagePersistence(sql_store=sql_store, thread_id=DEFAULT_THREAD_ID)
        vlm.with_memory(max_messages=5, persistence=True)
        mock_store.assert_called_once_with()

    assert isinstance(vlm._history._store, SQLMessagePersistence)


def test_vlm_with_memory_accepts_store_instance(sql_store):
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="vision-session")
    vlm = _bare_vlm(system_prompt="You are a visual assistant.")
    vlm.with_memory(max_messages=10, persistence=store)

    vlm._history.add_messages([HumanMessage(content="Remember this image."), AIMessage(content="I will remember it.")])

    reloaded = _bare_vlm(system_prompt="You are a visual assistant.")
    reloaded.with_memory(max_messages=10, persistence=SQLMessagePersistence(sql_store=sql_store, thread_id="vision-session"))

    contents = [m.content for m in reloaded._history.get_messages()]
    assert "Remember this image." in contents
    assert "I will remember it." in contents


def test_sql_store_preserves_multimodal_image_url_message(sql_store):
    content = [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
    ]
    store = SQLMessagePersistence(sql_store=sql_store, thread_id="vision-image")

    store.append([HumanMessage(content=content), AIMessage(content="It looks like a chair.")])

    loaded = store.load()
    assert loaded[0].content == content
    assert loaded[1].content == "It looks like a chair."
