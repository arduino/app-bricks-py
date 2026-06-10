# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from typing import List, Optional, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict

from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from .utils import logger


DEFAULT_HISTORY_TABLE = "chat_history"
DEFAULT_DATABASE = "llm.db"
DEFAULT_THREAD_ID = "default"


@runtime_checkable
class MessagePersistence(Protocol):
    """Backend interface for persisting chat messages"""

    def load(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """Return persisted messages, most-recent last.
        `limit` caps how many of the most recent ones are returned (None = all)."""
        ...

    def append(self, messages: List[BaseMessage]) -> None:
        """Persist `messages` in order."""
        ...

    def clear(self) -> None:
        """Remove all persisted messages owned by this store."""
        ...


class SQLMessagePersistence:
    """Chat message persistence store backed by a SQLite table via the SQLStore brick.
    Messages are keyed by `thread_id` to track separate conversations.
    """

    def __init__(
        self,
        sql_store: Optional[SQLStore] = None,
        thread_id: str = DEFAULT_THREAD_ID,
        table_name: str = DEFAULT_HISTORY_TABLE,
    ):
        """Initialize the store backend.

        Args:
            sql_store (SQLStore | None): An already-instantiated SQLStore brick.
                If None, a default `SQLStore("llm.db")` is created.
            thread_id (str): Identifier for the conversation thread. Use distinct values
                to keep separate conversations in the same database.
            table_name (str): Name of the table that stores serialized messages.
        """

        self._sql_store = sql_store if sql_store is not None else SQLStore(DEFAULT_DATABASE)
        self._sql_store.start()
        self._thread_id = thread_id
        self._table_name = table_name
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._sql_store.create_table(
            self._table_name,
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "thread_id": "TEXT NOT NULL",
                "message_data": "TEXT NOT NULL",
            },
        )

    def load(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """Return persisted messages in chronological order, capped to the last `limit`."""
        if limit is not None and limit <= 0:
            return []

        sql = f"SELECT message_data FROM {self._table_name} WHERE thread_id = ? ORDER BY id DESC"
        params: tuple = (self._thread_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (self._thread_id, limit)

        rows = self._sql_store.execute_sql(sql, params)
        if not rows:
            return []

        try:
            dicts = [json.loads(row["message_data"]) for row in reversed(rows)]
            return messages_from_dict(dicts)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to deserialize persisted messages for thread '{self._thread_id}': {e}")
            return []

    def append(self, messages: List[BaseMessage]) -> None:
        if not messages:
            return

        for message, payload in zip(messages, messages_to_dict(messages)):
            try:
                serialized = json.dumps(payload)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize message of type {type(message).__name__}: {e}")
                continue

            self._sql_store.store(
                self._table_name,
                {"thread_id": self._thread_id, "message_data": serialized},
                create_table=False,
            )

    def clear(self) -> None:
        self._sql_store.execute_sql(
            f"DELETE FROM {self._table_name} WHERE thread_id = ?",
            (self._thread_id,),
        )


class WindowedChatMessageHistory:
    """A chat history store that automatically keeps a window of the last k messages."""

    k: int

    def __init__(self, k: int, system_message: str = "", store: Optional[MessagePersistence] = None):
        self.k = k
        self._messages: list[BaseMessage] = []
        self._system_message = SystemMessage(content=system_message) if system_message else None
        self._store = store

        if store is not None and k > 0:
            # Fetch a margin above k so the windowing logic can back up over
            # tool-call boundaries without dropping context.
            self._apply_window(store.load(limit=k * 2))

    def _apply_window(self, messages: List[BaseMessage]) -> None:
        """Append messages to the in-memory cache and enforce the window. No persistence."""
        if self.k == 0:
            return

        self._messages.extend(messages)

        if len(self._messages) > self.k:
            start = len(self._messages) - self.k

            # Ensure we do not start the window with an AIMessage that has tool calls, as that would be not accepted by providers.
            if isinstance(self._messages[start], AIMessage) and len(getattr(self._messages[start], "tool_calls", None) or []) > 0:
                logger.debug("Adjusting memory window to avoid starting with AIMessage(tool_calls).")
                while start >= 0 and not isinstance(self._messages[start], HumanMessage):
                    start -= 1
                if start < 0:
                    raise RuntimeError("Inconsistent state: window starts with AIMessage(tool_calls) but no HumanMessage exists before it.")

            self._messages = self._messages[start:]

    def add_messages(self, messages: List[BaseMessage]) -> None:
        if self.k == 0:
            return

        if self._store is not None:
            self._store.append(messages)

        self._apply_window(messages)

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the history, including system message if set."""
        if self.k == 0:
            return [self._system_message] if self._system_message else []

        if self._system_message:
            return [self._system_message] + self._messages
        return self._messages.copy()

    def clear(self) -> None:
        """Clear the in-memory window and any persisted backend for this history."""
        self._messages = []
        if self._store is not None:
            self._store.clear()
