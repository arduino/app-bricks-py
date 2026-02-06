# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing import List

from .utils import logger


class WindowedChatMessageHistory:
    """A chat history store that automatically keeps a window of the last k messages."""

    k: int

    def __init__(self, k: int, system_message: str = ""):
        self.k = k
        self._messages: list[BaseMessage] = []
        if system_message != "":
            self._system_message = SystemMessage(content=system_message)
        else:
            self._system_message = None

    def add_messages(self, messages: List[BaseMessage]) -> None:
        if self.k == 0:
            # No memory
            return

        for message in messages:
            self._messages.append(message)

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

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the history, including system message if set."""
        if self.k == 0:
            # No memory
            if self._system_message:
                return [self._system_message]
            else:
                return []

        if self._system_message:
            return [self._system_message] + self._messages
        else:
            return self._messages.copy()

    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []
