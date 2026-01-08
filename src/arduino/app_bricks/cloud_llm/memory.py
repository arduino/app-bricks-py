# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.messages import BaseMessage, SystemMessage
from typing import List


class WindowedChatMessageHistory:
    """A chat history store that automatically keeps a window of the last k messages."""

    k: int
    messages: list[BaseMessage] = []

    def __init__(self, k: int, system_message: str = ""):
        self.k = k
        if system_message != "":
            self._system_message = SystemMessage(content=system_message)
        else:
            self._system_message = None

    def add_messages(self, messages: List[BaseMessage]) -> None:
        if self.k == 0:
            # No memory
            return

        for message in messages:
            self.messages.append(message)

        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k :]

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the history, including system message if set."""
        if self.k == 0:
            # No memory
            if self._system_message:
                return [self._system_message]
            else:
                return []

        if self._system_message:
            return [self._system_message] + self.messages
        else:
            return self.messages
