# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import List


class WindowedChatMessageHistory(InMemoryChatMessageHistory):
    """A chat history store that automatically keeps a window of the last k messages."""

    k: int

    def add_messages(self, messages: List[BaseMessage]) -> None:
        super().add_messages(messages)
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k :]
