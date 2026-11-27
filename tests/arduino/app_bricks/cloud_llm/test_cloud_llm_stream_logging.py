# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
from arduino.app_bricks.cloud_llm.cloud_llm import CloudLLM
from arduino.app_bricks.cloud_llm.memory import WindowedChatMessageHistory


class FailingStreamModel:
    def stream(self, *_args, **_kwargs):
        raise ValueError("provider exploded")


def test_chat_stream_logs_errors_raised_during_iteration():
    llm = CloudLLM.__new__(CloudLLM)
    llm._model = FailingStreamModel()
    llm._keep_streaming = threading.Event()
    llm._reasoning_effort_default = None
    llm._callbacks = None
    llm._history = WindowedChatMessageHistory(k=10)
    llm._user_message = lambda *_args, **_kwargs: HumanMessage(content="hello")

    with patch.object(cloud_llm_module.logger, "error") as log_error:
        with pytest.raises(RuntimeError, match="Response generation failed: provider exploded"):
            list(llm.chat_stream("hello"))

    log_error.assert_called_once_with("Response generation failed: provider exploded")
