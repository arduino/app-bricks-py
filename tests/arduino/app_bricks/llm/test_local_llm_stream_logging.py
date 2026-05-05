# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from unittest.mock import patch

import pytest

import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.llm.local_llm import LargeLanguageModel


class FailingStreamModel:
    def stream(self, *_args, **_kwargs):
        raise ValueError("provider exploded")


def test_local_llm_chat_stream_logs_non_api_errors_raised_during_iteration():
    llm = LargeLanguageModel.__new__(LargeLanguageModel)
    llm._model = FailingStreamModel()
    llm._keep_streaming = threading.Event()
    llm._get_message_with_history = lambda *_args, **_kwargs: []

    with patch.object(local_llm_module.logger, "error") as log_error:
        with pytest.raises(RuntimeError, match="Response generation failed: provider exploded"):
            list(llm.chat_stream("hello"))

    log_error.assert_called_once_with("Response generation failed: provider exploded")
