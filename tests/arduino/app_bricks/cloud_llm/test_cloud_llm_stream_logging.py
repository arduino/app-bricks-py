# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from unittest.mock import patch

import pytest

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
from arduino.app_bricks.cloud_llm.cloud_llm import CloudLLM


class FailingStreamModel:
    def stream(self, *_args, **_kwargs):
        raise ValueError("provider exploded")


def test_chat_stream_logs_errors_raised_during_iteration():
    llm = CloudLLM.__new__(CloudLLM)
    llm._model = FailingStreamModel()
    llm._keep_streaming = threading.Event()
    llm._get_message_with_history = lambda *_args, **_kwargs: []

    with patch.object(cloud_llm_module.logger, "error") as log_error:
        with pytest.raises(RuntimeError, match="Response generation failed: provider exploded"):
            list(llm.chat_stream("hello"))

    log_error.assert_called_once_with("Response generation failed: provider exploded")
