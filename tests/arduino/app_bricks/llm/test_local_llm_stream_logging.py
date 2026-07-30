# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
from unittest.mock import patch

import pytest

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_bricks.llm.local_llm import LargeLanguageModel


class FailingStreamModel:
    def stream(self, *_args, **_kwargs):
        raise ValueError("provider exploded")


class EmptyHistory:
    def get_messages(self):
        return []

    def add_messages(self, _messages):
        pass


def test_local_llm_logs_local_model_name_while_using_openai_compatible_adapter():
    captured_model = None

    def fake_cloud_llm_init(self, **kwargs):
        nonlocal captured_model
        captured_model = kwargs["model"]
        self._model_name = kwargs["model"]
        self._model_loaded = False
        self._history = EmptyHistory()

    with (
        patch.object(CloudLLM, "__init__", fake_cloud_llm_init),
        patch.object(LargeLanguageModel, "list_models", return_value=["qwen3_4b_instruct_2507"]),
    ):
        llm = LargeLanguageModel(model="genie:qwen3_4b_instruct_2507")

    assert captured_model == "openai:qwen3_4b_instruct_2507"
    assert llm._model_name == "genie:qwen3_4b_instruct_2507"

    with patch.object(cloud_llm_module.logger, "info") as log_info:
        llm._get_message_with_history("hello")

    log_info.assert_called_once_with("Initializing model genie:qwen3_4b_instruct_2507...")


def test_local_llm_chat_stream_logs_non_api_errors_raised_during_iteration():
    llm = LargeLanguageModel.__new__(LargeLanguageModel)
    llm._model = FailingStreamModel()
    llm._keep_streaming = threading.Event()
    llm._reasoning_effort_default = None
    llm._callbacks = None
    llm._get_message_with_history = lambda *_args, **_kwargs: []

    with patch.object(local_llm_module.logger, "error") as log_error:
        with pytest.raises(RuntimeError, match="Response generation failed: provider exploded"):
            list(llm.chat_stream("hello"))

    log_error.assert_called_once_with("Response generation failed: provider exploded")
