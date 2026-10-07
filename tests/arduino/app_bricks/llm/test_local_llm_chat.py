# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import httpx
import pytest
from openai import APIError

from arduino.app_bricks.llm.local_llm import LargeLanguageModel


def _make_llm() -> LargeLanguageModel:
    """Builds the brick without touching the network or the local runner discovery."""
    llm = LargeLanguageModel.__new__(LargeLanguageModel)
    llm._reasoning_effort_default = None
    llm._reasoning_model = None
    return llm


def test_chat_forwards_reasoning_effort_like_the_base_brick():
    with pytest.raises(ValueError, match="Unsupported reasoning effort .bogus."):
        _make_llm().chat("hi", reasoning_effort="bogus")


class FailingModel:
    def __init__(self, error: Exception):
        self._error = error

    def invoke(self, *_args, **_kwargs):
        raise self._error


@pytest.mark.parametrize("code", [503, "503"], ids=["number", "string"])
def test_init_reports_npu_memory_exhaustion_on_runner_code_503(code):
    llm = _make_llm()
    llm._base_model = FailingModel(APIError("Loading model", httpx.Request("POST", "http://genie-models-runner:9001/v1"), body={"code": code}))

    with pytest.raises(RuntimeError, match="Cannot load model due to a potential memory exhaustion on NPU sessions. message=Loading model"):
        llm.init()


def test_init_reports_other_runner_errors_with_their_code():
    llm = _make_llm()
    llm._base_model = FailingModel(APIError("boom", httpx.Request("POST", "http://genie-models-runner:9001/v1"), body={"code": 500}))

    with pytest.raises(RuntimeError, match="Error: status_code=500, message=boom"):
        llm.init()
