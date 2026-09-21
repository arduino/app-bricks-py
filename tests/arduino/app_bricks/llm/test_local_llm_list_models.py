# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the retry behaviour of `LargeLanguageModel.list_models`.

The brick queries the local models runner at construction time, but the runner lives in a
sibling container that may not be accepting connections yet. A transient connection error
must therefore be retried instead of reported as "no models available", while any other
failure (or an exhausted retry budget) still degrades gracefully to an empty list.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.llm.local_llm import LargeLanguageModel


class FakeOpenAI:
    """Stand-in for the OpenAI client: pops one scripted outcome per `models.list()` call."""

    outcomes: list = []
    constructor_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        FakeOpenAI.constructor_kwargs.append(kwargs)
        self.models = SimpleNamespace(list=self._list)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _list(self):
        outcome = FakeOpenAI.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(data=[SimpleNamespace(id=name) for name in outcome])


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("GET", "http://llamacpp-models-runner:9999/v1/models"))


@pytest.fixture
def llm(monkeypatch):
    FakeOpenAI.outcomes = []
    FakeOpenAI.constructor_kwargs = []
    sleeps: list[float] = []
    monkeypatch.setattr(local_llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(local_llm_module.time, "sleep", lambda s: sleeps.append(s))

    instance = LargeLanguageModel.__new__(LargeLanguageModel)
    instance._model = SimpleNamespace(openai_api_base="http://llamacpp-models-runner:9999/v1", openai_api_key="api_key")
    instance.sleeps = sleeps  # type: ignore[attr-defined]
    return instance


def test_list_models_returns_models_on_first_success(llm):
    FakeOpenAI.outcomes = [["gemma-4-E4B_q4_0-it"]]

    assert llm.list_models() == ["gemma-4-E4B_q4_0-it"]
    assert llm.sleeps == []


def test_list_models_retries_connection_errors_until_runner_is_up(llm):
    FakeOpenAI.outcomes = [_connection_error(), _connection_error(), ["gemma-4-E4B_q4_0-it"]]

    assert llm.list_models() == ["gemma-4-E4B_q4_0-it"]
    assert llm.sleeps == [local_llm_module.LIST_MODELS_RETRY_DELAY_S] * 2


def test_list_models_gives_up_after_max_attempts(llm):
    FakeOpenAI.outcomes = [_connection_error() for _ in range(local_llm_module.LIST_MODELS_MAX_ATTEMPTS)]

    assert llm.list_models() == []
    assert FakeOpenAI.outcomes == [], "every attempt in the budget must be consumed"
    assert len(llm.sleeps) == local_llm_module.LIST_MODELS_MAX_ATTEMPTS - 1


def test_list_models_does_not_retry_other_errors(llm):
    FakeOpenAI.outcomes = [RuntimeError("boom"), ["never-reached"]]

    assert llm.list_models() == []
    assert llm.sleeps == []
    assert FakeOpenAI.outcomes == [["never-reached"]]


def test_list_models_disables_the_client_builtin_retries(llm):
    FakeOpenAI.outcomes = [["gemma-4-E4B_q4_0-it"]]

    llm.list_models()

    assert FakeOpenAI.constructor_kwargs[0]["max_retries"] == 0
