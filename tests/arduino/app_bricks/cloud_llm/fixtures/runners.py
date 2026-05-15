# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import importlib
import pkgutil
from typing import Callable, Tuple, List, cast

from arduino.app_bricks.cloud_llm import CloudLLM
from runners.__init__ import ToolTrace
from conftest import ModelConfig

from deepeval.test_case import ToolCall


Runner = Callable[[ModelConfig, str], Tuple[CloudLLM, str, ToolTrace]]
ExecuteRunner = Callable[[str, ModelConfig, str], Tuple[str, List[ToolCall], List[ToolCall]]]


@pytest.fixture(scope="session")
def runners_registry():
    registry: dict[str, Runner] = {}
    package = "runners"
    module = importlib.import_module(package)

    for _, name, _ in pkgutil.iter_modules(module.__path__):
        mod = importlib.import_module(f"{package}.{name}")
        found = False
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if not callable(attr):
                continue
            if getattr(attr, "__module__", None) != mod.__name__:
                continue
            found = True
            runner = cast(Runner, attr)
            registry[f"{name}.{attr_name}"] = runner
        if not found:
            raise RuntimeError(f"Module {name} has no run* function")

    return registry


@pytest.fixture(scope="module")
def runners_cache():
    return {}


@pytest.fixture
def execute_runner(runners_registry, runners_cache) -> ExecuteRunner:
    """
    execute_runner(name, model, prompt) -> (response_text, available_tools, tools_called)
    Caches executions to avoid redundant calls.
    """

    def _normalize_llm_response(response) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            if all(isinstance(x, str) for x in response):
                return "\n".join(response)
            return "\n".join(map(str, response))
        return str(response)

    def _execute_runner(name: str, model: ModelConfig, prompt: str) -> Tuple[str, List[ToolCall], List[ToolCall]]:
        key = (name, model.name, model.provider, prompt)

        if key not in runners_cache:
            runner = cast(Runner, runners_registry[name])
            llm, response_raw, tool_trace = runner(model, prompt)
            tools = getattr(llm, "_tools", [])
            available_tools = [ToolCall(name=tool.name, input_parameters={}) for tool in tools]
            called_tools = [ToolCall(name=call.name, input_parameters=call.input_parameters) for call in tool_trace.tool_calls]
            response_text = _normalize_llm_response(response_raw)
            runners_cache[key] = (response_text, available_tools, called_tools)

        return runners_cache[key]

    return _execute_runner
