# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import inspect
from functools import wraps
from typing import get_type_hints, cast
from uuid import UUID

from conftest import ModelConfig

from deepeval.test_case import ToolCall
from langchain_core.callbacks import BaseCallbackHandler


class ToolCallWithRunId(ToolCall):
    run_id: str | None = None


class ToolTrace(BaseCallbackHandler):
    def __init__(self):
        self.tool_calls: list[ToolCallWithRunId] = []

    def _serialize_run_id(self, run_id: UUID | str | None) -> str | None:
        if run_id is None:
            return None
        if isinstance(run_id, UUID):
            return str(run_id)
        return run_id

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown")
        args = kwargs.get("inputs")
        if args is None:
            try:
                args = json.loads(input_str)
            except Exception:
                args = input_str

        self.tool_calls.append(ToolCallWithRunId(name=name, input_parameters=args, run_id=self._serialize_run_id(run_id)))

    def on_tool_end(self, output, *, run_id, **_):
        run_id = self._serialize_run_id(run_id)
        for call in self.tool_calls:
            if call.run_id == run_id:
                call.output = output
                break


def runner(fn):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) != 3:
        raise TypeError(f"{fn.__name__} must have exactly 3 parameters: (model, prompt, tool_trace)")

    p_model, p_prompt, p_tool_trace = params
    if p_model.name != "model" or p_prompt.name != "prompt" or p_tool_trace.name != "tool_trace":
        raise TypeError(f"{fn.__name__} parameters must be named exactly (model, prompt, tool_trace)")

    hints = get_type_hints(fn)
    if hints.get("model") is not ModelConfig:
        raise TypeError(f"{fn.__name__}: 'model' must be annotated as ModelConfig")
    if hints.get("prompt") is not str:
        raise TypeError(f"{fn.__name__}: 'prompt' must be annotated as str")
    if hints.get("tool_trace") is not ToolTrace:
        raise TypeError(f"{fn.__name__}: 'tool_trace' must be annotated as ToolTrace")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        model = args[0] if args else kwargs.get("model")
        model = cast(ModelConfig, model)

        if model.requires_api_key and not model.api_key:
            raise RuntimeError(f"{fn.__name__}: API key required but not provided for model {model}")

        tool_trace = ToolTrace()
        kwargs["tool_trace"] = tool_trace

        result = fn(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"{fn.__name__} must return a Tuple[CloudLLM, str]")

        return *result, tool_trace

    return wrapper
