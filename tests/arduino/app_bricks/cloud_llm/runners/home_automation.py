# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Tuple

from conftest import ModelConfig
from arduino.app_bricks.cloud_llm import CloudLLM
from stubs.weather_tool import get_current_weather
from runners import runner, ToolTrace

from langchain_mcp_adapters.client import MultiServerMCPClient


def _run(
    model: ModelConfig,
    prompt: str,
    port: int,
    tool_trace: ToolTrace,
    tools: List[Any] = [],
) -> Tuple[CloudLLM, str]:
    mcp_client = MultiServerMCPClient({
        "home_automation": {
            "transport": "streamable_http",
            "url": f"http://localhost:{port}/mcp",
        }
    })

    mcp_tools = asyncio.run(mcp_client.get_tools())
    tools.extend(mcp_tools)

    llm_kwargs: Dict[str, Any] = {
        "model": model.name,
        "base_url": model.base_url,
        "temperature": 0,
        "api_key": model.api_key,
        "tools": tools,
        "callbacks": [tool_trace],
    }

    llm = CloudLLM(**llm_kwargs)
    llm_response = llm.chat(prompt)

    return llm, llm_response


@runner
def run_granular(model: ModelConfig, prompt: str, tool_trace: ToolTrace) -> Tuple[CloudLLM, str]:
    return _run(model=model, prompt=prompt, port=8000, tool_trace=tool_trace)


@runner
def run_granular_with_weather(model: ModelConfig, prompt: str, tool_trace: ToolTrace) -> Tuple[CloudLLM, str]:
    return _run(model=model, prompt=prompt, port=8000, tools=[get_current_weather], tool_trace=tool_trace)


@runner
def run_object(model: ModelConfig, prompt: str, tool_trace: ToolTrace) -> Tuple[CloudLLM, str]:
    return _run(model=model, prompt=prompt, port=8001, tool_trace=tool_trace)


@runner
def run_object_with_weather(model: ModelConfig, prompt: str, tool_trace: ToolTrace) -> Tuple[CloudLLM, str]:
    return _run(model=model, prompt=prompt, port=8001, tools=[get_current_weather], tool_trace=tool_trace)
