# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

from arduino.app_bricks.cloud_llm import CloudLLM
from stubs.weather_tool import get_current_weather
from conftest import ModelConfig
from runners import runner, ToolTrace


@runner
def run(model: ModelConfig, prompt: str, tool_trace: ToolTrace) -> Tuple[CloudLLM, str]:
    llm = CloudLLM(
        model=model.name,
        temperature=0,
        api_key=model.api_key,
        tools=[get_current_weather],  # pyright: ignore[reportArgumentType]
        callbacks=[tool_trace],
    )

    llm_response = llm.chat(prompt)

    return llm, llm_response
