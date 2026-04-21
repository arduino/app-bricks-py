# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Chat with an Ollama"

from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App
import time

llm = CloudLLM(
    model="qwen3.5:0.8b",  # Replace with the actual model name you want to use
    base_url="http://localhost:11434/v1",
)


def ask_prompt():
    print(
        llm.chat(message="Who was Giuseppe Verdi?")
    )
    time.sleep(60)


App.run(ask_prompt)
