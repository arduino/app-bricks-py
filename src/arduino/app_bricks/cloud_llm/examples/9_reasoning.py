# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Chat with an Ollama model"

from arduino.app_bricks.cloud_llm import CloudLLM, ContentChunk, ReasoningChunk
from arduino.app_utils import App
import time

llm = CloudLLM(
    model="/var/lib/arduino-app-cli/models/llamacpp/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf",
    base_url="http://192.168.1.212:9998/v1",
    system_prompt="You are a helpful assistant that provides concise answers to questions about historical figures.",
)


def ask_prompt():
    print("\n----- Sending prompt to the model -----")
    final_response = False
    for chunk in llm.chat_stream_reasoning(message="Who was Giuseppe Verdi?"):
        if isinstance(chunk, ReasoningChunk):
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, ContentChunk):
            if not final_response:
                print("\n----- Final response -----")
                final_response = True
            print(chunk.content, end="", flush=True)
    print("\n----- Response complete -----")
    time.sleep(60)


App.run(ask_prompt)
