# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Chat with model with reasoning"

from arduino.app_bricks.cloud_llm import CloudLLM, ContentChunk, ReasoningChunk, ReasoningEffort
from arduino.app_bricks.cloud_llm.models import CloudModel
from arduino.app_utils import App

llm = CloudLLM(
    model=CloudModel.GOOGLE_GEMINI,
    api_key="xyz",
    system_prompt="You are a helpful assistant that provides concise answers to user questions.",
)


def ask_prompt():
    print("\n----- Sending prompt to the model -----")

    print_reasoning_banner = True
    print_final_response_banner = True

    # `reasoning_effort` controls how much the model thinks. Pass a discrete level
    # (ReasoningEffort.MINIMAL/LOW/MEDIUM/HIGH) or an explicit integer token budget
    # (-1 dynamic, 0 off, N>0 token budget).
    for chunk in llm.chat_stream_reasoning(
        message="How many apples do I need to buy to bake an apple pie for 4 people?",
        reasoning_effort=ReasoningEffort.HIGH,
    ):
        if isinstance(chunk, ReasoningChunk):
            if print_reasoning_banner:
                print("\n----- Reasoning -----")
                print_reasoning_banner = False
            print(chunk.content, end="", flush=True)

        elif isinstance(chunk, ContentChunk):
            if print_final_response_banner:
                print("\n----- Final response -----")
                print_final_response_banner = False
            print(chunk.content, end="", flush=True)

    print("\n----- Response complete -----")
    raise StopIteration


App.run(ask_prompt)
