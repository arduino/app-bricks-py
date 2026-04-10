# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ============================================================================
# CONFIGURATION - Adjust these for your environment
# ============================================================================
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:9001/v1")
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen2.5-3b")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
API_KEY = os.environ.get("OPENAI_API_KEY", "xxxx")
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))

# Set API key for tests
os.environ["OPENAI_API_KEY"] = API_KEY


@pytest.fixture(scope="session")
def chat_client():
    """Create a shared chat client for all tests"""
    return ChatOpenAI(base_url=BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE, timeout=TIMEOUT)


class TestInterruptionsInStreaming:
    """Test suite for simple streaming with client possible reinitialization between messages"""

    def test_client_reinitialization(self, chat_client):
        """Test streaming with client reinitialization between messages"""
        messages = [HumanMessage(content="Write a 1-sentence, encouraging story about a dog that goes to the forest.")]

        # Collect streamed content
        print("\n--- Starting first streaming session ---")
        story_content = ""
        for chunk in chat_client.stream(messages):
            if chunk.content and chunk.content != "":
                story_content += chunk.content
                break  # Interrupt after receiving the first chunk of content

        # Verify we got a response
        print("Received [partial] content:", story_content)
        assert len(story_content) > 0

        # Now simulate client reinitialization by creating a new instance
        # For example, a process that dies and restarts due to an update or crash
        print("\n--- Reinitializing client ----")
        new_story_content = ""
        new_chat_client = ChatOpenAI(
            base_url=BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE, timeout=10
        )  # Use a shorter timeout for the test to fail faster if the client isn't properly reinitialized

        # Collect streamed content
        print("\n--- Starting second streaming session ----")
        for chunk in new_chat_client.stream(messages):
            if chunk.content and chunk.content != "":
                new_story_content += chunk.content

        # Verify we got a response
        print("Received content after reinitialization:", new_story_content)
        assert len(new_story_content) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
