# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import base64
import os
import pytest
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import APIError

# ============================================================================
# CONFIGURATION - Adjust these for your environment
# ============================================================================
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:9001/v1")
LLM_MODEL_NAME = os.environ.get("LARGE_LLM_MODEL", "qwen2.5-7b")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL", "qwen3-vl-4b")
IMAGES_DIR = os.environ.get("VLM_IMAGES_DIR", "VLM-IMAGES")
API_KEY = os.environ.get("OPENAI_API_KEY", "xxxx")
TIMEOUT = 60

# Set API key for tests
os.environ["OPENAI_API_KEY"] = API_KEY


def image_to_base64(path):
    """Convert image file to base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class TestMemoryExhaustion:
    """Test container exit code in case of memory exhaustion"""

    def test_load_lage_vlm_llm_memory_ex(self):
        """Test streaming with client reinitialization between messages"""

        # Collect streamed content
        print("\n--- Starting first streaming session ---")
        llm = ChatOpenAI(base_url=BASE_URL, model=LLM_MODEL_NAME, timeout=TIMEOUT)
        vlm = ChatOpenAI(base_url=BASE_URL, model=VLM_MODEL_NAME, timeout=TIMEOUT)

        # Try a simple call to VLM. Models should load correctly and respond to a simple text query without OOM
        images_path = Path(IMAGES_DIR)

        if not images_path.exists():
            pytest.skip(f"Images directory '{IMAGES_DIR}' not found")

        for root, dirs, files in os.walk(images_path):
            for f in files:
                image_path = os.path.join(root, f)
                image_b64 = image_to_base64(image_path)

                vlm_query = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": ("Look at me and give me a stylish, friendly update on my outfit. "),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ]
                    ),
                ]

                vlm_response = ""
                for chunk in vlm.stream(vlm_query):
                    if chunk.content and chunk.content != "":
                        vlm_response += chunk.content

                assert len(vlm_response) > 0
                break  # We just want to trigger the memory exhaustion, we don't care about the actual response

        # Try to load a large LLM to trigger memory exhaustion
        llm_response = ""
        try:
            messages = [HumanMessage(content="Hello, how are you?")]
            for chunk in llm.stream(messages):
                if chunk.content and chunk.content != "":
                    llm_response += chunk.content
        except APIError as e:
            print(f"Caught APIError: status_code={e.code}, message={e.message} -> {e}")
            # Check if the error is a 503 Service Unavailable (memory exhaustion)
            if e.code == 503:
                print(f"✓ Test passed - caught expected 503 error due to memory exhaustion")
                return  # Test passed - expected memory exhaustion
            else:
                raise  # Re-raise if it's a different error

        assert len(llm_response) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
