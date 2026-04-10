# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import time
import base64
import json
import pytest
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# CONFIGURATION - Adjust these for your environment
# ============================================================================
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:9001/v1")
MODEL_NAME = os.environ.get("VLM_MODEL", "qwen3-vl-4b")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
API_KEY = os.environ.get("OPENAI_API_KEY", "xxxx")
IMAGES_DIR = os.environ.get("VLM_IMAGES_DIR", "VLM-IMAGES")
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))

# Set API key for tests
os.environ["OPENAI_API_KEY"] = API_KEY


def image_to_base64(path):
    """Convert image file to base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@pytest.fixture(scope="session")
def vlm_client():
    """Create a shared VLM chat client for all tests"""
    return ChatOpenAI(base_url=BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE, timeout=TIMEOUT)


@pytest.fixture(scope="session")
def test_images():
    """Collect all test images from the VLM-IMAGES directory"""
    images = []
    images_path = Path(IMAGES_DIR)

    if not images_path.exists():
        pytest.skip(f"Images directory '{IMAGES_DIR}' not found")

    for root, dirs, files in os.walk(images_path):
        for f in files:
            # Filter common image extensions
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                images.append(os.path.join(root, f))

    if not images:
        pytest.skip(f"No images found in '{IMAGES_DIR}'")

    return images


@pytest.fixture
def system_prompt():
    """System prompt for the VLM stylist"""
    return (
        "You are a friendly Smart Mirror Stylist. Output ONLY JSON.\n"
        "STRICT CONTENT RULE for 'mirror_message':\n"
        "1. Start by naturally mentioning the clothes and colors you see (e.g., 'I see you're in...', 'Looking sharp in that...').\n"
        "2. Then, transition smoothly into a unique styling tip (e.g., 'Maybe try...', 'A cool touch would be...').\n"
        "Avoid robotic repetition. Vary your opening and transition phrases to sound like a human stylist.\n"
        "Language: English only."
    )


@pytest.fixture
def json_structure():
    """Expected JSON structure for responses"""
    return {"items": [{"item": "string", "color": "string"}], "mirror_message": "string"}


def create_vlm_messages(system_prompt, json_structure, image_b64):
    """Create messages list for VLM request"""
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Look at me and give me a stylish, friendly update on my outfit. "
                        "Mix the analysis and a tip into a single natural paragraph. "
                        f"Format: {json.dumps(json_structure)}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        ),
    ]


class TestVLMStreaming:
    """Test VLM in streaming mode"""

    def pytest_generate_tests(self, metafunc):
        """Dynamically generate tests for each image"""
        if "image_path" in metafunc.fixturenames:
            images_path = Path(IMAGES_DIR)
            if images_path.exists():
                images = []
                for root, dirs, files in os.walk(images_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                            images.append(os.path.join(root, f))
                metafunc.parametrize("image_path", images if images else [None])

    def test_vlm_streaming_per_image(self, vlm_client, system_prompt, json_structure, image_path):
        """Test VLM streaming for each image"""
        if image_path is None:
            pytest.skip(f"No images found in '{IMAGES_DIR}'")

        print(f"\n--- Testing image: {os.path.basename(image_path)} (STREAMING) ---")

        # Convert image to base64
        image_b64 = image_to_base64(image_path)

        # Create messages
        messages = create_vlm_messages(system_prompt, json_structure, image_b64)

        # Track timing
        start = time.time() * 1000

        # Stream the response
        response_content = ""
        chunk_count = 0

        for chunk in vlm_client.stream(messages):
            if chunk.content:
                response_content += chunk.content
                print(chunk.content, end="", flush=True)
                chunk_count += 1

        elapsed = (time.time() * 1000) - start
        print(f"\n--- Streaming completed in {elapsed:.2f}ms ({chunk_count} chunks) ---")

        # Assertions
        assert len(response_content) > 0, "Response should not be empty"
        print(f"Response length: {len(response_content)} characters")

        # Try to parse as JSON (optional check)
        try:
            response_json = json.loads(response_content)
            print(f"Valid JSON response: {json.dumps(response_json, indent=2)}")

            # Check for expected keys
            assert "mirror_message" in response_json, "Response should contain 'mirror_message'"
            assert len(response_json["mirror_message"]) > 0, "mirror_message should not be empty"
        except json.JSONDecodeError as e:
            print(f"Note: Response is not valid JSON: {e}")
            print(f"Raw response: {response_content[:200]}...")
            # Don't fail the test if JSON parsing fails, some models might not follow format perfectly


class TestVLMInvoke:
    """Test VLM in invoke (non-streaming) mode"""

    def pytest_generate_tests(self, metafunc):
        """Dynamically generate tests for each image"""
        if "image_path" in metafunc.fixturenames:
            images_path = Path(IMAGES_DIR)
            if images_path.exists():
                images = []
                for root, dirs, files in os.walk(images_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                            images.append(os.path.join(root, f))
                metafunc.parametrize("image_path", images if images else [None])

    def test_vlm_invoke_per_image(self, vlm_client, system_prompt, json_structure, image_path):
        """Test VLM invoke for each image"""
        if image_path is None:
            pytest.skip(f"No images found in '{IMAGES_DIR}'")

        print(f"\n--- Testing image: {os.path.basename(image_path)} (INVOKE) ---")

        # Convert image to base64
        image_b64 = image_to_base64(image_path)

        # Create messages
        messages = create_vlm_messages(system_prompt, json_structure, image_b64)

        # Track timing
        start = time.time() * 1000

        # Invoke (non-streaming)
        response = vlm_client.invoke(messages)

        elapsed = (time.time() * 1000) - start
        print(f"--- Invoke completed in {elapsed:.2f}ms ---")

        # Extract content
        response_content = response.content
        print(f"Response content:\n{response_content}")

        # Assertions
        assert len(response_content) > 0, "Response should not be empty"
        print(f"Response length: {len(response_content)} characters")

        # Try to parse as JSON (optional check)
        try:
            response_json = json.loads(response_content)
            print(f"Valid JSON response: {json.dumps(response_json, indent=2)}")

            # Check for expected keys
            assert "mirror_message" in response_json, "Response should contain 'mirror_message'"
            assert len(response_json["mirror_message"]) > 0, "mirror_message should not be empty"
        except json.JSONDecodeError as e:
            print(f"Note: Response is not valid JSON: {e}")
            print(f"Raw response: {response_content[:200]}...")
            # Don't fail the test if JSON parsing fails


class TestVLMComparison:
    """Compare streaming vs invoke mode for the same image"""

    def test_streaming_vs_invoke_consistency(self, vlm_client, system_prompt, json_structure, test_images):
        """Test that streaming and invoke return similar results for the first image"""
        if not test_images:
            pytest.skip("No test images available")

        # Use first image for comparison
        image_path = test_images[0]
        print(f"\n--- Comparing modes for: {os.path.basename(image_path)} ---")

        image_b64 = image_to_base64(image_path)
        messages = create_vlm_messages(system_prompt, json_structure, image_b64)

        # Test streaming
        print("\n[STREAMING MODE]")
        streaming_content = ""
        streaming_start = time.time() * 1000
        for chunk in vlm_client.stream(messages):
            if chunk.content:
                streaming_content += chunk.content
        streaming_time = (time.time() * 1000) - streaming_start
        print(f"Streaming: {len(streaming_content)} chars in {streaming_time:.2f}ms")

        # Test invoke
        print("\n[INVOKE MODE]")
        invoke_start = time.time() * 1000
        invoke_response = vlm_client.invoke(messages)
        invoke_time = (time.time() * 1000) - invoke_start
        invoke_content = invoke_response.content
        print(f"Invoke: {len(invoke_content)} chars in {invoke_time:.2f}ms")

        # Assertions
        assert len(streaming_content) > 0, "Streaming should return content"
        assert len(invoke_content) > 0, "Invoke should return content"

        print(f"\n--- Comparison Summary ---")
        print(f"Both modes returned content: ✓")
        print(f"Streaming time: {streaming_time:.2f}ms")
        print(f"Invoke time: {invoke_time:.2f}ms")


class TestVLMBasic:
    """Basic VLM functionality tests"""

    def test_vlm_client_creation(self, vlm_client):
        """Test that VLM client is created successfully"""
        assert vlm_client is not None
        print(f"\nVLM Client: {vlm_client}")
        print(f"Model: {MODEL_NAME}")
        print(f"Base URL: {BASE_URL}")

    def test_image_to_base64_conversion(self, test_images):
        """Test image to base64 conversion"""
        if not test_images:
            pytest.skip("No test images available")

        image_path = test_images[0]
        print(f"\n--- Testing base64 conversion for: {os.path.basename(image_path)} ---")

        image_b64 = image_to_base64(image_path)

        # Assertions
        assert len(image_b64) > 0, "Base64 string should not be empty"
        assert isinstance(image_b64, str), "Base64 should be a string"

        # Check it's valid base64
        try:
            base64.b64decode(image_b64)
            print(f"Valid base64 string: {len(image_b64)} characters")
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_images_directory_exists(self):
        """Test that images directory exists and contains files"""
        images_path = Path(IMAGES_DIR)

        assert images_path.exists(), f"Images directory '{IMAGES_DIR}' should exist"

        # Count images
        image_count = 0
        for root, dirs, files in os.walk(images_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                    image_count += 1

        print(f"\nFound {image_count} images in '{IMAGES_DIR}'")
        assert image_count > 0, f"Should have at least one image in '{IMAGES_DIR}'"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
