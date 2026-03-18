# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import time
import json
import base64
import mimetypes
import re
from itertools import product
from pathlib import Path
from difflib import SequenceMatcher
from statistics import mean

import pytest
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:9001/v1")
MODEL_NAME = os.environ.get("VLM_MODEL", "qwen3-vl-4b")
IMAGES_DIR = os.environ.get("VLM_IMAGES_DIR", "VLM-IMAGES")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "xxxx")
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))

# Set API key for tests
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PARAM_GRID = {
    "temperature": [0.0, 0.9],
    "top_p": [0.6, 1.0],
    "max_tokens": [64, 512],
}


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_to_data_url(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "image/jpeg"
    image_b64 = image_to_base64(path)
    return f"data:{mime_type};base64,{image_b64}"


def collect_images(images_dir: str) -> list[str]:
    images = []
    images_path = Path(images_dir)

    if not images_path.exists():
        return images

    for root, _, files in os.walk(images_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                images.append(os.path.join(root, f))

    return sorted(images)


def normalize_text(text: str) -> str:
    """
    Apply light normalization to compare near-identical outputs.
    If the response is JSON, prefer comparing the structured payload.
    """
    if not text:
        return ""

    text = text.strip()

    try:
        payload = json.loads(text)

        mirror_message = payload.get("mirror_message", "")
        items = payload.get("items", [])

        normalized_items = json.dumps(items, sort_keys=True, ensure_ascii=False)
        normalized_message = re.sub(r"\s+", " ", str(mirror_message)).strip().lower()

        return f"items={normalized_items}|mirror_message={normalized_message}"
    except Exception:
        pass

    return re.sub(r"\s+", " ", text).strip().lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
    )


@pytest.fixture(scope="session")
def test_images():
    images = collect_images(IMAGES_DIR)
    if not images:
        pytest.skip(f"No images found in '{IMAGES_DIR}'")
    return images


@pytest.fixture(scope="session")
def first_image_path(test_images):
    return test_images[0]


@pytest.fixture(scope="session")
def system_prompt():
    return (
        "You are a friendly Smart Mirror Stylist. Output ONLY JSON.\n"
        "Return exactly this structure:\n"
        "{"
        '"items":[{"item":"string","color":"string"}],'
        '"mirror_message":"string"'
        "}\n"
        "STRICT CONTENT RULE for 'mirror_message':\n"
        "1. Start by naturally mentioning the clothes and colors you see.\n"
        "2. Then transition into a styling tip.\n"
        "3. Keep it natural and varied, not robotic.\n"
        "Language: English only."
    )


@pytest.fixture(scope="session")
def user_prompt():
    return (
        "Look at me and give me a stylish, friendly update on my outfit. "
        "Mix the analysis and a tip into a single natural paragraph. "
        "Return only valid JSON."
    )


class TestVLMBasic:
    def test_client_creation(self, openai_client):
        assert openai_client is not None

    def test_images_directory_exists(self):
        images_path = Path(IMAGES_DIR)
        assert images_path.exists(), f"Images directory '{IMAGES_DIR}' should exist"

        images = collect_images(IMAGES_DIR)
        assert len(images) > 0, f"Should have at least one image in '{IMAGES_DIR}'"

    def test_first_image_base64_conversion(self, first_image_path):
        image_b64 = image_to_base64(first_image_path)
        assert image_b64
        assert isinstance(image_b64, str)

        try:
            base64.b64decode(image_b64)
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")


class TestVLMParameterGrid:
    def test_parameter_combinations_produce_diverse_outputs(
        self,
        openai_client,
        first_image_path,
        system_prompt,
        user_prompt,
        tmp_path,
    ):
        print(f"\n--- Using first image: {os.path.basename(first_image_path)} ---")

        image_data_url = image_to_data_url(first_image_path)
        combinations = list(product(*PARAM_GRID.values()))
        results = []

        for values in combinations:
            params = dict(zip(PARAM_GRID.keys(), values))
            print(f"\nRunning with params: {params}")

            start = time.time()
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_tokens"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    },
                ],
            )
            latency = time.time() - start

            output_text = response.choices[0].message.content or ""
            assert output_text.strip(), f"Empty response for params {params}"

            normalized_output = normalize_text(output_text)

            result = {
                "params": params,
                "latency_sec": latency,
                "output": output_text,
                "normalized_output": normalized_output,
            }
            results.append(result)

            print(f"Latency: {latency:.3f}s")
            print(f"Output length: {len(output_text)}")
            print(f"Output preview: {output_text[:220]!r}")

        # Save results for debugging
        results_file = tmp_path / "grid_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {results_file}")

        raw_outputs = [r["output"] for r in results]
        normalized_outputs = [r["normalized_output"] for r in results]

        unique_raw_count = len(set(raw_outputs))
        unique_normalized_count = len(set(normalized_outputs))
        distinct_rate = unique_normalized_count / len(normalized_outputs)

        print(f"\n--- Diversity summary ---")
        print(f"Total runs: {len(results)}")
        print(f"Unique raw outputs: {unique_raw_count}")
        print(f"Unique normalized outputs: {unique_normalized_count}")
        print(f"Distinct normalized rate: {distinct_rate:.2%}")

        pairwise_similarities = []
        near_duplicate_pairs = []
        clearly_different_pairs = []

        for i in range(len(normalized_outputs)):
            for j in range(i + 1, len(normalized_outputs)):
                sim = similarity(normalized_outputs[i], normalized_outputs[j])
                pairwise_similarities.append(sim)

                pair_info = {
                    "i": i,
                    "j": j,
                    "params_i": results[i]["params"],
                    "params_j": results[j]["params"],
                    "similarity": sim,
                }

                if sim >= 0.97:
                    near_duplicate_pairs.append(pair_info)
                if sim <= 0.90:
                    clearly_different_pairs.append(pair_info)

        avg_similarity = mean(pairwise_similarities) if pairwise_similarities else 1.0
        near_duplicate_ratio = len(near_duplicate_pairs) / len(pairwise_similarities) if pairwise_similarities else 1.0

        print(f"Average pairwise similarity: {avg_similarity:.4f}")
        print(f"Near-duplicate pairs (>= 0.97): {len(near_duplicate_pairs)}")
        print(f"Clearly different pairs (<= 0.90): {len(clearly_different_pairs)}")
        print(f"Near-duplicate ratio: {near_duplicate_ratio:.2%}")

        if near_duplicate_pairs:
            print("\nExample near-duplicate pair:")
            print(json.dumps(near_duplicate_pairs[0], indent=2))

        if clearly_different_pairs:
            print("\nExample clearly-different pair:")
            print(json.dumps(clearly_different_pairs[0], indent=2))

        # Assertions:
        # 1) outputs must not all be identical
        # 2) outputs must not be almost all identical

        assert unique_raw_count > 1, "All raw outputs are identical across all parameter combinations."

        assert unique_normalized_count >= 3, (
            "Too few distinct outputs after normalization. Parameter combinations are not producing enough variation."
        )

        assert distinct_rate >= 0.10, "Distinct output rate is too low. Outputs are too similar across parameter combinations."

        assert clearly_different_pairs, "No clearly different output pair found. All outputs look too similar."

        assert near_duplicate_ratio < 0.90, (
            "Too many outputs are identical or almost identical. The model is not reacting enough to parameter changes."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
