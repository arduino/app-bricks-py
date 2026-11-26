# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards which catalog models App Lab offers to the VLM brick.

App Lab lists a model for a brick only when the brick id appears in the model's `bricks` entry
of models/models-list.yaml. The Gemma 4 "-multimodal" entries are the GGUF loaded with its vision
projector: they read images, so they are offered to arduino:vlm as well as to arduino:multimodal_llm.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_LIST = REPO_ROOT / "models" / "models-list.yaml"
GEMMA4_MULTIMODAL = ["llamacpp:gemma-4-E2B_q4_0-it-multimodal", "llamacpp:gemma-4-E4B_q4_0-it-multimodal"]


def _bricks(model: str) -> dict[str, dict]:
    catalog = {key: value for entry in yaml.safe_load(MODELS_LIST.read_text())["models"] for key, value in entry.items()}
    return {brick["id"]: brick.get("model_configuration", {}) for brick in catalog[model]["bricks"]}


@pytest.mark.parametrize("model", GEMMA4_MULTIMODAL)
def test_gemma4_multimodal_is_offered_to_vlm_with_the_multimodal_llm_configuration(model: str):
    bricks = _bricks(model)
    assert "arduino:vlm" in bricks
    assert bricks["arduino:vlm"] == bricks["arduino:multimodal_llm"]
