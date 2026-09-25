# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``common/model_source.py``."""

import pytest

from common.model_source import hf_publisher, hf_repo_id, model_publisher, model_runtime


@pytest.mark.parametrize(
    "handler, expected",
    [
        ("ai-hub-handler", "qnn"),
        ("ei-handler", "edge-impulse-sdk"),
        ("hf-handler", "llamacpp"),
        ("llamacpp", "llamacpp"),
        ("unknown-handler", None),
    ],
)
def test_model_runtime(handler, expected):
    assert model_runtime(handler) == expected


@pytest.mark.parametrize(
    "model_url, expected",
    [
        ("https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/f0b45be/gemma-3-1b-it-Q4_0.gguf", "unsloth/gemma-3-1b-it-GGUF"),
        ("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/sub/Qwen3-8B-Q8_0.gguf?download=true", "Qwen/Qwen3-8B-GGUF"),
        ("https://huggingface.co/gpt2/resolve/main/gpt2.gguf", "gpt2"),
        ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF", "unsloth/Qwen3-0.6B-GGUF"),
        ("https://example.com/unsloth/repo/blob/main/x.gguf", None),
        ("unsloth/Qwen3-0.6B-GGUF", "unsloth/Qwen3-0.6B-GGUF"),
        ("Qwen/Qwen3-8B-GGUF:Q8_0", "Qwen/Qwen3-8B-GGUF"),
        ("llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0", "Qwen/Qwen3-8B-GGUF"),
        ("llamacpp:Qwen/Qwen3-VL-GGUF:Q4_0:F16", "Qwen/Qwen3-VL-GGUF"),
        ("", None),
    ],
)
def test_hf_repo_id(model_url, expected):
    assert hf_repo_id(model_url) == expected


def test_hf_publisher_keeps_the_owner_case():
    assert hf_publisher("https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/q.gguf") == "Qwen"


def test_hf_publisher_prefers_the_url_over_the_directory():
    assert hf_publisher("unsloth/Qwen3-0.6B-GGUF:Q4_0", "custom/dir") == "unsloth"


def test_hf_publisher_falls_back_to_the_directory():
    assert hf_publisher("", "google/gemma-4-E2B-it-qat-q4_0-gguf") == "google"


def test_hf_publisher_canonical_repo_has_none():
    assert hf_publisher("gpt2:Q4_0") is None
    assert hf_publisher("", "gpt2") is None


@pytest.mark.parametrize(
    "handler, expected",
    [
        ("ai-hub-handler", "qualcomm-ai-hub"),
        ("ei-handler", "edge-impulse"),
        ("unknown-handler", None),
    ],
)
def test_model_publisher_defaults_per_handler(handler, expected):
    assert model_publisher(handler, {}) == expected


def test_model_publisher_hugging_face_is_the_repository_owner():
    url = "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/f0b45be/gemma-3-1b-it-Q4_0.gguf"
    assert model_publisher("hf-handler", {}, url, "unsloth/gemma-3-1b-it-GGUF") == "unsloth"


def test_model_publisher_metadata_overrides_the_default():
    """An Edge Impulse project built on an AI Hub model declares its publisher."""
    assert model_publisher("ei-handler", {"model_publisher": "qualcomm-ai-hub"}) == "qualcomm-ai-hub"
