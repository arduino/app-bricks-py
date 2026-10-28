# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""API/signature stability tests for MultimodalLanguageModel.

README snippets and generated user code call these methods by keyword (``images=``,
``audio=``), so renames or reorderings must fail here before they reach users.
"""

import inspect

import pytest

from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel


def _signature_fingerprint(cls, method_name):
    sig = inspect.signature(getattr(cls, method_name))
    return [(name, p.kind, p.default is not inspect.Parameter.empty) for name, p in sig.parameters.items()]


_P = inspect.Parameter
# `audio` comes right after `images`; `chat` adds the inherited `reasoning_effort` after it.
_CHAT = [
    ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ("message", _P.POSITIONAL_OR_KEYWORD, False),
    ("images", _P.POSITIONAL_OR_KEYWORD, True),
    ("audio", _P.POSITIONAL_OR_KEYWORD, True),
    ("reasoning_effort", _P.POSITIONAL_OR_KEYWORD, True),
]
_CHAT_STREAM = [
    ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ("message", _P.POSITIONAL_OR_KEYWORD, False),
    ("images", _P.POSITIONAL_OR_KEYWORD, True),
    ("audio", _P.POSITIONAL_OR_KEYWORD, True),
]
EXPECTED_SIGNATURES = {
    "__init__": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("system_prompt", _P.POSITIONAL_OR_KEYWORD, True),
        ("temperature", _P.POSITIONAL_OR_KEYWORD, True),
        ("max_tokens", _P.POSITIONAL_OR_KEYWORD, True),
        ("timeout", _P.POSITIONAL_OR_KEYWORD, True),
        ("tools", _P.POSITIONAL_OR_KEYWORD, True),
        ("model", _P.POSITIONAL_OR_KEYWORD, True),
        ("kwargs", _P.VAR_KEYWORD, False),
    ],
    "get_client": [("self", _P.POSITIONAL_OR_KEYWORD, False)],
    "chat": _CHAT,
    "chat_stream": _CHAT_STREAM,
    "stop_stream": [("self", _P.POSITIONAL_OR_KEYWORD, False)],
    "clear_memory": [("self", _P.POSITIONAL_OR_KEYWORD, False)],
    "with_memory": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("max_messages", _P.POSITIONAL_OR_KEYWORD, True),
        ("persistence", _P.POSITIONAL_OR_KEYWORD, True),
    ],
}


@pytest.mark.parametrize("method_name", sorted(EXPECTED_SIGNATURES))
def test_public_signatures_are_stable(method_name):
    assert _signature_fingerprint(MultimodalLanguageModel, method_name) == EXPECTED_SIGNATURES[method_name]


def test_default_temperature_is_zero():
    """Deterministic by default: at the llama.cpp default (0.8) the same image question gets different answers."""
    assert inspect.signature(MultimodalLanguageModel.__init__).parameters["temperature"].default == 0.0


def test_inheritance_chain_is_intact():
    assert issubclass(MultimodalLanguageModel, LargeLanguageModel)
    assert issubclass(LargeLanguageModel, CloudLLM)
