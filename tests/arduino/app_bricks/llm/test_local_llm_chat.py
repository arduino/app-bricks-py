# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_bricks.llm.local_llm import LargeLanguageModel


def _make_llm() -> LargeLanguageModel:
    """Builds the brick without touching the network or the local runner discovery."""
    llm = LargeLanguageModel.__new__(LargeLanguageModel)
    llm._reasoning_effort_default = None
    llm._reasoning_model = None
    return llm


def test_chat_forwards_reasoning_effort_like_the_base_brick():
    with pytest.raises(ValueError, match="Unsupported reasoning effort .bogus."):
        _make_llm().chat("hi", reasoning_effort="bogus")
