# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import pytest

from deepeval.models import GeminiModel


@pytest.fixture(scope="session")
def judge_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set for judge model")

    return GeminiModel(
        model=os.getenv("DEEPEVAL_JUDGE_MODEL", "gemini-3-pro-preview"),
        api_key=api_key,
    )
