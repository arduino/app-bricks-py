# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_bricks.vlm import VisionLanguageModel


def test_vlm_chat_applies_reasoning_effort():
    vlm = VisionLanguageModel.__new__(VisionLanguageModel)
    vlm._reasoning_model = None
    vlm._reasoning_effort_default = None

    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        vlm.chat("Describe this image.", reasoning_effort="extreme")
