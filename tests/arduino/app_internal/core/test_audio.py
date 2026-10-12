# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_internal.core.audio import AudioDetector

RESULT = {"result": {"classification": {"yes": 0.9, "no": 0.3}}}


def test_get_best_match_returns_the_most_confident_class():
    assert AudioDetector.get_best_match(RESULT, 0.5) == ("yes", 90.0)


def test_get_best_match_without_inference_result_finds_nothing():
    assert AudioDetector.get_best_match(None, 0.5) is None


def test_get_best_match_rejects_a_missing_confidence():
    with pytest.raises(ValueError, match="Confidence level must be provided."):
        AudioDetector.get_best_match(RESULT, None)
