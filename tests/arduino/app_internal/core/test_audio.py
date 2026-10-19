# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading

import pytest

from arduino.app_internal.core.audio import AudioDetector
from arduino.app_utils.slidingwindowbuffer import SlidingWindowBuffer

RESULT = {"result": {"classification": {"yes": 0.9, "no": 0.3}}}


def test_get_best_match_returns_the_most_confident_class():
    assert AudioDetector.get_best_match(RESULT, 0.5) == ("yes", 90.0)


def test_get_best_match_without_inference_result_finds_nothing():
    assert AudioDetector.get_best_match(None, 0.5) is None


def test_get_best_match_rejects_a_missing_confidence():
    with pytest.raises(ValueError, match="Confidence level must be provided."):
        AudioDetector.get_best_match(RESULT, None)


def test_inference_loop_returns_while_no_window_is_available():
    """One loop iteration must not wait forever on an empty buffer, or the worker could never notice a stop."""
    # Bypass __init__: no microphone and no model runner are needed to pull from the window buffer
    detector = AudioDetector.__new__(AudioDetector)
    detector._buffer = SlidingWindowBuffer(window_size=16000, slide_amount=16000)

    iteration = threading.Thread(target=detector._inference_loop, daemon=True)
    iteration.start()
    iteration.join(timeout=2)

    assert not iteration.is_alive()
