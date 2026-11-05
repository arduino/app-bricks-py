# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
from types import SimpleNamespace

import numpy as np
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


def test_mic_loop_stops_iterating_once_the_microphone_is_stopped():
    """The App stops a brick before its loops: the read loop then ends quietly, without logging
    an error and without being called again on a stopped microphone."""
    detector = AudioDetector.__new__(AudioDetector)
    detector._buffer = SlidingWindowBuffer(window_size=16000, slide_amount=16000)
    detector._mic = SimpleNamespace(stream=lambda: iter([]))  # the stream of a stopped microphone

    with pytest.raises(StopIteration):
        detector._read_mic_loop()


def test_audio_heard_while_a_callback_runs_is_not_classified():
    """A callback that takes long, like a voice assistant answering the request, pauses the
    detection: what the microphone heard meanwhile, including the tail of the keyword still in
    the overlapping windows, must not trigger the callback again once it returns."""
    detector = AudioDetector.__new__(AudioDetector)
    detector._buffer = SlidingWindowBuffer(window_size=4, slide_amount=2)
    detector._buffer.push(np.ones(4, dtype=np.int16))
    detector.confidence = 0.5
    detector._debounce_sec = 2.0
    detector._last_detected = {}
    detector.handlers_lock = threading.Lock()
    detector.infer_from_features = lambda features: RESULT
    calls = []

    def answer_request():
        calls.append("yes")
        detector._buffer.push(np.ones(4, dtype=np.int16))  # the microphone keeps reading meanwhile

    detector.handlers = {"yes": answer_request}

    detector._inference_loop()

    assert calls == ["yes"]
    assert len(detector._buffer.pull(timeout=0)) == 0


def test_inference_loop_returns_while_no_window_is_available():
    """One loop iteration must not wait forever on an empty buffer, or the worker could never notice a stop."""
    # Bypass __init__: no microphone and no model runner are needed to pull from the window buffer
    detector = AudioDetector.__new__(AudioDetector)
    detector._buffer = SlidingWindowBuffer(window_size=16000, slide_amount=16000)

    iteration = threading.Thread(target=detector._inference_loop, daemon=True)
    iteration.start()
    iteration.join(timeout=2)

    assert not iteration.is_alive()
