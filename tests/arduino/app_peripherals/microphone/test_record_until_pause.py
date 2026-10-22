# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""record_until_pause(): one spoken request, from the start of the recording to the speaker's pause."""

import numpy as np
import pytest

from arduino.app_peripherals.microphone import BaseMicrophone
from arduino.app_peripherals.microphone.errors import MicrophoneReadError

RATE = 16000
CHUNK = 1024  # 64 ms at 16 kHz
CHUNK_S = CHUNK / RATE
SPEECH = round(32767 * 10 ** (-20 / 20))  # -20 dBFS syllables...
DIP = round(32767 * 10 ** (-32 / 20))  # ...dipping 12 dB one chunk in four
NOISE = 3  # about -80 dBFS: a quiet room


class ScriptedMicrophone(BaseMicrophone):
    """Plays (amplitude, seconds) runs chunk by chunk, then a quiet room forever."""

    def __init__(self, *runs: tuple[int, float]):
        super().__init__(sample_rate=RATE, channels=1, format=np.int16, buffer_size=CHUNK, auto_reconnect=False)
        self._levels = [DIP if level == SPEECH and i % 4 == 3 else level for level, seconds in runs for i in range(round(seconds / CHUNK_S))]
        self.chunks_read = 0

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        level = self._levels[self.chunks_read] if self.chunks_read < len(self._levels) else NOISE
        self.chunks_read += 1
        return np.full(CHUNK, level, dtype=np.int16)


def _record(mic: ScriptedMicrophone, **kwargs) -> np.ndarray | None:
    with mic:
        return mic.record_until_pause(**kwargs)


def _seconds(recording: np.ndarray) -> float:
    return recording.size / RATE


def test_records_until_the_speaker_pauses():
    recording = _record(ScriptedMicrophone((NOISE, 0.5), (SPEECH, 1.5), (NOISE, 5.0)), pause_s=1.0)

    assert recording is not None
    # the silence before the speech is kept, the pause ends it; runs round to whole chunks
    assert 0.5 + 1.5 + 1.0 - CHUNK_S <= _seconds(recording) <= 0.5 + 1.5 + 1.0 + 2 * CHUNK_S
    assert recording.dtype == np.int16


def test_a_short_gap_inside_the_speech_does_not_end_it():
    recording = _record(ScriptedMicrophone((NOISE, 0.5), (SPEECH, 1.0), (NOISE, 0.4), (SPEECH, 1.0), (NOISE, 5.0)), pause_s=1.0)

    assert recording is not None
    assert _seconds(recording) > 0.5 + 1.0 + 0.4 + 1.0


def test_nobody_speaking_returns_none_after_the_timeout():
    mic = ScriptedMicrophone((NOISE, 30.0))

    assert _record(mic, timeout_s=2.0) is None
    assert mic.chunks_read * CHUNK_S == pytest.approx(2.0, abs=2 * CHUNK_S)


def test_a_click_is_not_a_request():
    assert _record(ScriptedMicrophone((NOISE, 0.5), (SPEECH, 0.128), (NOISE, 10.0)), timeout_s=3.0) is None


def test_speech_without_pauses_ends_at_max_s():
    recording = _record(ScriptedMicrophone((NOISE, 0.5), (SPEECH, 20.0)), max_s=3.0)

    assert recording is not None
    assert 0.5 + 3.0 - CHUNK_S <= _seconds(recording) <= 0.5 + 3.0 + 2 * CHUNK_S


def test_chunks_read_while_waiting_are_not_recorded():
    """The spoken prompt plays for 1 s: its chunks only teach the detector the room noise."""
    mic = ScriptedMicrophone((NOISE, 1.0), (SPEECH, 1.5), (NOISE, 5.0))
    prompt_chunks = round(1.0 / CHUNK_S)

    recording = _record(mic, pause_s=1.0, wait_while=lambda: mic.chunks_read <= prompt_chunks)

    assert recording is not None
    assert 1.5 + 1.0 - CHUNK_S <= _seconds(recording) <= 1.5 + 1.0 + 2 * CHUNK_S


def test_a_microphone_that_is_not_started_is_refused():
    with pytest.raises(MicrophoneReadError, match="before starting it"):
        ScriptedMicrophone().record_until_pause()


@pytest.mark.parametrize("argument", ["pause_s", "max_s", "min_speech_s", "timeout_s"])
def test_durations_must_be_positive(argument):
    with pytest.raises(ValueError, match="> 0"):
        _record(ScriptedMicrophone(), **{argument: 0})
