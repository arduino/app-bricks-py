# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Early transcripts are requested at pauses in the audio, not on a fixed clock:
a flush in the middle of a word makes the server transcribe half of it."""

import asyncio
import json
import queue
import threading
import time

import numpy as np
import pytest

from arduino.app_bricks.asr import WAVAutomaticSpeechRecognition
from arduino.app_peripherals.microphone import chunk_level
from arduino.app_bricks.asr.local_asr import (
    _END_SENTINEL,
    _FLUSH_MARK,
    SessionInfo,
    _PauseFlushPolicy,
)

RATE = 16000
CHUNK = 1024  # Microphone() default buffer size: 64 ms at 16 kHz
CHUNK_S = CHUNK / RATE
SPEECH = 10 ** (-20 / 20)  # -20 dBFS syllables...
DIP = 10 ** (-32 / 20)  # ...with 12 dB dips between them, one chunk every four
NOISE = 10 ** (-80 / 20)  # quiet room


def _policy(**kw) -> _PauseFlushPolicy:
    args = {"min_s": 3.0, "max_s": 10.0, "pause_s": 0.25, "min_voiced_s": 0.5}
    args.update(kw)
    return _PauseFlushPolicy(**args)


def _feed(policy: _PauseFlushPolicy, pattern: list[tuple[float, float]]) -> list[float]:
    """Feed (level, seconds) runs chunk by chunk; return the times at which a flush was due.
    A SPEECH run is modulated like real speech: syllables with short dips."""
    t, flushes = 0.0, []
    for level, seconds in pattern:
        for i in range(round(seconds / CHUNK_S)):
            t += CHUNK_S
            chunk_level = DIP if level == SPEECH and i % 4 == 3 else level
            if policy.update(chunk_level, CHUNK_S):
                flushes.append(round(t, 3))
    return flushes


class TestChunkLevel:
    def test_int16_full_scale_is_one(self):
        chunk = np.full(CHUNK, 32767, dtype=np.int16)
        assert chunk_level(chunk) == pytest.approx(1.0, abs=1e-4)

    def test_int16_silence_is_zero(self):
        assert chunk_level(np.zeros(CHUNK, dtype=np.int16)) == 0.0

    def test_uint8_midpoint_is_silence(self):
        assert chunk_level(np.full(CHUNK, 128, dtype=np.uint8)) == 0.0

    def test_float32_is_taken_as_is(self):
        assert chunk_level(np.full(CHUNK, 0.5, dtype=np.float32)) == pytest.approx(0.5)

    def test_packed_24_bit_in_int32_uses_24_bit_scale(self):
        chunk = np.full(CHUNK, 2**23 - 1, dtype=np.int32)
        assert chunk_level(chunk, is_packed=True) == pytest.approx(1.0, abs=1e-4)

    def test_empty_chunk(self):
        assert chunk_level(np.zeros(0, dtype=np.int16)) == 0.0


class TestPauseFlushPolicy:
    def test_flushes_at_first_pause_after_min(self):
        # 4 s of speech, then a breath: the cut lands 0.25 s into the pause
        flushes = _feed(_policy(), [(NOISE, 1.0), (SPEECH, 4.0), (NOISE, 1.0)])
        assert len(flushes) == 1
        # runs are rounded to whole chunks, hence the one-chunk tolerance
        assert 5.0 + 0.25 - CHUNK_S <= flushes[0] <= 5.0 + 0.25 + 2 * CHUNK_S

    def test_ignores_pauses_before_min(self):
        # a pause 1 s into the segment is too early; the next one, after 3 s, is used
        flushes = _feed(_policy(), [(SPEECH, 1.0), (NOISE, 0.5), (SPEECH, 2.5), (NOISE, 0.5)])
        assert len(flushes) == 1
        assert flushes[0] > 4.0

    def test_short_gap_is_not_a_pause(self):
        # gaps shorter than pause_s happen inside words (stop consonants)
        pattern = [(SPEECH, 4.0), (NOISE, 0.128), (SPEECH, 2.0)]
        assert _feed(_policy(), pattern) == []

    def test_forces_a_cut_without_pauses(self):
        flushes = _feed(_policy(), [(SPEECH, 25.0)])
        assert len(flushes) == 2
        # speech from the very first chunk: the segment starts at the first syllable
        # dip, when there is a floor to compare with, so the cut may slip a little
        assert 10.0 <= flushes[0] <= 10.0 + 10 * CHUNK_S
        assert 20.0 <= flushes[1] <= 20.0 + 10 * CHUNK_S

    def test_silence_alone_never_flushes(self):
        assert _feed(_policy(), [(NOISE, 60.0)]) == []

    def test_digital_silence_is_a_pause(self):
        assert len(_feed(_policy(), [(SPEECH, 4.0), (0.0, 1.0)])) == 1

    def test_adapts_to_a_noisy_room(self):
        # -45 dBFS of background noise must still read as a pause next to -20 dBFS speech
        room = 10 ** (-45 / 20)
        flushes = _feed(_policy(), [(room, 5.0), (SPEECH, 4.0), (room, 1.0)])
        assert len(flushes) == 1

    def test_floor_does_not_climb_to_speech_level(self):
        # 9 s of continuous speech must not be taken for background noise
        flushes = _feed(_policy(), [(NOISE, 1.0), (SPEECH, 9.0)])
        assert flushes == []

    def test_reset_starts_a_new_segment(self):
        policy = _policy()
        _feed(policy, [(SPEECH, 2.5)])
        policy.reset()  # the server closed the segment on its own
        assert _feed(policy, [(SPEECH, 1.0), (NOISE, 1.0)]) == []


def _tone(seconds: float, level: float) -> np.ndarray:
    """A 220 Hz tone at ``level`` RMS, dipping 12 dB for one chunk in four like syllables."""
    n = round(seconds * RATE)
    envelope = np.where((np.arange(n) // CHUNK) % 4 == 3, DIP / SPEECH, 1.0)
    return (level * envelope * 32767 * np.sqrt(2) * np.sin(2 * np.pi * 220 * np.arange(n) / RATE)).astype(np.int16)


def _run_reader(asr, before_chunk=None) -> list[object]:
    """Run the reader thread over the whole WAV source and return what it queued."""
    session = SessionInfo(
        session_id="s",
        duration=0,
        start_time=time.time(),
        result_queue=queue.Queue(),
        chunk_queue=queue.Queue(),
        cancelled=threading.Event(),
    )
    if before_chunk is not None:
        capture = asr._source.capture
        count = [0]

        def hooked():
            before_chunk(count[0], session)
            count[0] += 1
            return capture()

        asr._source.capture = hooked
    asr._reader_thread_body(session)
    items = []
    while not session.chunk_queue.empty():
        items.append(session.chunk_queue.get_nowait())
    return items


class TestReaderQueuesFlushMarks:
    def test_mark_follows_the_pause_chunk(self):
        pcm = np.concatenate([_tone(4.0, SPEECH), np.zeros(RATE // 2, np.int16), _tone(2.0, SPEECH)])
        items = _run_reader(WAVAutomaticSpeechRecognition(pcm))

        assert items[-1] is _END_SENTINEL
        marks = [i for i, x in enumerate(items) if x is _FLUSH_MARK]
        assert len(marks) == 1
        audio_before = sum(len(x) for x in items[: marks[0]] if isinstance(x, bytes)) // 2 / RATE
        # inside the pause (4.0 s - 4.5 s), about 0.25 s into it; the chunk that
        # straddles the end of speech is half silence and already counts as pause
        assert 4.25 - CHUNK_S <= audio_before <= 4.5
        # no audio is lost around the mark
        assert sum(len(x) for x in items if isinstance(x, bytes)) == pcm.nbytes

    def test_server_closed_segment_restarts_the_count(self):
        pcm = np.concatenate([_tone(2.5, SPEECH), _tone(1.0, SPEECH), np.zeros(RATE, np.int16)])

        def vad_end_at_2_5_s(index, session):
            if index == round(2.5 / CHUNK_S):
                session.segment_closed.set()

        items = _run_reader(WAVAutomaticSpeechRecognition(pcm), before_chunk=vad_end_at_2_5_s)

        # without the reset the pause at 3.5 s would be past the 3 s minimum
        assert _FLUSH_MARK not in items


class TestReceiverClosesSegments:
    def test_full_text_marks_the_segment_closed(self):
        session = SessionInfo(
            session_id="s",
            duration=0,
            start_time=time.time(),
            result_queue=queue.Queue(),
            chunk_queue=queue.Queue(),
            cancelled=threading.Event(),
        )
        messages = [
            {"type": "transcript.text.delta", "text": "ciao", "session_id": "s"},
            {"type": "transcript.text.done", "text": "ciao a tutti", "session_id": "s"},
        ]
        seen_after: list[bool] = []

        class _WS:
            async def recv(self):
                seen_after.append(session.segment_closed.is_set())
                if messages:
                    return json.dumps(messages.pop(0))
                session.cancelled.set()
                raise TimeoutError

        asr = WAVAutomaticSpeechRecognition(np.zeros(16, np.int16))
        asyncio.run(asr._receive_transcription(_WS(), session))

        # not set by the partial, set by the final
        assert seen_after == [False, False, True]
        assert session.result_queue.qsize() == 2
