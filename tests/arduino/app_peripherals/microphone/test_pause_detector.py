# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from arduino.app_peripherals.microphone import PauseDetector, chunk_level

RATE = 16000
CHUNK = 1024


def _chunk(dbfs: float | None, channels: int = 1) -> np.ndarray:
    """A 220 Hz int16 chunk at ``dbfs`` RMS, or digital silence for None."""
    n = CHUNK * channels
    if dbfs is None:
        return np.zeros(n, np.int16)
    t = np.arange(n) / (RATE * channels)
    return (10 ** (dbfs / 20) * np.sqrt(2) * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _speech(detector: PauseDetector, chunks: int, channels: int = 1) -> None:
    """Syllables at -20 dBFS with a -32 dBFS dip one chunk in four; odd counts end on a syllable."""
    for i in range(chunks):
        detector.update(_chunk(-32 if i % 4 == 3 else -20, channels), RATE, channels)


class TestChunkLevel:
    def test_int16_sine_level(self):
        assert chunk_level(_chunk(-20)) == pytest.approx(0.1, rel=0.02)

    def test_float_chunk(self):
        assert chunk_level(np.full(CHUNK, -0.25, np.float32)) == pytest.approx(0.25)


class TestPauseDetector:
    def test_starts_quiet_and_not_paused(self):
        detector = PauseDetector()
        assert detector.quiet is True
        assert detector.quiet_s == 0.0
        assert detector.paused is False

    def test_pause_after_speech(self):
        detector = PauseDetector(pause_s=0.25)
        _speech(detector, 41)
        assert detector.paused is False

        for _ in range(3):  # 192 ms
            assert detector.update(_chunk(-80), RATE) is True
        assert detector.paused is False
        detector.update(_chunk(-80), RATE)  # 256 ms
        assert detector.paused is True
        assert detector.quiet_s == pytest.approx(4 * CHUNK / RATE)

    def test_speech_resets_the_quiet_run(self):
        detector = PauseDetector()
        _speech(detector, 41)
        for _ in range(5):
            detector.update(_chunk(None), RATE)
        assert detector.paused is True
        assert detector.update(_chunk(-20), RATE) is False
        assert detector.quiet_s == 0.0
        assert detector.paused is False

    def test_room_noise_alone_is_quiet(self):
        detector = PauseDetector()
        rng = np.random.default_rng(0)
        # -45 dBFS of background noise, well above the -70 dBFS absolute threshold
        noise = [(rng.normal(0, 10 ** (-45 / 20) * 32767, CHUNK)).astype(np.int16) for _ in range(50)]
        assert [detector.update(c, RATE) for c in noise] == [True] * 50

    def test_stereo_duration_accounts_for_channels(self):
        detector = PauseDetector(pause_s=0.25)
        _speech(detector, 41, channels=2)
        for _ in range(4):
            detector.update(_chunk(None, channels=2), RATE, channels=2)
        # four stereo chunks of 1024 frames each are 256 ms, not 512
        assert detector.quiet_s == pytest.approx(4 * CHUNK / RATE)
