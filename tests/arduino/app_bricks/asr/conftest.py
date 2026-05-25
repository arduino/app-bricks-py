# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import io
import wave

import numpy as np
import pytest

from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone


@pytest.fixture(autouse=True)
def _patch_brick_lookup(monkeypatch: pytest.MonkeyPatch):
    """Avoid hitting the real service-discovery."""
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.get_brick_config", lambda cls: {"id": None, "model": "test-model"})
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.get_brick_configured_model", lambda _id: None)


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Build a WAV-container ``bytes`` payload around the given samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(samples.dtype.itemsize)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _mock_transcribe_stream(monkeypatch, asr, events):
    """
    Replace ``asr._transcribe_stream`` with a generator yielding ``events``,
    while recording the kwargs it was called with.
    """
    seen: dict = {}

    def fake(duration=0, vad_ms=None):
        seen["duration"] = duration
        seen["vad_ms"] = vad_ms
        yield from events

    monkeypatch.setattr(asr, "_transcribe_stream", fake)
    return seen


class _FakeMic(BaseMicrophone):
    """Minimal BaseMicrophone that yields nothing, used as an inert source."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, format=np.int16, buffer_size: int = 1024):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            format=format,
            buffer_size=buffer_size,
            auto_reconnect=False,
        )

    def _open_microphone(self):  # pragma: no cover - trivial
        pass

    def _close_microphone(self):  # pragma: no cover - trivial
        pass

    def _read_audio(self):  # pragma: no cover - trivial
        return None


def _started_mic() -> _FakeMic:
    """A ``_FakeMic`` that already passes the source-started check."""
    mic = _FakeMic()
    mic.start()
    return mic
