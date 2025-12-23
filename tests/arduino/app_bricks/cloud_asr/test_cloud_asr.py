# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import queue
import sys
import threading
import time
import types
from typing import Iterable, List

import numpy as np
import pytest

# Provide a minimal alsaaudio stub so tests can import CloudASR without the native alsaaudio dependency.
dummy_pcm = types.SimpleNamespace(
    read=lambda *args, **kwargs: (0, b""),
    setchannels=lambda *args, **kwargs: None,
    setrate=lambda *args, **kwargs: None,
    setformat=lambda *args, **kwargs: None,
    setperiodsize=lambda *args, **kwargs: None,
    rate=lambda *args, **kwargs: 16000,
    close=lambda *args, **kwargs: None,
)
sys.modules["alsaaudio"] = types.SimpleNamespace(
    ALSAAudioError=Exception,
    PCM=lambda *args, **kwargs: dummy_pcm,
    PCM_CAPTURE=0,
    PCM_NORMAL=0,
    PCM_NONBLOCK=0,
    PCM_PLAYBACK=0,
    Mixer=lambda *args, **kwargs: types.SimpleNamespace(getvolume=lambda: [0], getrange=lambda: (0, 100)),
    mixers=lambda *args, **kwargs: [],
    cards=lambda: [],
    card_indexes=lambda: [],
    card_name=lambda idx: "",
    pcms=lambda *args, **kwargs: [],
    PCM_FORMAT_S8=0,
    PCM_FORMAT_U8=0,
    PCM_FORMAT_S16_LE=0,
    PCM_FORMAT_S16_BE=0,
    PCM_FORMAT_U16_LE=0,
    PCM_FORMAT_U16_BE=0,
    PCM_FORMAT_S24_LE=0,
    PCM_FORMAT_S24_BE=0,
    PCM_FORMAT_S24_3LE=0,
    PCM_FORMAT_S24_3BE=0,
    PCM_FORMAT_S32_LE=0,
    PCM_FORMAT_S32_BE=0,
    PCM_FORMAT_U32_LE=0,
    PCM_FORMAT_U32_BE=0,
    PCM_FORMAT_FLOAT_LE=0,
    PCM_FORMAT_FLOAT_BE=0,
    PCM_FORMAT_FLOAT64_LE=0,
    PCM_FORMAT_FLOAT64_BE=0,
    PCM_FORMAT_MU_LAW=0,
    PCM_FORMAT_A_LAW=0,
    PCM_FORMAT_IMA_ADPCM=0,
    PCM_FORMAT_MPEG=0,
    PCM_FORMAT_GSM=0,
)

from arduino.app_bricks.cloud_asr.cloud_asr import CloudASR
from arduino.app_bricks.cloud_asr.providers import CloudProvider
from arduino.app_bricks.cloud_asr.providers.types import ASREvent
from arduino.app_utils.app import App


class MockMicrophone:
    """Lightweight microphone stub that yields pre-loaded chunks."""

    def __init__(self, chunks: Iterable, sample_rate: int = 16000, delay_between_chunks: float = 0.0):
        self.sample_rate = sample_rate
        self.is_recording = threading.Event()
        self._chunks: List = list(chunks)
        self._delay = delay_between_chunks
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        self.is_recording.set()

    def stop(self):
        self.stop_calls += 1
        self.is_recording.clear()

    def stream(self):
        while self.is_recording.is_set() and self._chunks:
            if self._delay:
                time.sleep(self._delay)
            yield self._chunks.pop(0)


class DummyProvider:
    """ASR provider stub to drive CloudASR without network traffic."""

    def __init__(self, events: Iterable[ASREvent] | None = None, partial_mode: str = "append"):
        self.partial_mode = partial_mode
        self._events: queue.Queue[ASREvent] = queue.Queue()
        for ev in events or []:
            self._events.put(ev)
        self.sent_audio: list[bytes] = []
        self.stop_called = False

    def send_audio(self, pcm_chunk: bytes) -> None:
        self.sent_audio.append(pcm_chunk)

    def recv(self) -> ASREvent | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.stop_called = True


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch):
    def _factory(events: Iterable[ASREvent] | None = None, partial_mode: str = "append") -> DummyProvider:
        provider = DummyProvider(events=events, partial_mode=partial_mode)
        monkeypatch.setattr("arduino.app_bricks.cloud_asr.cloud_asr.provider_factory", lambda *, api_key, name, language, sample_rate: provider)
        return provider

    return _factory


def test_start_and_stop_use_microphone_state(make_provider):
    provider = make_provider()
    mic = MockMicrophone(chunks=[])
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        asr.start()
        asr.start()  # second call should be a no-op
        assert mic.start_calls == 1
        assert mic.is_recording.is_set()

        asr.stop()
        asr.stop()  # second call should be a no-op
        assert mic.stop_calls == 1
        assert not mic.is_recording.is_set()
        assert provider.stop_called is False
    finally:
        App.unregister(asr)


def test_transcribe_aggregates_partial_text_in_append_mode(make_provider):
    events = [
        ASREvent(event="partial_text", data="Hel"),
        ASREvent(event="partial_text", data="lo"),
        ASREvent(event="text", data=None),
    ]
    provider = make_provider(events=events, partial_mode="append")
    mic = MockMicrophone(
        chunks=[np.array([1, 2, 3], dtype=np.int16), None, np.array([4, 5, 6], dtype=np.int16)],
        delay_between_chunks=0.002,
    )
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    mic.start()
    try:
        results = list(asr.transcribe())
    finally:
        asr.stop()
        App.unregister(asr)

    assert [msg["event"] for msg in results] == ["partial_text", "partial_text", "text"]
    assert [msg["data"] for msg in results[:2]] == ["Hel", "lo"]
    assert results[-1]["data"] == "Hello"
    assert provider.sent_audio == [
        np.asarray([1, 2, 3], dtype=np.int16).tobytes(),
        np.asarray([4, 5, 6], dtype=np.int16).tobytes(),
    ]
    assert provider.stop_called is True


def test_transcribe_resets_partial_buffer_in_replace_mode(make_provider):
    events = [
        ASREvent(event="partial_text", data="uno"),
        ASREvent(event="partial_text", data="due"),
        ASREvent(event="text", data=None),
        ASREvent(event="partial_text", data="tre"),
        ASREvent(event="text", data=None),
    ]
    provider = make_provider(events=events, partial_mode="replace")
    mic = MockMicrophone(
        chunks=[np.ones(4, dtype=np.int16) for _ in range(5)],
        delay_between_chunks=0.002,
    )
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.GOOGLE_SPEECH)

    mic.start()
    try:
        results = list(asr.transcribe())
    finally:
        asr.stop()
        App.unregister(asr)

    assert [msg["event"] for msg in results] == ["partial_text", "partial_text", "text", "partial_text", "text"]
    assert results[2]["data"] == "due"
    assert results[4]["data"] == "tre"
    assert provider.stop_called is True


def test_transcribe_surfaces_provider_errors(monkeypatch: pytest.MonkeyPatch):
    class FailingProvider(DummyProvider):
        def recv(self):
            raise RuntimeError("boom")

    provider = FailingProvider()
    monkeypatch.setattr("arduino.app_bricks.cloud_asr.cloud_asr.provider_factory", lambda *, api_key, name, language, sample_rate: provider)

    mic = MockMicrophone(
        chunks=[np.array([7, 8], dtype=np.int16), np.array([9, 10], dtype=np.int16)],
        delay_between_chunks=0.001,
    )
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    mic.start()
    try:
        results = list(asr.transcribe())
    finally:
        asr.stop()
        App.unregister(asr)

    assert results[0]["event"] == "error"
    assert results[0]["data"] == "boom"
    assert provider.stop_called is True
