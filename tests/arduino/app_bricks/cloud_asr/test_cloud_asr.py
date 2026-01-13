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
from arduino.app_bricks.cloud_asr.providers.types import ASRProviderEvent, ASRProviderError
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

    def __init__(self, events: Iterable[ASRProviderEvent] | None = None, partial_mode: str = "append", audio_chunks_len: int = 0):
        self.partial_mode = partial_mode
        self._events: queue.Queue[ASRProviderEvent] = queue.Queue()
        for ev in events or []:
            self._events.put(ev)
        self.sent_audio: list[bytes] = []
        self.start_called = False
        self.stop_called = False
        self.audio_chunks_len = audio_chunks_len

    def send_audio(self, pcm_chunk: bytes) -> None:
        self.sent_audio.append(pcm_chunk)

    def recv(self):
        if len(self.sent_audio) < self.audio_chunks_len:
            return None
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch):
    def _factory(
        events: Iterable[ASRProviderEvent] | None = None,
        partial_mode: str = "append",
        audio_chunks_len: int = 0,
    ) -> DummyProvider:
        provider = DummyProvider(events=events, partial_mode=partial_mode, audio_chunks_len=audio_chunks_len)
        monkeypatch.setattr("arduino.app_bricks.cloud_asr.cloud_asr.provider_factory", lambda *, api_key, name, language, sample_rate: provider)
        return provider

    return _factory


def test_transcribe_stream_use_microphone_state(make_provider):
    mic = MockMicrophone(chunks=[])
    provider = make_provider(events=[ASRProviderEvent(type="text", data="mock")])
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            next(stream)
            assert mic.start_calls == 1
            assert mic.is_recording.is_set()
            assert provider.start_called is True

        assert mic.stop_calls == 1
        assert not mic.is_recording.is_set()
        assert provider.stop_called is True
    finally:
        App.unregister(asr)


def test_transcribe_stream_aggregates_partial_text_in_append_mode(make_provider):
    events = [
        ASRProviderEvent(type="partial_text", data="Hel"),
        ASRProviderEvent(type="partial_text", data="lo"),
        ASRProviderEvent(type="text", data=None),
    ]
    audio_chunks = [np.array([1, 2, 3], dtype=np.int16), None, np.array([4, 5, 6], dtype=np.int16)]
    mic = MockMicrophone(
        chunks=audio_chunks,
        delay_between_chunks=0.002,
    )
    provider = make_provider(events=events, partial_mode="append", audio_chunks_len=sum(ch is not None for ch in audio_chunks))
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            results = []
            for ev in stream:
                results.append(ev)
                if ev.type == "text":
                    break
    finally:
        App.unregister(asr)

    assert provider.start_called is True
    assert [msg.type for msg in results] == ["partial_text", "partial_text", "text"]
    assert [msg.data for msg in results[:2]] == ["Hel", "lo"]
    assert results[-1].data == "Hello"
    assert provider.sent_audio == [
        np.asarray([1, 2, 3], dtype=np.int16).tobytes(),
        np.asarray([4, 5, 6], dtype=np.int16).tobytes(),
    ]
    assert provider.stop_called is True


def test_transcribe_stream_resets_partial_buffer_in_replace_mode(make_provider):
    events = [
        ASRProviderEvent(type="partial_text", data="uno"),
        ASRProviderEvent(type="partial_text", data="due"),
        ASRProviderEvent(type="text", data=None),
        ASRProviderEvent(type="partial_text", data="tre"),
        ASRProviderEvent(type="text", data=None),
    ]
    audio_chunks = [np.ones(4, dtype=np.int16) for _ in range(5)]
    mic = MockMicrophone(
        chunks=audio_chunks,
        delay_between_chunks=0.002,
    )
    provider = make_provider(events=events, partial_mode="replace", audio_chunks_len=sum(ch is not None for ch in audio_chunks))
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.GOOGLE_SPEECH)

    try:
        with asr.transcribe_stream() as stream:
            results = []
            text_count = 0
            for ev in stream:
                results.append(ev)
                if ev.type == "text":
                    text_count += 1
                if text_count == 2:
                    break
    finally:
        App.unregister(asr)

    assert provider.start_called is True
    assert [msg.type for msg in results] == ["partial_text", "partial_text", "text", "partial_text", "text"]
    assert results[2].data == "due"
    assert results[4].data == "tre"
    assert provider.stop_called is True


def test_transcribe_stream_surfaces_provider_errors(monkeypatch: pytest.MonkeyPatch):
    class FailingProvider(DummyProvider):
        def recv(self):
            raise ASRProviderError("boom")

    provider = FailingProvider()
    monkeypatch.setattr("arduino.app_bricks.cloud_asr.cloud_asr.provider_factory", lambda *, api_key, name, language, sample_rate: provider)

    mic = MockMicrophone(
        chunks=[np.array([7, 8], dtype=np.int16), np.array([9, 10], dtype=np.int16)],
        delay_between_chunks=0.001,
    )
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            next(stream)
    except Exception as exc:
        assert isinstance(exc, ASRProviderError)
        assert str(exc) == "boom"
    finally:
        App.unregister(asr)

    assert provider.start_called is True
    assert provider.stop_called is True
