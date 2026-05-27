# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import queue
from typing import Iterable, List

import numpy as np
import pytest

from arduino.app_bricks.cloud_asr import CloudASR, CloudProvider
from arduino.app_bricks.cloud_asr.cloud_asr import TranscriptionStreamError
from arduino.app_bricks.cloud_asr.providers import ASRProviderEvent, ASRProviderError
from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone


class MockMicrophone(BaseMicrophone):
    """Lightweight microphone stub that yields pre-loaded chunks."""

    def __init__(
        self,
        chunks: Iterable,
        sample_rate: int = 16000,
        channels: int = 1,
        format: type | np.dtype | str = np.int16,
        buffer_size: int = 1024,
        auto_reconnect: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, channels=channels, format=format, buffer_size=buffer_size, auto_reconnect=auto_reconnect)
        self._chunks: List = list(chunks)

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        if not self._chunks:
            return None
        return self._chunks.pop(0)


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
    mic.start()
    provider = make_provider(events=[ASRProviderEvent(type="text", data="mock")])
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            next(stream)
            assert provider.start_called is True

        assert provider.stop_called is True
    finally:
        asr.stop()
        mic.stop()


def test_transcribe_stream_aggregates_partial_text_in_append_mode(make_provider):
    events = [
        ASRProviderEvent(type="partial_text", data="Hel"),
        ASRProviderEvent(type="partial_text", data="lo"),
        ASRProviderEvent(type="text", data=None),
    ]
    audio_chunks = [np.array([1, 2, 3], dtype=np.int16), None, np.array([4, 5, 6], dtype=np.int16)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
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
        asr.stop()
        mic.stop()

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
    mic = MockMicrophone(audio_chunks)
    mic.start()
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
        asr.stop()
        mic.stop()

    assert provider.start_called is True
    assert [msg.type for msg in results] == ["partial_text", "partial_text", "text", "partial_text", "text"]
    assert results[2].data == "due"
    assert results[4].data == "tre"
    assert provider.stop_called is True


def test_transcribe_sentence_returns_first_text(make_provider):
    events = [
        ASRProviderEvent(type="partial_text", data="Hel"),
        ASRProviderEvent(type="text", data="Hello"),
        ASRProviderEvent(type="text", data="Should not be returned"),
    ]
    audio_chunks = [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5, 6], dtype=np.int16)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
    provider = make_provider(events=events, audio_chunks_len=2)
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        result = asr.transcribe_sentence()
    finally:
        asr.stop()
        mic.stop()

    assert result == "Hello"
    assert provider.start_called is True
    assert provider.stop_called is True


def test_transcribe_sentence_stream_stops_at_first_text(make_provider):
    events = [
        ASRProviderEvent(type="partial_text", data="Hel"),
        ASRProviderEvent(type="partial_text", data="lo"),
        ASRProviderEvent(type="text", data=None),
        ASRProviderEvent(type="text", data="Should not appear"),
    ]
    audio_chunks = [np.array([1, 2, 3], dtype=np.int16) for _ in range(4)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
    make_provider(events=events, partial_mode="append", audio_chunks_len=4)
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_sentence_stream() as stream:
            seen = list(stream)
    finally:
        asr.stop()
        mic.stop()

    assert [e.type for e in seen] == ["partial_text", "partial_text", "text"]
    assert seen[-1].data == "Hello"


def test_transcribe_until_cancelled_yields_sentences_and_stops_on_cancel(make_provider):
    events = [
        ASRProviderEvent(type="text", data="first"),
        ASRProviderEvent(type="text", data="second"),
    ]
    audio_chunks = [np.array([1, 2, 3], dtype=np.int16) for _ in range(5)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
    make_provider(events=events, audio_chunks_len=2)
    asr = CloudASR(
        api_key="dummy",
        mic=mic,
        provider=CloudProvider.OPENAI_TRANSCRIBE,
        silence_timeout=60.0,
    )

    try:
        with asr.transcribe_until_cancelled() as stream:
            first = next(stream)
            second = next(stream)
            asr.cancel()
            remaining = list(stream)
    finally:
        asr.stop()
        mic.stop()

    assert first == "first"
    assert second == "second"
    assert remaining == []


def test_is_transcribing_reflects_session_state(make_provider):
    events = [ASRProviderEvent(type="text", data="hello")]
    audio_chunks = [np.array([1, 2], dtype=np.int16)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
    make_provider(events=events, audio_chunks_len=1)
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        assert asr.is_transcribing() is False
        with asr.transcribe_stream() as stream:
            next(stream, None)
            assert asr.is_transcribing() is True
        assert asr.is_transcribing() is False
    finally:
        asr.stop()
        mic.stop()


def test_concurrent_session_raises(make_provider):
    events = [ASRProviderEvent(type="text", data="ok")]
    audio_chunks = [np.array([1, 2], dtype=np.int16)]
    mic = MockMicrophone(audio_chunks)
    mic.start()
    make_provider(events=events, audio_chunks_len=1)
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            next(stream, None)
            with pytest.raises(TranscriptionStreamError, match="already active"):
                asr.transcribe(duration=1.0)
    finally:
        asr.stop()
        mic.stop()


def test_transcribe_stream_surfaces_provider_errors(monkeypatch: pytest.MonkeyPatch):
    class FailingProvider(DummyProvider):
        def recv(self):
            raise ASRProviderError("boom")

    provider = FailingProvider()
    monkeypatch.setattr("arduino.app_bricks.cloud_asr.cloud_asr.provider_factory", lambda *, api_key, name, language, sample_rate: provider)

    mic = MockMicrophone(
        chunks=[np.array([7, 8], dtype=np.int16), np.array([9, 10], dtype=np.int16)],
    )
    mic.start()
    asr = CloudASR(api_key="dummy", mic=mic, provider=CloudProvider.OPENAI_TRANSCRIBE)

    try:
        with asr.transcribe_stream() as stream:
            next(stream)
    except Exception as exc:
        assert isinstance(exc, ASRProviderError)
        assert str(exc) == "boom"
    finally:
        asr.stop()
        mic.stop()

    assert provider.start_called is True
    assert provider.stop_called is True
