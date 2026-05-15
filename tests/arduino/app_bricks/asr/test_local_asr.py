# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import io
import threading
import wave

import numpy as np
import pytest

from arduino.app_bricks.asr import (
    ASREvent,
    AutomaticSpeechRecognition,
    TranscriptionStream,
)
from arduino.app_bricks.asr.local_asr import (
    AudioSourceExhausted,
    InMemoryAudioSource,
)
from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone


# HELPERS


@pytest.fixture(autouse=True)
def _patch_brick_lookup(monkeypatch: pytest.MonkeyPatch):
    """Avoid hitting the real service-discovery"""
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.get_brick_config", lambda cls: {"id": None, "model": "test-model"})
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.get_brick_configured_model", lambda _id: None)


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(samples.dtype.itemsize)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _mock_transcribe_stream(monkeypatch, asr, events):
    """Replace asr._transcribe_stream with a generator yielding `events`,
    while recording the kwargs it was called with."""
    seen: dict = {}

    def fake(duration=0, vad_ms=None):
        seen["duration"] = duration
        seen["vad_ms"] = vad_ms
        yield from events

    monkeypatch.setattr(asr, "_transcribe_stream", fake)
    return seen


class _FakeMic(BaseMicrophone):
    """Minimal BaseMicrophone that yields nothing — used as an inert source."""

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


# TESTS


class TestInMemoryAudioSource:
    def test_from_wav_bytes_parses_metadata(self):
        samples = np.arange(1000, dtype=np.int16)
        src = InMemoryAudioSource(_wav_bytes(samples, sample_rate=22050))
        assert src.sample_rate == 22050
        assert src.channels == 1
        assert src.format == np.dtype(np.int16)
        assert src.is_started() is True

    def test_from_ndarray_uses_defaults(self):
        samples = np.zeros(100, dtype=np.int16)
        src = InMemoryAudioSource(samples)
        assert src.sample_rate == 16000
        assert src.channels == 1
        assert src.format == np.dtype(np.int16)

    def test_capture_yields_then_raises_exhausted(self):
        # buffer_size defaults to 1024 → expect 2 captures from 2048 samples
        src = InMemoryAudioSource(np.zeros(2048, dtype=np.int16))
        assert len(src.capture()) == 1024
        assert len(src.capture()) == 1024
        with pytest.raises(AudioSourceExhausted):
            src.capture()

    def test_start_stop_toggle_state(self):
        src = InMemoryAudioSource(np.zeros(10, dtype=np.int16))
        assert src.is_started()
        src.stop()
        assert not src.is_started()
        src.start()
        assert src.is_started()

    def test_unsupported_source_type_raises(self):
        with pytest.raises(TypeError):
            InMemoryAudioSource(42)  # type: ignore[arg-type]


class TestTranscriptionStream:
    def test_iterates_and_closes_on_context_exit(self):
        closed = threading.Event()

        def gen():
            try:
                yield 1
                yield 2
                yield 3
            finally:
                closed.set()

        with TranscriptionStream(gen()) as stream:
            assert next(stream) == 1
            assert next(stream) == 2
        assert closed.is_set()

    def test_close_propagates_on_exception(self):
        closed = threading.Event()

        def gen():
            try:
                yield 1
            finally:
                closed.set()

        with pytest.raises(RuntimeError, match="oops!"):
            with TranscriptionStream(gen()) as stream:
                next(stream)
                raise RuntimeError("oops!")
        assert closed.is_set()


class TestConstructor:
    def test_base_microphone_is_not_owned(self):
        mic = _FakeMic()
        asr = AutomaticSpeechRecognition(source=mic)
        assert asr._source is mic
        assert asr._owns_source is False

    def test_wav_bytes_are_wrapped(self):
        asr = AutomaticSpeechRecognition(source=_wav_bytes(np.zeros(10, dtype=np.int16)))
        assert isinstance(asr._source, InMemoryAudioSource)
        assert asr._owns_source is False

    def test_ndarray_are_wrapped(self):
        asr = AutomaticSpeechRecognition(source=np.zeros(100, dtype=np.int16))
        assert isinstance(asr._source, InMemoryAudioSource)
        assert asr._owns_source is False

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError):
            AutomaticSpeechRecognition(source=42)  # type: ignore[arg-type]
    
    def test_unsupported_dtype_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            AutomaticSpeechRecognition(source=np.zeros(4, dtype="<c8"))  # complex


class TestSourceStartedCheck:
    @pytest.fixture
    def stopped_asr(self):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        asr._source.stop()
        return asr

    def test_transcribe(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe()

    def test_transcribe_stream(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_stream()

    def test_transcribe_sentence(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_sentence()

    def test_transcribe_sentence_stream(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_sentence_stream()

    def test_transcribe_continuous(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_continuous()

    def test_transcribe_continuous_stream(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_continuous_stream()


class TestTranscribe:
    def test_concatenates_full_text(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [
            ASREvent("partial_text", "hel"),
            ASREvent("full_text", "hello "),
            ASREvent("full_text", "world"),
        ])
        assert asr.transcribe() == "hello world"

    def test_falls_back_to_last_partial_when_no_full_text(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [
            ASREvent("partial_text", "hi"),
            ASREvent("partial_text", "hello world"),
        ])
        assert asr.transcribe() == "hello world"

    def test_returns_empty_when_no_speech(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [])
        assert asr.transcribe() == ""


class TestTranscribeSentence:
    def test_returns_first_full_text_and_stops(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        consumed = []

        def fake(duration=0, vad_ms=None):
            for ev in [
                ASREvent("partial_text", "hel"),
                ASREvent("partial_text", "hello"),
                ASREvent("full_text", "hello"),
                ASREvent("full_text", "world"),  # must not be yielded
            ]:
                consumed.append(ev)
                yield ev

        monkeypatch.setattr(asr, "_transcribe_stream", fake)
        assert asr.transcribe_sentence() == "hello"
        # Only the first three events should have been pulled before close.
        assert [e.data for e in consumed] == ["hel", "hello", "hello"]

    def test_falls_back_to_last_partial_when_source_exhausts(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [
            ASREvent("partial_text", "hello"),
            ASREvent("partial_text", "hello world"),
        ])
        assert asr.transcribe_sentence() == "hello world"

    def test_passes_hangover_as_vad_ms(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        seen = _mock_transcribe_stream(monkeypatch, asr, [ASREvent("full_text", "ok")])
        asr.transcribe_sentence(hangover=350, timeout=12)
        assert seen["vad_ms"] == 350
        assert seen["duration"] == 12

    def test_empty_full_text_does_not_terminate_stream(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [
            ASREvent("full_text", "   "),  # blank — should not stop
            ASREvent("partial_text", "hi"),
            ASREvent("full_text", "hi there"),
        ])
        assert asr.transcribe_sentence() == "hi there"


class TestTranscribeContinuous:
    def test_yields_non_empty_full_text_only(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [
            ASREvent("partial_text", "hi"),
            ASREvent("full_text", "hi"),
            ASREvent("partial_text", "there"),
            ASREvent("full_text", "   "),  # filtered out
            ASREvent("full_text", "there"),
        ])
        with asr.transcribe_continuous() as sentences:
            collected = list(sentences)
        assert collected == ["hi", "there"]
        assert not asr.is_transcribing()

    def test_break_closes_underlying_stream(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        inner_closed = threading.Event()

        def fake(duration=0, vad_ms=None):
            try:
                for ev in [ASREvent("full_text", "one"), ASREvent("full_text", "two")]:
                    yield ev
            finally:
                inner_closed.set()

        monkeypatch.setattr(asr, "_transcribe_stream", fake)
        with asr.transcribe_continuous() as sentences:
            for sentence in sentences:
                assert sentence == "one"
                break
        assert inner_closed.is_set()
        assert not asr.is_transcribing()

    def test_timeout_passed_as_duration(self, monkeypatch):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        seen = _mock_transcribe_stream(monkeypatch, asr, [])
        with asr.transcribe_continuous(timeout=42) as sentences:
            list(sentences)
        assert seen["duration"] == 42
        assert not asr.is_transcribing()


class TestIdleState:
    def test_fresh_instance_is_not_transcribing(self):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        assert asr.is_transcribing() is False

    def test_cancel_on_idle_is_noop(self):
        asr = AutomaticSpeechRecognition(source=np.zeros(10, dtype=np.int16))
        asr.cancel()  # must not raise
        assert asr.is_transcribing() is False
