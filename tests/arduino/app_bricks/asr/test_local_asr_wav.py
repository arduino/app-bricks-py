# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the offline (in-memory) ASR brick (``WAVAutomaticSpeechRecognition``)
and its helper ``InMemoryAudioSource``."""

import numpy as np
import pytest

from arduino.app_bricks.asr import (
    ASREvent,
    WAVAutomaticSpeechRecognition,
)
from arduino.app_bricks.asr.local_asr_wav import InMemoryAudioSource
from arduino.app_bricks.asr.local_asr import AudioSourceExhausted

from conftest import (
    _FakeMic,
    _mock_transcribe_stream,
    _wav_bytes,
)


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


class TestConstructor:
    """Constructor semantics of WAVAutomaticSpeechRecognition."""

    def test_wav_bytes_are_wrapped(self):
        asr = WAVAutomaticSpeechRecognition(wav=_wav_bytes(np.zeros(10, dtype=np.int16)))
        assert isinstance(asr._source, InMemoryAudioSource)
        assert asr._owns_source is False

    def test_ndarray_are_wrapped(self):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(100, dtype=np.int16))
        assert isinstance(asr._source, InMemoryAudioSource)
        assert asr._owns_source is False

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError):
            WAVAutomaticSpeechRecognition(wav=42)  # type: ignore[arg-type]

    def test_microphone_rejected(self):
        with pytest.raises(TypeError):
            WAVAutomaticSpeechRecognition(wav=_FakeMic())  # type: ignore[arg-type]

    def test_unsupported_dtype_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            WAVAutomaticSpeechRecognition(wav=np.zeros(4, dtype="<c8"))  # complex


class TestSourceStartedCheck:
    """The eager source-started check fires on the WAV brick's public surface."""

    @pytest.fixture
    def stopped_asr(self):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        asr._source.stop()
        return asr

    def test_transcribe(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe()

    def test_transcribe_stream(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_stream()


class TestTranscribe:
    """Covers the shared ``_collect_transcription`` helper through the WAV brick's no-arg ``transcribe()``."""

    def test_concatenates_full_text(self, monkeypatch):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(
            monkeypatch,
            asr,
            [
                ASREvent("partial_text", "hel"),
                ASREvent("full_text", "hello "),
                ASREvent("full_text", "world"),
            ],
        )
        assert asr.transcribe() == "hello world"

    def test_falls_back_to_last_partial_when_no_full_text(self, monkeypatch):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(
            monkeypatch,
            asr,
            [
                ASREvent("partial_text", "hi"),
                ASREvent("partial_text", "hello world"),
            ],
        )
        assert asr.transcribe() == "hello world"

    def test_returns_empty_when_no_speech(self, monkeypatch):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        _mock_transcribe_stream(monkeypatch, asr, [])
        assert asr.transcribe() == ""


class TestIdleState:
    """Idle-state introspection inherited from ``_ASRBase``."""

    def test_fresh_instance_is_not_transcribing(self):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        assert asr.is_transcribing() is False

    def test_cancel_on_idle_is_noop(self):
        asr = WAVAutomaticSpeechRecognition(wav=np.zeros(10, dtype=np.int16))
        asr.cancel()  # must not raise
        assert asr.is_transcribing() is False
