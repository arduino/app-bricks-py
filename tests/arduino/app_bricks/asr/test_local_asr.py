# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import threading

import numpy as np
import pytest

from arduino.app_bricks.asr import (
    ASREvent,
    ASRUnavailableError,
    AutomaticSpeechRecognition,
    TranscriptionStream,
)

from conftest import (
    _FakeMic,
    _mock_session_endpoints,
    _mock_transcribe_stream,
    _started_mic,
    _wav_bytes,
)


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
    """Constructor semantics of AutomaticSpeechRecognition (mic brick)."""

    def test_base_microphone_is_not_owned(self):
        mic = _FakeMic()
        asr = AutomaticSpeechRecognition(mic=mic)
        assert asr._source is mic
        assert asr._owns_source is False

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError):
            AutomaticSpeechRecognition(mic=42)  # type: ignore[arg-type]

    def test_bytes_rejected(self):
        with pytest.raises(TypeError):
            AutomaticSpeechRecognition(
                mic=_wav_bytes(np.zeros(10, dtype=np.int16))  # type: ignore[arg-type]
            )

    def test_ndarray_rejected(self):
        with pytest.raises(TypeError):
            AutomaticSpeechRecognition(mic=np.zeros(10, dtype=np.int16))  # type: ignore[arg-type]


class TestSourceStartedCheck:
    """The eager source-started check fires for every public transcribe* method on the mic brick."""

    @pytest.fixture
    def stopped_asr(self):
        # _FakeMic is created in the un-started state.
        return AutomaticSpeechRecognition(mic=_FakeMic())

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

    def test_transcribe_until_cancelled(self, stopped_asr):
        with pytest.raises(RuntimeError, match="started"):
            stopped_asr.transcribe_until_cancelled()


class TestTranscribe:
    """Mic-brick ``transcribe(duration=...)`` forwards the duration to the underlying generator."""

    def test_duration_passed_through(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        seen = _mock_transcribe_stream(monkeypatch, asr, [ASREvent("full_text", "hi")])
        assert asr.transcribe(duration=7) == "hi"
        assert seen["duration"] == 7


class TestTranscribeSentence:
    def test_returns_first_full_text_and_stops(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
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
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        _mock_transcribe_stream(
            monkeypatch,
            asr,
            [
                ASREvent("partial_text", "hello"),
                ASREvent("partial_text", "hello world"),
            ],
        )
        assert asr.transcribe_sentence() == "hello world"

    def test_timeout_passed_as_duration(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        seen = _mock_transcribe_stream(monkeypatch, asr, [ASREvent("full_text", "ok")])
        asr.transcribe_sentence(timeout=12)
        assert seen["duration"] == 12

    def test_empty_full_text_does_not_terminate_stream(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        _mock_transcribe_stream(
            monkeypatch,
            asr,
            [
                ASREvent("full_text", "   "),  # blank — should not stop
                ASREvent("partial_text", "hi"),
                ASREvent("full_text", "hi there"),
            ],
        )
        assert asr.transcribe_sentence() == "hi there"


class TestTranscribeUntilCancelled:
    def test_yields_event_stream(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        events = [
            ASREvent("partial_text", "hi"),
            ASREvent("full_text", "hi"),
            ASREvent("partial_text", "there"),
            ASREvent("full_text", "there"),
        ]
        _mock_transcribe_stream(monkeypatch, asr, events)
        with asr.transcribe_until_cancelled() as stream:
            collected = list(stream)
        assert collected == events
        assert not asr.is_transcribing()

    def test_break_closes_underlying_stream(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        inner_closed = threading.Event()

        def fake(duration=0, vad_ms=None):
            try:
                yield from [ASREvent("full_text", "one"), ASREvent("full_text", "two")]
            finally:
                inner_closed.set()

        monkeypatch.setattr(asr, "_transcribe_stream", fake)
        with asr.transcribe_until_cancelled() as stream:
            for event in stream:
                assert event.data == "one"
                break
        assert inner_closed.is_set()
        assert not asr.is_transcribing()

    def test_calls_underlying_generator_unbounded(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        seen = _mock_transcribe_stream(monkeypatch, asr, [])
        with asr.transcribe_until_cancelled() as stream:
            list(stream)
        assert seen["duration"] == 0
        assert not asr.is_transcribing()


class TestTranslate:
    """The ``translate`` flag reaches the inference server on every session."""

    def test_defaults_to_false(self):
        asr = AutomaticSpeechRecognition(mic=_FakeMic())
        assert asr.translate is False

    def test_constructor_value_is_exposed(self):
        asr = AutomaticSpeechRecognition(mic=_FakeMic(), translate=True)
        assert asr.translate is True

    def test_session_body_carries_the_flag(self, monkeypatch):
        bodies = _mock_session_endpoints(monkeypatch)
        asr = AutomaticSpeechRecognition(mic=_FakeMic(), translate=True)

        asr._create_transcription_session(translate=asr.translate)

        assert bodies[0]["translate"] is True

    def test_session_body_carries_the_flag_when_off(self, monkeypatch):
        bodies = _mock_session_endpoints(monkeypatch)
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        asr._create_transcription_session(translate=asr.translate)

        # Sent explicitly rather than omitted: the server reuses one ASR engine
        # across sessions, so an absent flag would leave the previous one in place.
        assert bodies[0]["translate"] is False

    def test_warmup_uses_the_current_value(self, monkeypatch):
        bodies = _mock_session_endpoints(monkeypatch)
        asr = AutomaticSpeechRecognition(mic=_FakeMic(), translate=True)

        asr._warmup()

        assert bodies[0]["translate"] is True

    def test_reassignment_applies_to_the_next_session(self, monkeypatch):
        bodies = _mock_session_endpoints(monkeypatch)
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        asr._warmup()
        asr.translate = True
        asr._warmup()

        create_bodies = [b for b in bodies if "model" in b]
        assert [b["translate"] for b in create_bodies] == [False, True]

    def test_streaming_session_snapshots_the_flag(self, monkeypatch):
        """The streaming path reads ``self.translate`` when it creates the session."""
        seen: dict = {}

        def fake_create(vad_ms=None, language=None, translate=False):
            seen["translate"] = translate
            raise ASRUnavailableError("stop here")

        asr = AutomaticSpeechRecognition(mic=_started_mic(), translate=True)
        monkeypatch.setattr(asr, "_create_transcription_session", fake_create)
        # _transcribe_stream waits on the worker loop before doing anything else.
        loop = asyncio.new_event_loop()
        asr._worker_loop.set_result(loop)
        try:
            with pytest.raises(ASRUnavailableError):
                next(iter(asr.transcribe_stream()))
        finally:
            loop.close()

        assert seen["translate"] is True
