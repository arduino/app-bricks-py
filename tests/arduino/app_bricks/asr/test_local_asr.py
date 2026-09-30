# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import threading
import time

import numpy as np
import pytest

from arduino.app_bricks.asr import (
    ASRBusyError,
    ASRError,
    ASREvent,
    ASRServiceBusyError,
    ASRUnavailableError,
    AutomaticSpeechRecognition,
    TranscriptionStream,
)

from conftest import (
    _FakeMic,
    _FakeResponse,
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

    def test_joins_partial_pieces_when_source_exhausts(self, monkeypatch):
        # partial_text events are consecutive pieces of the sentence, not revisions
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        _mock_transcribe_stream(
            monkeypatch,
            asr,
            [
                ASREvent("partial_text", " Alcuni festival dispongono"),
                ASREvent("partial_text", " di aree di campeggio."),
            ],
        )
        assert asr.transcribe_sentence() == " Alcuni festival dispongono di aree di campeggio."

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


class TestSessionCreate:
    """How ``_create_transcription_session`` maps the server's answers."""

    @staticmethod
    def _answer(monkeypatch, payload: dict, status_code: int) -> dict:
        seen: dict = {}

        def fake_post(url=None, json=None, timeout=None, **kwargs):
            seen["timeout"] = timeout
            return _FakeResponse(payload, status_code=status_code)

        monkeypatch.setattr("arduino.app_bricks.asr.local_asr.requests.post", fake_post)
        return seen

    def test_conflict_raises_service_busy(self, monkeypatch):
        # Body and status the audio-analytics 1.0.5 API returns while another session is open
        self._answer(
            monkeypatch,
            {
                "error": {
                    "message": "A transcription session is already active (pending-8eb7d19b). Please close it first via /transcriptions/close.",
                    "type": "server_error",
                    "code": "conflict",
                    "sessions": [],
                }
            },
            status_code=409,
        )
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        with pytest.raises(ASRServiceBusyError, match="already active"):
            asr._create_transcription_session()

    def test_conflict_without_message_still_raises_service_busy(self, monkeypatch):
        self._answer(monkeypatch, {}, status_code=409)
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        with pytest.raises(ASRServiceBusyError):
            asr._create_transcription_session()

    def test_engine_init_failure_is_not_reported_as_busy(self, monkeypatch):
        self._answer(
            monkeypatch,
            {"error": {"message": "ASR engine failed to initialize: whisper_init failed", "type": "server_error", "code": None}},
            status_code=400,
        )
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        with pytest.raises(ASRError, match="whisper_init failed") as exc_info:
            asr._create_transcription_session()
        assert not isinstance(exc_info.value, ASRServiceBusyError)

    def test_timeout_covers_a_slow_model_load(self, monkeypatch):
        seen = self._answer(monkeypatch, {"session_id": "sess-1", "state": "asr_initialized"}, status_code=200)
        asr = AutomaticSpeechRecognition(mic=_FakeMic())

        asr._create_transcription_session()

        # The server retries a failed NPU load 3 times: ~17 s on the board before it answers.
        assert seen["timeout"] == asr._CREATE_TIMEOUT_SECONDS
        assert seen["timeout"] >= 30


class TestSessionSlot:
    """Stop-then-start from a UI: the previous session is cancelled but still closing
    on the server (~4 s on the board) when the next start arrives.

    Session A always runs through ``transcribe_stream`` so that its teardown is the
    real one; the server-side close is a gate the test opens when it wants."""

    @pytest.fixture
    def asr(self, monkeypatch):
        asr = AutomaticSpeechRecognition(mic=_started_mic())
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        asr._worker_loop.set_result(loop)
        self.created: list[str] = []
        self.handled: list[bool] = []  # was the session already cancelled when its handler started?
        self.close_gate: dict[str, threading.Event] = {}  # the server closing a session, until set

        def fake_create(vad_ms=None, language=None, translate=False):
            sid = f"sess-{len(self.created) + 1}"
            self.created.append(sid)
            return sid

        async def fake_handler(session_info):
            self.handled.append(session_info.cancelled.is_set())
            await asyncio.to_thread(session_info.cancelled.wait)
            gate = self.close_gate.get(session_info.session_id)
            if gate is not None:
                await asyncio.to_thread(gate.wait)

        monkeypatch.setattr(asr, "_create_transcription_session", fake_create)
        monkeypatch.setattr(asr, "_transcription_session_handler", fake_handler)
        yield asr
        for gate in self.close_gate.values():
            gate.set()
        asr.cancel()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)

    @staticmethod
    def _start(asr) -> tuple[threading.Thread, list]:
        """Run a transcription in a thread; the list gets its events, or its exception."""
        out: list = []

        def run():
            try:
                out.append(list(asr.transcribe_stream()))
            except Exception as e:  # noqa: BLE001
                out.append(e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, out

    @staticmethod
    def _wait(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, "timed out"
            time.sleep(0.005)

    def _running(self, asr, n: int) -> None:
        """Wait until the n-th session is running, i.e. its handler has started."""
        self._wait(lambda: len(self.handled) >= n)

    def _closing_session(self, asr) -> threading.Thread:
        """Start session A, stop it, and hold its server-side close open."""
        self.close_gate["sess-1"] = threading.Event()
        thread, _ = self._start(asr)
        self._running(asr, 1)
        asr.cancel()
        return thread

    @staticmethod
    def _active_id(asr) -> str | None:
        active = asr._active_session
        return active.session_id if active else None

    def test_start_waits_for_a_cancelled_session_to_close(self, asr):
        a = self._closing_session(asr)
        b, _ = self._start(asr)

        time.sleep(0.2)
        assert self.created == ["sess-1"]  # B is waiting, not busy and not started
        assert b.is_alive()

        self.close_gate["sess-1"].set()
        self._running(asr, 2)
        assert self._active_id(asr) == "sess-2"
        a.join(1)
        asr.cancel()
        b.join(1)
        assert self.handled == [False, False]

    def test_start_during_a_running_session_is_busy(self, asr):
        a, _ = self._start(asr)
        self._running(asr, 1)
        t0 = time.monotonic()

        with pytest.raises(ASRBusyError):
            list(asr.transcribe_stream())

        assert time.monotonic() - t0 < 0.2
        assert self.created == ["sess-1"]
        asr.cancel()
        a.join(1)

    def test_gives_up_when_the_close_never_ends(self, asr):
        asr._CLOSING_WAIT_SECONDS = 0.2
        self._closing_session(asr)

        with pytest.raises(ASRBusyError):
            list(asr.transcribe_stream())
        assert self.created == ["sess-1"]

    def test_cancel_while_creating_cancels_the_new_session(self, asr, monkeypatch):
        def create_then_user_stops(vad_ms=None, language=None, translate=False):
            self.created.append("sess-1")
            asr.cancel()  # the stop arrives while the server is still creating the session
            return "sess-1"

        monkeypatch.setattr(asr, "_create_transcription_session", create_then_user_stops)

        list(asr.transcribe_stream())

        assert self.created == ["sess-1"]
        assert self.handled == [True]

    def test_stop_while_waiting_for_the_close_skips_the_new_session(self, asr):
        # stop A, start B while A closes, stop again while B waits for the slot
        a = self._closing_session(asr)
        b, b_out = self._start(asr)
        self._wait(lambda: asr._pending_starts)
        asr.cancel()

        self.close_gate["sess-1"].set()
        a.join(1)
        b.join(1)
        assert not b.is_alive()
        assert self.created == ["sess-1"]  # B never reached the server
        assert b_out == [[]]  # and yielded nothing
        assert self.handled == [False]

    def test_stop_while_creating_after_the_close_cancels_the_new_session(self, asr, monkeypatch):
        # stop A, start B while A closes, stop again while the server creates B
        def create_then_user_stops(vad_ms=None, language=None, translate=False):
            sid = f"sess-{len(self.created) + 1}"
            self.created.append(sid)
            if sid == "sess-2":
                asr.cancel()
            return sid

        monkeypatch.setattr(asr, "_create_transcription_session", create_then_user_stops)
        a = self._closing_session(asr)
        b, _ = self._start(asr)
        self._wait(lambda: asr._pending_starts)

        self.close_gate["sess-1"].set()
        a.join(1)
        b.join(1)
        assert not b.is_alive()
        assert self.created == ["sess-1", "sess-2"]
        assert self.handled == [False, True]  # B started already cancelled, so it closes at once

    def test_stop_during_the_close_does_not_cancel_the_next_session(self, asr):
        a = self._closing_session(asr)
        asr.cancel()  # e.g. "new recording" pressed while A closes, no B waiting yet
        self.close_gate["sess-1"].set()
        a.join(1)

        b, _ = self._start(asr)
        self._running(asr, 2)
        assert self._active_id(asr) == "sess-2"
        assert self.handled == [False, False]
        asr.cancel()
        b.join(1)
