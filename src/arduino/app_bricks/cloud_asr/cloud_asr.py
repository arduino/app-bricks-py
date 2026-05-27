# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import math
import os
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterator, Union, cast

import numpy as np

from arduino.app_peripherals.microphone import Microphone
from arduino.app_peripherals.microphone.base_microphone import BaseMicrophone
from arduino.app_utils import Logger, brick

from .providers import ASRProvider, CloudProvider, DEFAULT_PROVIDER, provider_factory
from .providers.types import ASRProviderEvent, ASRProviderError
from .types import ASREvent, ASREventType, ASREventTypeValues

logger = Logger("CloudASR")

DEFAULT_LANGUAGE = "en"


class TranscriptionTimeoutError(TimeoutError):
    pass


class TranscriptionStreamError(RuntimeError):
    pass


@dataclass
class SessionInfo:
    cancelled: threading.Event
    duration: float
    overall_deadline: float
    silence_deadline: float


def _normalize_duration(value: float) -> float:
    return math.inf if value <= 0 else value


@brick
class CloudASR:
    """
    Cloud-based speech-to-text with pluggable cloud providers.
    It captures audio from a microphone and streams it to the selected cloud ASR provider for transcription.
    The recognized text is yielded as events in real-time.
    """

    def __init__(
        self,
        api_key: str = os.getenv("API_KEY", ""),
        provider: CloudProvider = DEFAULT_PROVIDER,
        mic: BaseMicrophone | None = None,
        language: str = os.getenv("LANGUAGE", ""),
        silence_timeout: float = 10.0,
    ):
        if mic is not None:
            logger.debug(f"Using provided microphone: {mic.name}")
            self._mic = mic
            self._owns_mic = False
        else:
            logger.info("No microphone provided, using default Microphone.")
            self._mic = Microphone()
            self._owns_mic = True

        self._language = language
        self.silence_timeout = silence_timeout
        self._provider: ASRProvider = provider_factory(
            api_key=api_key,
            language=self._language,
            sample_rate=self._mic.sample_rate,
            name=provider,
        )
        self._shutdown = threading.Event()

        self._active_session_lock = threading.Lock()
        self._active_session: SessionInfo | None = None

    def start(self):
        """Start the ASR service by initializing the microphone."""
        self._shutdown.clear()
        # Not guarded for retrocompatibility, but generally if the mic is externally
        # managed it should also be externally started
        self._mic.start()

    def stop(self):
        """
        Stop the ASR service: signal in-flight transcriptions and release
        the mic if owned.
        """
        self._shutdown.set()
        self.cancel()
        if self._owns_mic:
            self._mic.stop()

    def cancel(self) -> None:
        """Cancel the active transcription session, if any."""
        if self._active_session is None:
            return
        self._active_session.cancelled.set()

    def is_transcribing(self) -> bool:
        """Return True if a transcription session is currently active on this instance."""
        return self._active_session is not None

    def transcribe(self, duration: float = 60.0) -> str:
        """
        Returns the first utterance transcribed from speech to text.

        Args:
            duration (float): Max seconds for the transcription session.
                ``0`` means unbounded.

        Returns:
            str: The transcribed text.
        """
        with self._session_scope(duration=_normalize_duration(duration)) as session:
            for resp in self._transcribe_stream(session):
                if resp.type == "text":
                    return resp.data or ""
            raise TranscriptionStreamError("No transcription received.")

    @contextmanager
    def transcribe_stream(self, duration: float = 60.0) -> Iterator[Iterator[ASREvent]]:
        """
        Perform continuous speech-to-text recognition.

        Args:
            duration (float): Max seconds for the transcription session.
                ``0`` means unbounded.

        Returns:
            Iterator[ASREvent]: Generator yielding transcription events.
        """
        with self._session_scope(duration=_normalize_duration(duration)) as session:
            gen = self._transcribe_stream(session)
            try:
                yield gen
            finally:
                gen.close()

    def transcribe_sentence(self, timeout: float = 60.0) -> str:
        """
        Transcribe a single sentence and return its text.

        Stops at the first sentence boundary produced by the provider, or when
        ``timeout`` elapses. VAD is managed by the cloud provider.

        Args:
            timeout (float): Max seconds to wait for the sentence.
                ``0`` means no timeout.

        Returns:
            str: The transcribed sentence.
        """
        with self.transcribe_sentence_stream(timeout=timeout) as stream:
            for event in stream:
                if event.type == "text":
                    return event.data or ""
        raise TranscriptionStreamError("No transcription received.")

    @contextmanager
    def transcribe_sentence_stream(self, timeout: float = 60.0) -> Iterator[Iterator[ASREvent]]:
        """
        Yield transcription events for a single sentence.

        The stream ends after the first ``text`` event or when ``timeout``
        elapses. VAD is managed by the cloud provider.

        Args:
            timeout (float): Max seconds to wait for the sentence.
                ``0`` means no timeout.

        Yields:
            ASREvent: Transcription events.
        """
        with self._session_scope(duration=_normalize_duration(timeout)) as session:

            def sentence_gen() -> Generator[ASREvent, None, None]:
                inner = self._transcribe_stream(session)
                try:
                    for event in inner:
                        yield event
                        if event.type == "text":
                            return
                finally:
                    inner.close()

            gen = sentence_gen()
            try:
                yield gen
            finally:
                gen.close()

    @contextmanager
    def transcribe_until_cancelled(self) -> Iterator[Iterator[str]]:
        """
        Yield one sentence per ``text`` event until ``cancel()`` is called
        or the silence timeout fires. VAD is managed by the cloud provider.

        Yields:
            str: A complete sentence as recognized by the provider.
        """
        with self._session_scope(duration=math.inf) as session:

            def sentence_gen() -> Generator[str, None, None]:
                inner = self._transcribe_stream(session)
                try:
                    for event in inner:
                        data = event.data
                        if event.type == "text" and data and data.strip():
                            yield data
                finally:
                    inner.close()

            gen = sentence_gen()
            try:
                yield gen
            finally:
                gen.close()

    @contextmanager
    def _session_scope(self, duration: float) -> Iterator[SessionInfo]:
        if not self._active_session_lock.acquire(blocking=False):
            raise TranscriptionStreamError("transcription session already active")
        now = time.monotonic()
        session = SessionInfo(
            cancelled=threading.Event(),
            duration=duration,
            overall_deadline=now + duration,
            silence_deadline=now + self.silence_timeout,
        )
        self._active_session = session
        try:
            yield session
        finally:
            self._active_session = None
            self._active_session_lock.release()

    def _transcribe_stream(self, session: SessionInfo) -> Generator[ASREvent, None, None]:
        """
        Perform continuous speech-to-text recognition with detailed events.

        Returns:
            Iterator[dict]: Generator yielding
            {"event": ("speech_start|partial_text|text|error|speech_stop"), "data": "<payload>"}
            messages.
        """
        messages: queue.Queue[Union[ASRProviderEvent, BaseException]] = queue.Queue()

        def _send():
            try:
                for chunk in self._mic.stream():
                    if session.cancelled.is_set() or self._shutdown.is_set():
                        break
                    if chunk is None:
                        continue
                    pcm_chunk_np = np.asarray(chunk, dtype=np.int16)
                    self._provider.send_audio(pcm_chunk_np.tobytes())
            except Exception as exc:
                if session.cancelled.is_set() or self._shutdown.is_set():
                    return
                messages.put(ASRProviderError(f"Error while streaming microphone audio: {exc}"))
                session.cancelled.set()

        partial_buffer = ""

        def _recv():
            nonlocal partial_buffer
            try:
                while not session.cancelled.is_set() and not self._shutdown.is_set():
                    result = self._provider.recv()
                    if result is None:
                        time.sleep(0.005)  # Avoid busy waiting
                        continue

                    data = result.data
                    if result.type == "partial_text":
                        if self._provider.partial_mode == "replace":
                            partial_buffer = str(data)
                        else:
                            partial_buffer += str(data)
                    elif result.type == "text":
                        final = (result.data or "") or partial_buffer
                        partial_buffer = ""
                        result = ASRProviderEvent(type="text", data=final)
                    messages.put(result)

            except Exception as exc:
                if session.cancelled.is_set() or self._shutdown.is_set():
                    return
                messages.put(exc)
                session.cancelled.set()

        send_thread = threading.Thread(target=_send, daemon=True)
        recv_thread = threading.Thread(target=_recv, daemon=True)
        self._provider.start()
        send_thread.start()
        recv_thread.start()

        try:
            while (
                (recv_thread.is_alive() or send_thread.is_alive() or not messages.empty())
                and not self._shutdown.is_set()
                and not session.cancelled.is_set()
                and time.monotonic() < session.overall_deadline
                and time.monotonic() < session.silence_deadline
            ):
                try:
                    msg = messages.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(msg, BaseException):
                    raise msg

                if msg.type in ("partial_text", "text"):
                    session.silence_deadline = time.monotonic() + self.silence_timeout

                api_event = self._to_api(msg)
                if api_event is not None:
                    yield api_event

            # Drain any remaining messages
            while True:
                try:
                    msg = messages.get_nowait()
                    if isinstance(msg, BaseException):
                        raise msg
                except queue.Empty:
                    break

            if time.monotonic() >= session.overall_deadline:
                raise TranscriptionTimeoutError(f"Maximum ASR time of {session.duration}s exceeded")
            if time.monotonic() >= session.silence_deadline:
                raise TranscriptionTimeoutError(f"No speech detected for {self.silence_timeout}s, timing out.")

        finally:
            logger.debug("Releasing ASR resources...")
            session.cancelled.set()
            self._provider.stop()
            send_thread.join(timeout=1)
            recv_thread.join(timeout=1)

    def _to_api(self, event: ASRProviderEvent) -> ASREvent | None:
        if event.type in ASREventTypeValues:
            return ASREvent(
                type=cast(ASREventType, event.type),
                data=event.data,
            )
        return None
