# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional, Union, Iterator, Generator, cast

import numpy as np

from arduino.app_peripherals.microphone import Microphone
from arduino.app_utils import Logger, brick

from .providers import ASRProvider, CloudProvider, DEFAULT_PROVIDER, provider_factory
from .providers.types import ASRProviderEvent, ASRProviderError
from .types import ASREvent, ASREventType, ASREventTypeValues

logger = Logger(__name__)

DEFAULT_LANGUAGE = "en"


class TranscriptionTimeoutError(TimeoutError):
    pass


class TranscriptionStreamError(RuntimeError):
    pass


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
        mic: Optional[Microphone] = None,
        language: str = os.getenv("LANGUAGE", ""),
        silence_timeout: float = 10.0,
    ):
        if mic:
            logger.info(f"[{self.__class__.__name__}] Using provided microphone: {mic}")
            self._mic = mic
        else:
            self._mic = Microphone()

        self._language = language
        self.silence_timeout = silence_timeout
        self._mic_lock = threading.Lock()
        self._provider: ASRProvider = provider_factory(
            api_key=api_key,
            name=provider,
            language=self._language,
            sample_rate=self._mic.sample_rate,
        )

    def _transcribe_stream(self, duration: float = 60.0) -> Generator[ASREvent, None, None]:
        """Perform continuous speech-to-text recognition with detailed events.

        Args:
            duration (float): Max seconds for the transcription session.

        Returns:
            Iterator[dict]: Generator yielding
            {"event": ("speech_start|partial_text|text|error|speech_stop"), "data": "<payload>"}
            messages.
        """

        provider = self._provider
        messages: queue.Queue[Union[ASRProviderEvent, BaseException]] = queue.Queue()
        stop_event = threading.Event()
        send_done = threading.Event()
        overall_deadline = time.monotonic() + duration
        silence_deadline = time.monotonic() + self.silence_timeout

        with self._mic_lock:
            if self._mic.is_recording.is_set():
                raise RuntimeError("Microphone is busy.")
            self._mic.start()
            logger.info(f"[{self.__class__.__name__}] Microphone started.")

        def _send():
            try:
                for chunk in self._mic.stream():
                    if stop_event.is_set():
                        break
                    if chunk is None:
                        continue
                    pcm_chunk_np = np.asarray(chunk, dtype=np.int16)
                    provider.send_audio(pcm_chunk_np.tobytes())
            except KeyboardInterrupt:
                logger.info("Recognition interrupted by user. Exiting...")
            except Exception as exc:
                logger.error("Error while streaming microphone audio: %s", exc)
                raise ASRProviderError(f"Error while streaming microphone audio: {exc}") from exc
            finally:
                send_done.set()

        partial_buffer = ""

        def _recv():
            nonlocal partial_buffer
            try:
                while not stop_event.is_set():
                    result = provider.recv()
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
                messages.put(exc)
                stop_event.set()

        send_thread = threading.Thread(target=_send, daemon=True)
        recv_thread = threading.Thread(target=_recv, daemon=True)
        provider.start()
        send_thread.start()
        recv_thread.start()

        try:
            while (
                (recv_thread.is_alive() or send_thread.is_alive() or not messages.empty())
                and time.monotonic() < overall_deadline
                and time.monotonic() < silence_deadline
            ):
                try:
                    msg = messages.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(msg, BaseException):
                    raise msg

                if msg.type in ("partial_text", "text"):
                    silence_deadline = time.monotonic() + self.silence_timeout

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

            if time.monotonic() >= overall_deadline:
                raise TranscriptionTimeoutError(f"Maximum ASR time of {duration}s exceeded")
            if time.monotonic() >= silence_deadline:
                raise TranscriptionTimeoutError(f"No speech detected for {self.silence_timeout}s, timing out.")

        finally:
            logger.info("Releasing ASR resources...")
            stop_event.set()
            with self._mic_lock:
                if self._mic.is_recording.is_set():
                    self._mic.stop()
                    logger.info(f"[{self.__class__.__name__}] Microphone stopped.")
            send_thread.join(timeout=1)
            recv_thread.join(timeout=1)
            provider.stop()

    def _to_api(self, event: ASRProviderEvent) -> ASREvent | None:
        if event.type in ASREventTypeValues:
            return ASREvent(
                type=cast(ASREventType, event.type),
                data=event.data,
            )
        return None

    def transcribe(self, duration: float = 60.0) -> str:
        """Returns the first utterance transcribed from speech to text.

        Args:
            duration (float): Max seconds for the transcription session.
        Returns:
            str: The transcribed text.
        """

        gen = self._transcribe_stream(duration=duration)

        try:
            for resp in gen:
                if resp.type == "text":
                    return resp.data or ""
            raise TranscriptionStreamError("No transcription received.")
        finally:
            gen.close()

    @contextmanager
    def transcribe_stream(self, duration: float = 60.0) -> Iterator[Iterator[ASREvent]]:
        """Perform continuous speech-to-text recognition.

        Args:
            duration (float): Max seconds for the transcription session.

        Returns:
            Iterator[ASREvent]: Generator yielding transcription events.
        """

        gen = self._transcribe_stream(duration=duration)

        try:
            yield gen
        finally:
            gen.close()
