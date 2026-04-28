# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Generator, Union, Iterator, Generator, cast

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
        if self._owns_mic:
            self._mic.stop()

    def transcribe(self, duration: float = 60.0) -> str:
        """
        Returns the first utterance transcribed from speech to text.

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
        """
        Perform continuous speech-to-text recognition.

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

    def _transcribe_stream(self, duration: float = 60.0) -> Generator[ASREvent, None, None]:
        """
        Perform continuous speech-to-text recognition with detailed events.

        Args:
            duration (float): Max seconds for the transcription session.

        Returns:
            Iterator[dict]: Generator yielding
            {"event": ("speech_start|partial_text|text|error|speech_stop"), "data": "<payload>"}
            messages.
        """
        messages: queue.Queue[Union[ASRProviderEvent, BaseException]] = queue.Queue()
        stop_event = threading.Event()
        overall_deadline = time.monotonic() + duration
        silence_deadline = time.monotonic() + self.silence_timeout

        def _send():
            try:
                for chunk in self._mic.stream():
                    if stop_event.is_set() or self._shutdown.is_set():
                        break
                    if chunk is None:
                        continue
                    pcm_chunk_np = np.asarray(chunk, dtype=np.int16)
                    self._provider.send_audio(pcm_chunk_np.tobytes())
            except Exception as exc:
                if stop_event.is_set() or self._shutdown.is_set():
                    return
                messages.put(ASRProviderError(f"Error while streaming microphone audio: {exc}"))
                stop_event.set()

        partial_buffer = ""

        def _recv():
            nonlocal partial_buffer
            try:
                while not stop_event.is_set() and not self._shutdown.is_set():
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
                if stop_event.is_set() or self._shutdown.is_set():
                    return
                messages.put(exc)
                stop_event.set()

        send_thread = threading.Thread(target=_send, daemon=True)
        recv_thread = threading.Thread(target=_recv, daemon=True)
        self._provider.start()
        send_thread.start()
        recv_thread.start()

        try:
            while (
                (recv_thread.is_alive() or send_thread.is_alive() or not messages.empty())
                and not self._shutdown.is_set()
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
            logger.debug("Releasing ASR resources...")
            stop_event.set()
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
