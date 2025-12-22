# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import queue
import threading
from typing import Iterator, Callable, Optional

import numpy as np

from arduino.app_peripherals.microphone import Microphone
from arduino.app_utils import Logger, brick

from .providers import ASRProvider, CloudProvider, DEFAULT_PROVIDER, provider_factory

logger = Logger(__name__)

DEFAULT_LANGUAGE = "en"


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
    ):
        if mic:
            logger.info(f"[{self.__class__.__name__}] Using provided microphone: {mic}")
            self._mic = mic
        else:
            self._mic = Microphone()

        self._language = language
        self._mic_lock = threading.Lock()
        self._provider: ASRProvider = provider_factory(
            api_key=api_key,
            name=provider,
            language=self._language,
            sample_rate=self._mic.sample_rate,
        )

        self.detect_handlers: list[Callable[[dict], None]] = []
        self.detect_handlers_lock = threading.Lock()
        self.partial_handlers: list[Callable[[dict], None]] = []
        self.partial_handlers_lock = threading.Lock()

    def start(self):
        with self._mic_lock:
            if not self._mic.is_recording.is_set():
                self._mic.start()
                logger.info(f"[{self.__class__.__name__}] Microphone started.")

    def stop(self):
        with self._mic_lock:
            if self._mic.is_recording.is_set():
                self._mic.stop()
                logger.info(f"[{self.__class__.__name__}] Microphone stopped.")

    def on_detect(self, handler):
        """Register a callback to be invoked when speech is detected."""
        with self.detect_handlers_lock:
            self.detect_handlers.append(handler)

    @brick.loop
    def _detect_loop(self):
        """Continuously listen for speech and invoke handlers when final text is detected."""
        for resp in self.transcribe():
            match resp["event"]:
                case "error":
                    logger.error(f"ASR error: {resp['data']}")
                case "text":
                    with self.detect_handlers_lock:
                        for handler in self.detect_handlers:
                            try:
                                handler(resp["data"])
                            except Exception as exc:
                                logger.error(f"Error in speech detected handler: {exc}")

    def on_update(self, handler):
        """Register a callback to be invoked for each partial speech update."""
        with self.partial_handlers_lock:
            self.partial_handlers.append(handler)

    @brick.loop
    def _update_loop(self):
        """Continuously listen for partial speech and invoke handlers."""
        for resp in self.transcribe():
            with self.partial_handlers_lock:
                for handler in self.partial_handlers:
                    try:
                        handler(resp)
                    except Exception as exc:
                        logger.error(f"Error in partial speech handler: {exc}")

    def transcribe(self) -> Iterator[dict]:
        """Perform speech-to-text recognition.

        Returns:
            Iterator[dict]: Generator yielding
            {"event": ("speech_start|partial_text|text|error|speech_stop"), "data": "<payload>"}
            messages.
        """

        provider = self._provider
        messages: queue.Queue[dict] = queue.Queue()
        stop_event = threading.Event()

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
                messages.put({"event": "error", "data": str(exc)})
            finally:
                stop_event.set()

        partial_buffer = ""

        def _recv():
            nonlocal partial_buffer
            try:
                while not stop_event.is_set():
                    result = provider.recv()
                    if result is None:
                        continue

                    data = result.data
                    if result.event == "partial_text":
                        if self._provider.partial_mode == "replace":
                            partial_buffer = str(data)
                        else:
                            partial_buffer += str(data)
                    elif result.event == "text":
                        data = data or partial_buffer
                        partial_buffer = ""
                    messages.put({"event": result.event, "data": data})

            except Exception as exc:
                logger.error("Error receiving transcription events: %s", exc)
                messages.put({"event": "error", "data": str(exc)})
                stop_event.set()

        send_thread = threading.Thread(target=_send, daemon=True)
        recv_thread = threading.Thread(target=_recv, daemon=True)
        send_thread.start()
        recv_thread.start()

        try:
            while recv_thread.is_alive() or send_thread.is_alive() or not messages.empty():
                try:
                    msg = messages.get(timeout=0.1)
                    yield msg
                except queue.Empty:
                    continue
        finally:
            stop_event.set()
            send_thread.join(timeout=1)
            recv_thread.join(timeout=1)
            provider.stop()
