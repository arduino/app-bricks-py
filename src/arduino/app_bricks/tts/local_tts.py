# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import queue
import re
import threading
import time
from collections.abc import Generator, Iterator
from typing import ContextManager

import numpy as np
import requests

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker
from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import brick, Logger

logger = Logger("TextToSpeech")

TTS_MAX_CHARS = 1024
TTS_MAX_QUEUE_SIZE = 128

_SPEECH_QUEUE_STOP = object()


class TTSError(Exception):
    """Base class for TTS errors."""


class TTSBusyError(TTSError):
    """Raised when this TTS instance already has an active speech session."""


class SynthesisStream(ContextManager["SynthesisStream"], Iterator[bytes]):
    """Iterator wrapper that guarantees proper teardown on context exit."""

    def __init__(self, generator: Generator[bytes, None, None]):
        self._generator = generator

    def __enter__(self) -> "SynthesisStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> "SynthesisStream":
        return self

    def __next__(self) -> bytes:
        return next(self._generator)

    def close(self) -> None:
        self._generator.close()


@brick
class TextToSpeech:
    """Text-to-Speech brick for offline speech synthesis using local TTS service."""

    _APP_SERVICE_NAME = "audio-analytics-runner"

    def __init__(self, speaker: BaseSpeaker | None = None, max_queue_size: int = TTS_MAX_QUEUE_SIZE):
        """Initialize the TextToSpeech brick.
        Args:
            speaker (BaseSpeaker, optional): Speaker instance to use for audio output. If not provided, a default Speaker will be used.
            max_queue_size (int): Maximum number of pending ``speak(block=False)`` requests. When the
                queue is full, further non-blocking calls raise TTSBusyError instead of piling up.
        """
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0.")

        self._speaker = speaker or Speaker(sample_rate=Speaker.RATE_44K, shared=True)

        # API configuration
        self.api_host = resolve_address(self._APP_SERVICE_NAME)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"

        logger.debug(f"Initialized TextToSpeech with API base URL: {self.api_base_url}")

        # Resolve the model: app.yaml override (per-brick `model:`) takes precedence over the brick default.
        brick_config = get_brick_config(self.__class__) or {}
        brick_id = brick_config.get("id")
        override = get_brick_configured_model(brick_id) if brick_id else None
        model_name = override or brick_config.get("model")
        if not model_name:
            raise RuntimeError("No TTS model configured for the TextToSpeech brick.")

        self._voice = self._resolve_voice(model_name)
        logger.debug(f"Using TTS model '{self._voice['model']}' (language='{self._voice['language']}').")

        self._active_session_lock = threading.Lock()
        self._cancelled: threading.Event | None = None
        self._speak_thread: threading.Thread | None = None
        self._speech_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_lock = threading.Lock()
        self._cancel_epoch = 0
        self._pending_speech = 0

    def start(self):
        """Start the TextToSpeech brick by initializing the speaker."""
        self._speaker.start()
        self._warmup()

    def stop(self):
        """Stop the TextToSpeech brick by stopping the speaker."""
        self.cancel()
        with self._worker_lock:
            speak_thread = self._speak_thread
            self._speak_thread = None
        if speak_thread is not None and speak_thread.is_alive():
            try:
                self._speech_queue.put_nowait(_SPEECH_QUEUE_STOP)
            except queue.Full:
                logger.warning("Speech queue is full, the worker cannot be notified to stop")
            speak_thread.join(timeout=1.0)
            if speak_thread.is_alive():
                logger.warning("Background speech worker did not terminate in time")
        self._speaker.stop()

    def cancel(self):
        """Cancel active speech playback and drop any queued speech, without stopping the speaker."""
        with self._worker_lock:
            self._cancel_epoch += 1
            self._drain_speech_queue()
            cancelled = self._cancelled
        if cancelled is None:
            logger.debug("No active speech session to cancel")
            return
        logger.debug("Cancelling active speech session")
        cancelled.set()
        self._cancel_remote_tts()

    def is_speaking(self) -> bool:
        """Return True if this instance has an active speech or synthesis session, or queued speech pending playback."""
        with self._worker_lock:
            pending = self._pending_speech
        return pending > 0 or self._active_session_lock.locked()

    def speak(self, text: str, block: bool = True):
        """
        Synthesize speech from text and play it through the provided speaker.
        Long text is split into 1024-character chunks before synthesis.

        Args:
            text (str): The text to be synthesized into speech.
            block (bool): If True, block until playback completes. If False, return
                immediately: the text is enqueued and played sequentially (FIFO) by a
                single background worker thread, so ``speak()`` can be called
                repeatedly (e.g. sentence by sentence from a streaming LLM) without
                waiting. In this mode synthesis or playback errors are logged instead
                of raised. Use ``cancel()`` to interrupt playback and drop queued
                text, and ``is_speaking()`` to poll for completion.

        Raises:
            TTSBusyError: If ``block`` is True and this instance already has an active speech
                session, or if ``block`` is False and the speech queue is full.
            RuntimeError: If the synthesis fails (only when ``block`` is True).
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return

        if not block:
            self._enqueue_speech(chunks)
            return

        if not self._active_session_lock.acquire(blocking=False):
            raise TTSBusyError("A speech session is already active on this instance. Create a separate TextToSpeech instance for concurrent speech.")

        cancelled = threading.Event()
        self._cancelled = cancelled
        self._run_speech_session(chunks, cancelled)

    def _enqueue_speech(self, chunks: list[str]) -> None:
        """Enqueue pre-chunked text for FIFO playback, starting the worker thread if needed."""
        with self._worker_lock:
            try:
                self._speech_queue.put_nowait((self._cancel_epoch, chunks))
            except queue.Full:
                raise TTSBusyError(
                    f"The speech queue is full ({self._speech_queue.maxsize} pending requests). "
                    "Wait for playback to catch up or call cancel() to drop queued speech."
                )
            self._pending_speech += 1
            if self._speak_thread is None or not self._speak_thread.is_alive():
                self._speak_thread = threading.Thread(target=self._speech_worker, daemon=True, name="TTS-SpeakWorker")
                self._speak_thread.start()

    def _speech_worker(self) -> None:
        """Consume queued speech requests sequentially until the stop sentinel arrives."""
        while True:
            item = self._speech_queue.get()
            if item is _SPEECH_QUEUE_STOP:
                return
            epoch, chunks = item
            try:
                self._active_session_lock.acquire()
                with self._worker_lock:
                    if epoch < self._cancel_epoch:
                        self._active_session_lock.release()
                        continue
                    cancelled = threading.Event()
                    self._cancelled = cancelled
                try:
                    self._run_speech_session(chunks, cancelled)
                except Exception as e:
                    logger.error(f"Background speech session failed: {e}")
            finally:
                with self._worker_lock:
                    self._pending_speech -= 1

    def _drain_speech_queue(self) -> None:
        """Drop all queued speech requests. Must be called with the worker lock held."""
        while True:
            try:
                item = self._speech_queue.get_nowait()
            except queue.Empty:
                return
            if item is _SPEECH_QUEUE_STOP:
                self._speech_queue.put(item)
                return
            self._pending_speech -= 1

    def _run_speech_session(self, chunks: list[str], cancelled: threading.Event) -> None:
        """Run a speech session over pre-chunked text. The session lock must already be held."""
        try:
            for chunk in chunks:
                if cancelled.is_set():
                    logger.debug("Speech session cancelled before synthesis")
                    return

                pcm_stream = self._synthesize_pcm_stream(
                    chunk,
                    cancelled=cancelled,
                    keep_alive=True,
                )
                try:
                    self._play_pcm_stream(pcm_stream, cancelled)
                finally:
                    pcm_stream.close()
        finally:
            cancelled.set()
            self._cancelled = None
            self._active_session_lock.release()

    def synthesize_wav(self, text: str) -> bytes:
        """
        Synthesize speech from text and return the audio in WAV format.

        Args:
            text (str): The text to be synthesized into speech.

        Returns:
            bytes: The synthesized audio in WAV format.

        Raises:
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """
        pcm_audio = self.synthesize_pcm(text)

        import io
        import wave

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16 bits
                wf.setframerate(self._speaker.sample_rate)
                wf.writeframes(pcm_audio)
            wav_data = wav_io.getvalue()

        return wav_data

    def synthesize_pcm(self, text: str) -> bytes:
        """
        Synthesize speech from text and return the audio in PCM format (mono, 16-bit, 44.1kHz).

        Args:
            text (str): The text to be synthesized into speech.

        Returns:
            bytes: The synthesized audio in PCM format.

        Raises:
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """
        with self.synthesize_pcm_stream(text) as stream:
            return b"".join(stream)

    def synthesize_pcm_stream(self, text: str) -> SynthesisStream:
        """
        Synthesize speech from text and stream PCM audio chunks as they arrive.

        Args:
            text (str): The text to be synthesized into speech.

        Returns:
            SynthesisStream: An iterable/context-manager yielding PCM audio chunks. Use as a
                ``with`` block to guarantee teardown of the underlying HTTP response and
                release of the session lock.

        Raises:
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """

        def locked_stream() -> Generator[bytes, None, None]:
            if not self._active_session_lock.acquire(blocking=False):
                raise TTSBusyError(
                    "A speech session is already active on this instance. Create a separate TextToSpeech instance for concurrent speech."
                )
            try:
                yield from self._synthesize_pcm_stream(text)
            finally:
                self._active_session_lock.release()

        return SynthesisStream(locked_stream())

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Catalog ids and runner-reported names differ only by separators
        (e.g. catalog `piper-tts-en` vs runner `pipertts_en`), so compare them
        stripped of `-` and `_`."""
        return name.replace("-", "").replace("_", "").lower()

    def _resolve_voice(self, model_name: str) -> dict:
        """Fetch available TTS models from the runner and return the voice config for `model_name`."""
        try:
            response = requests.get(f"{self.api_base_url}/tts/models")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch TTS models: {e}.")

        if response.status_code != 200:
            error_msg = "Failed to fetch TTS models."
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", error_msg)
            except Exception:
                pass
            raise RuntimeError(error_msg)

        wanted = self._normalize_model_name(model_name)
        for entry in response.json() or []:
            entry_name = entry.get("name")
            if not entry_name or self._normalize_model_name(entry_name) != wanted:
                continue
            voices = entry.get("voices") or []
            if voices:
                voice = voices[0]
                return {
                    # We don't capture sample_rate since the TTS service resamples as needed
                    "model": entry_name,
                    "name": voice.get("name", "default"),
                    "language": voice.get("language"),
                }

        raise RuntimeError(f"TTS model '{model_name}' is not available on the runner.")

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks accepted by the local TTS service.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        started_at = time.perf_counter()
        input_chars = len(text)

        text = text.strip()
        chunks = []

        while len(text) > TTS_MAX_CHARS:
            window = text[:TTS_MAX_CHARS]
            match = re.search(r"[.!?][^.!?]*$", window)
            if match:
                cut = match.start() + 1
            else:
                newline_cut = window.rfind("\n")
                space_cut = window.rfind(" ")
                cut = next((index for index in (newline_cut, space_cut) if index > 0), len(window))
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()

        if text:
            chunks.append(text)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"TTS chunk_text completed in {elapsed_ms:.2f} ms (input_chars={input_chars}, text_chunks={len(chunks)})")

        return chunks

    def _warmup(self) -> None:
        """Best-effort warmup: synthesize a short text so the inference container loads
        the TTS model before the first real speak()."""
        started_at = time.perf_counter()
        try:
            for _ in self._synthesize_pcm_stream("ok", keep_alive=True):
                pass
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"TTS warmup completed in {elapsed_ms:.2f} ms")

    def _synthesize_pcm_stream(
        self,
        text: str,
        cancelled: threading.Event | None = None,
        keep_alive: bool = False,
    ) -> Iterator[bytes]:
        if cancelled is not None and cancelled.is_set():
            logger.debug("Speech session cancelled before synthesis")
            return

        payload = {
            "text": text,
            "model": self._voice["model"],
            "language": self._voice["language"],
            "voice": self._voice["name"],
            "sample_rate": self._speaker.sample_rate,
            "keep_alive": keep_alive,
        }
        url = f"{self.api_base_url}/tts/synthesize"
        started_at = time.perf_counter()
        response = requests.post(url, json=payload, stream=True)
        total_audio_bytes = 0
        first_chunk_logged = False

        try:
            if response.status_code != 200:
                error_msg = f"Failed to synthesize text."
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)

            if cancelled is not None and cancelled.is_set():
                logger.debug("Speech session cancelled before reading synthesis stream")
                return

            stream_chunk_size = self._speaker.buffer_size * self._speaker.channels * self._speaker.format.itemsize
            for audio_chunk in response.iter_content(chunk_size=stream_chunk_size):
                if cancelled is not None and cancelled.is_set():
                    logger.debug("Speech session cancelled while reading synthesis stream")
                    return
                if not audio_chunk:
                    continue

                total_audio_bytes += len(audio_chunk)
                if not first_chunk_logged:
                    first_chunk_logged = True
                    first_chunk_ms = (time.perf_counter() - started_at) * 1000
                    logger.debug(
                        f"TTS PCM stream first chunk received in {first_chunk_ms:.2f} ms "
                        f"(input_chars={len(text)}, pcm_chunk_bytes={len(audio_chunk)}, keep_alive={keep_alive})"
                    )
                yield audio_chunk

            if total_audio_bytes == 0 and (cancelled is None or not cancelled.is_set()):
                raise RuntimeError("No audio data returned from synthesis API")

        finally:
            response.close()
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.debug(
                f"TTS PCM stream completed in {elapsed_ms:.2f} ms "
                f"(input_chars={len(text)}, status_code={response.status_code}, "
                f"pcm_bytes={total_audio_bytes}, keep_alive={keep_alive})"
            )

    def _cancel_remote_tts(self) -> None:
        try:
            response = requests.post(f"{self.api_base_url}/tts/cancel")
            if response.status_code >= 400:
                logger.warning(f"Failed to cancel remote TTS session: status_code={response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to cancel remote TTS session: {e}")

    def _play_pcm(self, pcm_audio: np.ndarray, cancelled: threading.Event) -> None:
        if pcm_audio is None or len(pcm_audio) == 0:
            raise ValueError("Audio data cannot be empty")

        if pcm_audio.dtype != self._speaker.format:
            raise ValueError(f"Audio data with dtype {pcm_audio.dtype} does not match expected {self._speaker.format}")

        offset = 0
        total_samples = len(pcm_audio)
        while offset < total_samples:
            if cancelled.is_set():
                logger.debug("Speech playback cancelled")
                return

            chunk_size = min(self._speaker.buffer_size * self._speaker.channels, total_samples - offset)
            chunk = pcm_audio[offset : offset + chunk_size]
            self._speaker.play(chunk)
            offset += chunk_size

    def _play_pcm_stream(self, pcm_chunks: Iterator[bytes], cancelled: threading.Event) -> None:
        pending = b""
        sample_width = np.dtype(np.int16).itemsize

        for pcm_chunk in pcm_chunks:
            if cancelled.is_set():
                logger.debug("Speech playback cancelled")
                return

            audio_bytes = pending + pcm_chunk
            aligned_size = len(audio_bytes) - (len(audio_bytes) % sample_width)
            if aligned_size:
                audio_array = np.frombuffer(audio_bytes[:aligned_size], dtype=np.int16)  # melo-tts uses 16-bit PCM
                self._play_pcm(audio_array, cancelled)
            pending = audio_bytes[aligned_size:]

        if pending and not cancelled.is_set():
            raise RuntimeError("Incomplete PCM sample returned from synthesis API")
