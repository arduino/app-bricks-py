# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import re
import threading
import time
from collections.abc import Generator, Iterator
from typing import ContextManager, Literal

import numpy as np
import requests

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker
from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import brick, Logger

logger = Logger("TextToSpeech")

TTS_MAX_CHARS = 1024


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

    def __init__(self, language: str | None = None, speaker: BaseSpeaker | None = None):
        """Initialize the TextToSpeech brick.
        Args:
            language (str, optional): Preferred language for TTS. If not specified, it follow App configuration.
            speaker (BaseSpeaker, optional): Speaker instance to use for audio output. If not provided, a default Speaker will be used.
        """
        self._speaker = speaker or Speaker(sample_rate=Speaker.RATE_44K, shared=True)

        # API configuration
        self.api_port = 8085
        self.api_host = "audio-analytics-runner"  # Default hostname for the TTS service in the compose network
        self.api_host = resolve_address(self.api_host)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"

        logger.info(f"Initialized TextToSpeech with API base URL: {self.api_base_url}")

        # Load the model configured at bricks level
        brick_config = get_brick_config(self.__class__)
        app_configured_model = get_brick_configured_model(brick_config.get("id") if brick_config else None)
        if app_configured_model:
            model = app_configured_model
        else:
            model = brick_config.get("model", None)

        # TTS configuration
        self._language_to_voice = {}
        self._model_to_language = {}
        try:
            url = f"{self.api_base_url}/tts/models"
            response = requests.get(url)
            if response.status_code != 200:
                error_msg = f"Failed to fetch TTS models."
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)

            models = response.json() or []
            for model_entry in models:
                model_name = model_entry.get("name")
                for voice in model_entry.get("voices", []):
                    lang = voice.get("language")
                    if lang and lang not in self._language_to_voice:
                        self._language_to_voice[lang] = {
                            "voice": voice.get("name", "default"),
                            "model": model_name,
                            "sample_rate": voice.get("sample_rate", 44100),
                        }
                        self._model_to_language[model_name] = lang
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS models: {e}.")

        self._selected_language = None
        if language:
            if language in self._language_to_voice:
                self._selected_language = language
            else:
                logger.warning(f"Configured language '{language}' not found in available TTS models. Defaulting to en.")
                self._selected_language = "en"
        if model:
            if model in self._model_to_language:
                self._selected_language = self._model_to_language[model]
            else:
                logger.warning(f"Configured model '{model}' not found in available TTS models. Defaulting to en.")
                self._selected_language = "en"

        self._active_session_lock = threading.Lock()
        self._cancelled: threading.Event | None = None

    def start(self):
        """Start the TextToSpeech brick by initializing the speaker."""
        self._speaker.start()

    def stop(self):
        """Stop the TextToSpeech brick by stopping the speaker."""
        self.cancel()
        self._speaker.stop()

    def cancel(self):
        """Cancel active speech playback, if any, without stopping the speaker."""
        cancelled = self._cancelled
        if cancelled is None:
            logger.debug("No active speech session to cancel")
            return
        logger.debug("Cancelling active speech session")
        cancelled.set()
        self._cancel_remote_tts()

    def speak(self, text: str):
        """
        Synthesize speech from text and play it through the provided speaker.
        Long text is split into 1024-character chunks before synthesis.

        Args:
            text (str): The text to be synthesized into speech.

        Raises:
            ValueError: If the specified language is not supported.
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return

        if not self._active_session_lock.acquire(blocking=False):
            raise TTSBusyError("A speech session is already active on this instance. Create a separate TextToSpeech instance for concurrent speech.")

        cancelled = threading.Event()
        self._cancelled = cancelled
        try:
            for chunk in chunks:
                if cancelled.is_set():
                    logger.debug("Speech session cancelled before synthesis")
                    return

                pcm_stream = self._synthesize_pcm_stream(
                    chunk,
                    language=self._selected_language,
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
            ValueError: If the specified language is not supported.
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """
        pcm_audio = self.synthesize_pcm(text, language=self._selected_language)

        import io
        import wave

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16 bits
                wf.setframerate(44100)  # 44.1kHz sample rate
                wf.writeframes(pcm_audio)
            wav_data = wav_io.getvalue()

        return wav_data

    def synthesize_pcm(self, text: str, language: Literal["en", "es", "zh"] = "en") -> bytes:
        """
        Synthesize speech from text and return the audio in PCM format (mono, 16-bit, 44.1kHz).

        Args:
            text (str): The text to be synthesized into speech.
            language (Literal["en", "es", "zh"]): The language of the text.

        Returns:
            bytes: The synthesized audio in PCM format.

        Raises:
            ValueError: If the specified language is not supported.
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """
        with self.synthesize_pcm_stream(text, language=language) as stream:
            return b"".join(stream)

    def synthesize_pcm_stream(self, text: str, language: Literal["en", "es", "zh"] = "en") -> SynthesisStream:
        """
        Synthesize speech from text and stream PCM audio chunks as they arrive.

        Args:
            text (str): The text to be synthesized into speech.
            language (Literal["en", "es", "zh"]): The language of the text.

        Returns:
            SynthesisStream: An iterable/context-manager yielding PCM audio chunks. Use as a
                ``with`` block to guarantee teardown of the underlying HTTP response and
                release of the session lock.

        Raises:
            ValueError: If the specified language is not supported.
            TTSBusyError: If this instance already has an active speech session.
            RuntimeError: If the synthesis fails.
        """

        def locked_stream() -> Generator[bytes, None, None]:
            if not self._active_session_lock.acquire(blocking=False):
                raise TTSBusyError(
                    "A speech session is already active on this instance. Create a separate TextToSpeech instance for concurrent speech."
                )
            try:
                yield from self._synthesize_pcm_stream(text, language=language)
            finally:
                self._active_session_lock.release()

        return SynthesisStream(locked_stream())

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

    def _synthesize_pcm_stream(
        self,
        text: str,
        language: Literal["en", "es", "zh"] = "en",
        cancelled: threading.Event | None = None,
        keep_alive: bool = False,
    ) -> Iterator[bytes]:
        if language not in self._language_to_voice:
            raise ValueError(f"Unsupported language: {language}")

        if cancelled is not None and cancelled.is_set():
            logger.debug("Speech session cancelled before synthesis")
            return

        model_params = self._language_to_voice[language]
        payload = {
            "text": text,
            "model": model_params["model"],
            "language": language,
            "voice": model_params["voice"],
            "sample_rate": model_params["sample_rate"],
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
