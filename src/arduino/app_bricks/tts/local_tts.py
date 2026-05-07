# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
import requests

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker
from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import brick, Logger

logger = Logger("TextToSpeech")


@dataclass(eq=False)
class _SpeechSession:
    cancelled: threading.Event


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

        self._synthesis_lock = threading.Lock()
        self._active_sessions_lock = threading.Lock()
        self._active_sessions: set[_SpeechSession] = set()
        self._playback_lock = threading.Lock()

    def start(self):
        """Start the TextToSpeech brick by initializing the speaker."""
        self._speaker.start()

    def stop(self):
        """Stop the TextToSpeech brick by stopping the speaker."""
        self.cancel()
        self._speaker.stop()

    def cancel(self):
        """Cancel active speech playback, if any, without stopping the speaker."""
        with self._active_sessions_lock:
            active_sessions = tuple(self._active_sessions)

        if not active_sessions:
            logger.debug("No active speech session to cancel")
            return

        logger.debug(f"Cancelling {len(active_sessions)} speech session(s)")
        for session in active_sessions:
            session.cancelled.set()

    def speak(self, text: str):
        """
        Synthesize speech from text and play it through the provided speaker.

        Args:
            text (str): The text to be synthesized into speech.

        Raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If the synthesis fails.
        """
        session = _SpeechSession(cancelled=threading.Event())
        with self._active_sessions_lock:
            self._active_sessions.add(session)

        try:
            audio_bytes = self._synthesize_pcm(text, language=self._selected_language, cancelled=session.cancelled)
            if audio_bytes is None:
                return
            if session.cancelled.is_set():
                logger.debug("Speech session cancelled before playback")
                return

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # melo-tts uses 16-bit PCM
            with self._playback_lock:
                if session.cancelled.is_set():
                    logger.debug("Speech session cancelled before playback")
                    return
                self._play_pcm(audio_array, session.cancelled)
        finally:
            session.cancelled.set()
            with self._active_sessions_lock:
                self._active_sessions.discard(session)

    def synthesize_wav(self, text: str) -> bytes:
        """
        Synthesize speech from text and return the audio in WAV format.

        Args:
            text (str): The text to be synthesized into speech.

        Returns:
            bytes: The synthesized audio in WAV format.

        Raises:
            ValueError: If the specified language is not supported.
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
            RuntimeError: If the synthesis fails.
        """
        audio_bytes = self._synthesize_pcm(text, language=language)
        if audio_bytes is None:
            raise RuntimeError("Synthesis was cancelled")
        return audio_bytes

    def _synthesize_pcm(self, text: str, language: Literal["en", "es", "zh"] = "en", cancelled: threading.Event | None = None) -> bytes | None:
        if language not in self._language_to_voice:
            raise ValueError(f"Unsupported language: {language}")

        with self._synthesis_lock:
            if cancelled is not None and cancelled.is_set():
                logger.debug("Speech session cancelled before synthesis")
                return None

            model_params = self._language_to_voice[language]
            payload = {
                "text": text,
                "model": model_params["model"],
                "language": language,
                "voice": model_params["voice"],
                "sample_rate": model_params["sample_rate"],
            }
            url = f"{self.api_base_url}/tts/synthesize"
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                error_msg = f"Failed to synthesize text."
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)

            if not response.content:
                raise RuntimeError("No audio data returned from synthesis API")

            return response.content  # The API returns raw PCM audio data

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
