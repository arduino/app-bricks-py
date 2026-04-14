# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from typing import Literal

import numpy as np
import requests

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker
from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import brick, Logger

logger = Logger("TextToSpeech")


@brick
class TextToSpeech:
    """Text-to-Speech brick for offline speech synthesis using local TTS service."""

    def __init__(self, language: str | None = None, speaker: BaseSpeaker | None = None):
        """Initialize the TextToSpeech brick.
        Args:
            language (str, optional): Preferred language for TTS. If not specified, it follow App configuration.
            speaker (BaseSpeaker, optional): Speaker instance to use for audio output. If not provided, a default Speaker will be used.
        """
        self.max_concurrent_syntheses = 3
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

        # Limit concurrency
        self._session_semaphore = threading.Semaphore(self.max_concurrent_syntheses)

    def start(self):
        """Start the TextToSpeech brick by initializing the speaker."""
        self._speaker.start()

    def stop(self):
        """Stop the TextToSpeech brick by stopping the speaker."""
        self._speaker.stop()

    def speak(self, text: str):
        """
        Synthesize speech from text and play it through the provided speaker.

        Args:
            text (str): The text to be synthesized into speech.

        Raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If the synthesis fails or maximum concurrency is reached.
        """
        audio_bytes = self.synthesize_pcm(text, language=self._selected_language)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # melo-tts uses 16-bit PCM
        self._speaker.play_pcm(audio_array)

    def synthesize_wav(self, text: str) -> bytes:
        """
        Synthesize speech from text and return the audio in WAV format.

        Args:
            text (str): The text to be synthesized into speech.

        Returns:
            bytes: The synthesized audio in WAV format.

        Raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If the synthesis fails or maximum concurrency is reached.
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
            RuntimeError: If the synthesis fails or maximum concurrency is reached.
        """
        if language not in self._language_to_voice:
            raise ValueError(f"Unsupported language: {language}")

        if not self._session_semaphore.acquire(blocking=False):
            raise RuntimeError(f"Maximum concurrent syntheses ({self.max_concurrent_syntheses}) reached. Wait for an existing synthesis to complete.")

        try:
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

            audio_bytes = response.content  # The API returns raw PCM audio data
            return audio_bytes

        finally:
            self._session_semaphore.release()
