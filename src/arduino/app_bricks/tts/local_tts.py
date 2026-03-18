# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from typing import Literal

import numpy as np
import requests

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker
from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils import brick, Logger

logger = Logger("TextToSpeech")


@brick
class TextToSpeech:
    """Text-to-Speech brick for offline speech synthesis using local TTS service."""

    def __init__(self):
        self.max_concurrent_syntheses = 3

        # API configuration
        self.api_port = 8085
        self.api_host = "localhost"
        infra = load_brick_compose_file(self.__class__) or {}
        for k, _ in infra["services"].items():
            self.api_host = k
            break  # Only one service is expected
        self.api_host = resolve_address(self.api_host)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"

        # TTS configuration
        self._language_to_voice = {}
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
            for model in models:
                for voice in model.get("voices", []):
                    lang = voice.get("language")
                    if lang and lang not in self._language_to_voice:
                        self._language_to_voice[lang] = {
                            "voice": voice.get("name", "default"),
                            "model": model.get("name"),
                            "sample_rate": voice.get("sample_rate", 44100),
                        }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS models: {e}.")

        # Limit concurrency
        self._session_semaphore = threading.Semaphore(self.max_concurrent_syntheses)

    def speak(self, text: str, language: Literal["en", "es", "zh"] = "en", speaker: BaseSpeaker = Speaker(sample_rate=Speaker.RATE_44K)):
        """
        Synthesize speech from text and play it through the provided speaker.

        Args:
            text (str): The text to be synthesized into speech.
            language (Literal["en", "es", "zh"]): The language of the text.
            speaker (BaseSpeaker): The speaker instance to play the synthesized audio.
                If None, a default Speaker will be used.

        Raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If the synthesis fails or maximum concurrency is reached.
        """
        audio_bytes = self.synthesize_pcm(text, language=language)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # melo-tts uses 16-bit PCM
        speaker.play_pcm(audio_array)

    def synthesize_wav(self, text: str, language: Literal["en", "es", "zh"] = "en") -> bytes:
        """
        Synthesize speech from text and return the audio in WAV format.

        Args:
            text (str): The text to be synthesized into speech.
            language (Literal["en", "es", "zh"]): The language of the text.

        Returns:
            bytes: The synthesized audio in WAV format.

        Raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If the synthesis fails or maximum concurrency is reached.
        """
        pcm_audio = self.synthesize_pcm(text, language=language)

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
