# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import queue
import threading

from arduino.app_utils import Logger
from google.api_core.client_options import ClientOptions
from google.cloud.speech import SpeechClient, StreamingRecognitionConfig, RecognitionConfig, StreamingRecognizeRequest, StreamingRecognizeResponse

from .types import ASRProviderEvent, ASRProviderError

logger = Logger(__name__)


class GoogleSpeech:
    """
    Google ASR cloud provider implementation.

    It uses google cloud speech package to connect to Google Speech-to-Text API
    for streaming transcription.
    For English locales it uses the default streaming model. For non-English
    locales the standard model segments poorly, so `latest_short` is used to
    get faster segmentation even though it emits a single utterance; when that
    happens the stream is restarted transparently while preserving queued
    audio so callers keep a continuous feed of events.
    """

    provider_name = "google-speech"
    partial_mode = "replace"

    GOOGLE_LANG_MAP = {
        "en": "en-US",
        "it": "it-IT",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "pt": "pt-PT",
        "pt-br": "pt-BR",
    }
    DEFAULT_LANGUAGE = "en"

    def __init__(
        self,
        api_key: str,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 16000,
    ):
        if not api_key:
            raise RuntimeError("Google Speech requires an API key; set API_KEY for this cloud provider.")
        self._api_key = api_key

        self._language = self._resolve_google_language(language)
        if not self._language:
            self._language = self.DEFAULT_LANGUAGE

        self._sample_rate = sample_rate
        self._use_short_model = not (self._language.lower().startswith("en"))

        self._stop_event = threading.Event()
        self._audio_q: queue.Queue[bytes | None] = queue.Queue()
        self._resp_q: queue.Queue[ASRProviderEvent | None] = queue.Queue()

        self._client = SpeechClient(client_options=ClientOptions(api_key=self._api_key))
        self._config = self._build_config()
        self._thread: threading.Thread

    def _resolve_google_language(self, language: str) -> str:
        if not language:
            return self.GOOGLE_LANG_MAP[self.DEFAULT_LANGUAGE]
        key = language.strip().lower()
        return self.GOOGLE_LANG_MAP.get(key, language)

    def _build_config(self) -> StreamingRecognitionConfig:
        config_kwargs = dict(
            encoding=RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self._sample_rate,
            language_code=self._language,
            enable_automatic_punctuation=True,
        )
        if self._use_short_model:
            config_kwargs["model"] = "latest_short"

        return StreamingRecognitionConfig(
            config=RecognitionConfig(**config_kwargs),
            interim_results=True,
            enable_voice_activity_events=True,
            single_utterance=self._use_short_model,
        )

    def start(self) -> None:
        """Start the ASR session."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._asr_worker, daemon=True)
        self._thread.start()

    def _request_loop(self, session_end: threading.Event):
        while not self._stop_event.is_set() and not session_end.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # When stop() is called, a None value is pushed into the audio queue
            # to unblock audio_q.get() and force this generator to exit immediately.
            # This allows streaming_recognize to terminate cleanly even if no audio
            # chunks are currently being produced.
            if chunk is None:
                return

            yield StreamingRecognizeRequest(audio_content=chunk)

    def _asr_worker(self):
        """
        ASR worker thread that streams audio to Google Speech and emits transcription events.

        The worker runs streaming_recognize in a loop to support both standard and short models.
        For short models, Google ends the stream at utterance boundaries; when
        END_OF_SINGLE_UTTERANCE is received, `session_end` stops audio consumption in
        `_request_loop`. The outer loop then restarts streaming_recognize to continue
        processing subsequent audio.
        For standard models, this event never occurs, so the stream remains open until
        explicitly stopped or an error occurs.
        """
        try:
            while not self._stop_event.is_set():
                session_end = threading.Event()
                try:
                    for response in self._client.streaming_recognize(
                        config=self._config,
                        requests=self._request_loop(session_end),
                    ):
                        ev = self._format_event(response)
                        if ev is None:
                            continue
                        if ev.type == "utterance_end":
                            session_end.set()
                            continue

                        self._resp_q.put(ev)

                except Exception as exc:
                    if not self._stop_event.is_set():
                        raise ASRProviderError(f"Google Speech ASR error: {exc}") from exc
                    break

        finally:
            self._resp_q.put(None)

    def _format_event(self, message: object) -> ASRProviderEvent | None:
        match getattr(message, "speech_event_type", None):
            case StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                return ASRProviderEvent(type="speech_start")
            case StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                return ASRProviderEvent(type="speech_stop")
            case StreamingRecognizeResponse.SpeechEventType.END_OF_SINGLE_UTTERANCE:
                return ASRProviderEvent(type="utterance_end")
        results = getattr(message, "results", None)
        if not results:
            return None

        final_text: str | None = None
        best_partial_text: str | None = None
        best_stability: float = -1.0

        for result in results:
            alternatives = getattr(result, "alternatives", [])
            if not alternatives:
                continue

            # First alternative is the most probable one
            transcript = (alternatives[0].transcript or "").strip()
            if not transcript:
                continue

            if getattr(result, "is_final", False):
                final_text = transcript
            else:
                stability = float(getattr(result, "stability", 0.0))
                if stability > best_stability:
                    best_stability = stability
                    best_partial_text = transcript

        if final_text:
            return ASRProviderEvent(type="text", data=final_text)
        if best_partial_text:
            return ASRProviderEvent(type="partial_text", data=best_partial_text)
        return None

    def send_audio(self, pcm_chunk: bytes) -> None:
        self._audio_q.put(pcm_chunk)

    def recv(self) -> ASRProviderEvent | None:
        try:
            return self._resp_q.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        self._audio_q.put(None)
        try:
            self._thread.join(timeout=1)
        except Exception as exc:
            raise ASRProviderError(f"ASR worker thread join error: {exc}") from exc


__all__ = ["GoogleSpeech"]
