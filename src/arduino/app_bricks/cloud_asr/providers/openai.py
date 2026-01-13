# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json

import websocket

from arduino.app_utils import Logger

from .types import ASRProviderEvent, ASRProviderError

logger = Logger(__name__)


class OpenAITranscribe:
    """
    OpenAI ASR cloud provider implementation.
    It leverages the Realtime API to enable streaming transcription powered by a GPT-based model.
    Audio is transmitted and received over WebSockets, while voice activity detection (VAD) server-side
    is used to segment utterances.
    If custom VAD behavior is desired, the VoiceActivityDetector class can be used client-side to
    trigger commits based on local audio analysis. In that case, track the audio with the vad process_chunk method and
    register the vad commit() method to send a `{"type": "input_audio_buffer.commit"}` message to the server.
    """

    provider_name = "openai-transcribe"
    partial_mode = "append"

    REALTIME_MODEL = "gpt-realtime"
    TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
    BASE_URL = "wss://api.openai.com/v1/realtime"
    IGNORED_COMMIT_CODES = {
        "input_audio_buffer_commit_empty",
        "input_audio_buffer_commit_short",
    }
    VAD_MIN_BUFFER_MS = 120.0
    DEFAULT_LANGUAGE = "en"

    def __init__(
        self,
        api_key: str,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 16000,
    ):
        if not api_key:
            raise ValueError("API key is required for OpenAI Realtime client.")

        self._api_key = api_key
        self._language = language
        if not self._language:
            self._language = self.DEFAULT_LANGUAGE

        self._url = f"{self.BASE_URL}?model={self.REALTIME_MODEL}"
        self._headers = [
            f"Authorization: Bearer {self._api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        self._sample_rate = sample_rate
        self._ws: websocket.WebSocket

    def _connect(self) -> websocket.WebSocket:
        logger.info("Connecting to realtime ASR endpoint: %s", self._url)
        ws = websocket.WebSocket()
        ws.connect(self._url, header=self._headers, ping_interval=20, ping_timeout=20)
        self._send_session_update(ws)
        return ws

    def _send_session_update(self, ws: websocket.WebSocket) -> None:
        ws.send(
            json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_transcription": {
                        "model": self.TRANSCRIPTION_MODEL,
                        "language": self._language,
                    },
                    "instructions": "You are a transcription engine. Only return verbatim transcripts and do not chat or respond.",
                },
            })
        )

    def start(self) -> None:
        """Start the ASR session."""
        self._ws = self._connect()

    def _decode_message(self, raw: object) -> object:
        if isinstance(raw, (str, bytes, bytearray)):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _extract_error_code(self, message: object) -> str | None:
        """Try to find an error code either nested under 'error' or at top-level."""
        if not isinstance(message, dict):
            return None

        err = message.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                return code

        code = message.get("code")
        return code if isinstance(code, str) else None

    def _extract_error_payload(self, message: object) -> object:
        """Prefer nested 'error' payload if present, otherwise return message."""
        if isinstance(message, dict) and "error" in message:
            return message.get("error")
        return message

    def _format_event(self, message: dict) -> ASRProviderEvent | None:
        match message.get("type"):
            case "input_audio_buffer.speech_started":
                return ASRProviderEvent(type="speech_start", data=None)
            case "input_audio_buffer.speech_stopped":
                return ASRProviderEvent(type="speech_stop", data=None)
            case "conversation.item.input_audio_transcription.delta":
                delta_text = message.get("delta", "").strip()
                if delta_text:
                    return ASRProviderEvent(type="partial_text", data=delta_text)

            case "conversation.item.input_audio_transcription.completed":
                text = message.get("transcript", "").strip()
                if text:
                    return ASRProviderEvent(type="text", data=text)
                raise ASRProviderError("Transcription completed with no text.")

            case "error" | "invalid_request_error":
                code = self._extract_error_code(message)
                if code in self.IGNORED_COMMIT_CODES:
                    logger.debug("Ignoring empty commit warning from server.")
                    return None
                payload = self._extract_error_payload(message)
                raise ASRProviderError(f"OpenAI error: {payload}")

        return None

    def recv(self) -> ASRProviderEvent | None:
        try:
            raw = self._ws.recv()
        except Exception as exc:
            raise ASRProviderError(f"WebSocket receive error: {exc}") from exc

        message = self._decode_message(raw)
        if not isinstance(message, dict):
            return None

        try:
            return self._format_event(message)
        except Exception as exc:  # pragma: no cover
            raise ASRProviderError(f"Error processing message: {exc}") from exc

    def send_audio(self, pcm_chunk: bytes) -> None:
        if not pcm_chunk:
            return

        audio_payload = base64.b64encode(pcm_chunk).decode("ascii")
        self._ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_payload}))

    def stop(self) -> None:
        try:
            self._ws.close()
        except Exception as exc:
            raise ASRProviderError(f"WebSocket close error: {exc}") from exc


__all__ = ["OpenAITranscribe"]
