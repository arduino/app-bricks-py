# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json

import numpy as np
import pytest

from arduino.app_bricks.cloud_asr.providers import openai as openai_provider
from arduino.app_bricks.cloud_asr.providers.openai import OpenAITranscribe, _resample_pcm16
from arduino.app_bricks.cloud_asr.providers.types import ASRProviderError


class FakeWebSocket:
    """websocket.WebSocket stand-in that records traffic and replays messages."""

    def __init__(self):
        self.connect_url: str | None = None
        self.connect_headers: list[str] | None = None
        self.sent: list[str] = []
        self.recv_messages: list[object] = []
        self.closed = False

    def connect(self, url, header=None, **kwargs):
        self.connect_url = url
        self.connect_headers = header

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self):
        return self.recv_messages.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sockets(monkeypatch: pytest.MonkeyPatch) -> list[FakeWebSocket]:
    created: list[FakeWebSocket] = []

    def _factory(*args, **kwargs) -> FakeWebSocket:
        ws = FakeWebSocket()
        created.append(ws)
        return ws

    monkeypatch.setattr(openai_provider.websocket, "WebSocket", _factory)
    return created


def make_provider(language: str = "en", sample_rate: int = 24000) -> OpenAITranscribe:
    return OpenAITranscribe(api_key="test-key", language=language, sample_rate=sample_rate)


def sent_audio_samples(ws: FakeWebSocket) -> np.ndarray:
    """Decode every input_audio_buffer.append frame sent so far into one int16 array."""
    chunks = []
    for payload in ws.sent:
        message = json.loads(payload)
        if message["type"] == "input_audio_buffer.append":
            chunks.append(np.frombuffer(base64.b64decode(message["audio"]), dtype="<i2"))
    if not chunks:
        return np.empty(0, dtype="<i2")
    return np.concatenate(chunks)


def test_missing_api_key_raises_value_error():
    with pytest.raises(ValueError):
        OpenAITranscribe(api_key="")


def test_connect_url_has_transcription_intent_and_no_model(fake_sockets):
    provider = make_provider()
    provider.start()
    ws = fake_sockets[-1]
    assert ws.connect_url == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert "model=" not in ws.connect_url


def test_connect_sends_no_beta_header(fake_sockets):
    provider = make_provider()
    provider.start()
    ws = fake_sockets[-1]
    assert ws.connect_headers == ["Authorization: Bearer test-key"]
    assert not any("OpenAI-Beta" in header for header in ws.connect_headers)


def test_session_update_is_ga_shape(fake_sockets):
    provider = make_provider(language="it")
    provider.start()
    ws = fake_sockets[-1]
    message = json.loads(ws.sent[0])
    assert message["type"] == "session.update"
    session = message["session"]
    assert session["type"] == "transcription"
    audio_input = session["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"] == {"model": "gpt-4o-mini-transcribe", "language": "it"}
    assert audio_input["turn_detection"] == {"type": "server_vad"}
    # Beta-only keys must not leak into the GA payload.
    for beta_key in ("modalities", "instructions", "input_audio_format", "input_audio_transcription"):
        assert beta_key not in session


def test_language_defaults_to_en_when_empty(fake_sockets):
    provider = make_provider(language="")
    provider.start()
    session = json.loads(fake_sockets[-1].sent[0])["session"]
    assert session["audio"]["input"]["transcription"]["language"] == "en"


def recv_event(fake_sockets, message: object):
    provider = make_provider()
    provider.start()
    ws = fake_sockets[-1]
    ws.recv_messages.append(json.dumps(message) if isinstance(message, dict) else message)
    return provider.recv()


def test_recv_delta_yields_partial_text(fake_sockets):
    event = recv_event(fake_sockets, {"type": "conversation.item.input_audio_transcription.delta", "delta": " Hello"})
    assert event is not None
    assert event.type == "partial_text"
    assert event.data == "Hello"


def test_recv_empty_delta_yields_none(fake_sockets):
    assert recv_event(fake_sockets, {"type": "conversation.item.input_audio_transcription.delta", "delta": "  "}) is None


def test_recv_completed_yields_text(fake_sockets):
    event = recv_event(fake_sockets, {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Hello world"})
    assert event is not None
    assert event.type == "text"
    assert event.data == "Hello world"


def test_recv_completed_without_text_is_ignored(fake_sockets):
    # Server VAD can close a turn on noise and return an empty transcript: ignore it,
    # don't abort the stream.
    assert recv_event(fake_sockets, {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}) is None


def test_recv_transcription_failed_raises(fake_sockets):
    with pytest.raises(ASRProviderError, match="transcription failed"):
        recv_event(
            fake_sockets,
            {"type": "conversation.item.input_audio_transcription.failed", "error": {"code": "audio_unintelligible"}},
        )


def test_recv_speech_events(fake_sockets):
    started = recv_event(fake_sockets, {"type": "input_audio_buffer.speech_started"})
    assert started is not None
    assert started.type == "speech_start"
    stopped = recv_event(fake_sockets, {"type": "input_audio_buffer.speech_stopped"})
    assert stopped is not None
    assert stopped.type == "speech_stop"


def test_recv_ignores_empty_commit_error(fake_sockets):
    message = {"type": "error", "error": {"code": "input_audio_buffer_commit_empty", "message": "buffer too small"}}
    assert recv_event(fake_sockets, message) is None


def test_recv_error_event_raises(fake_sockets):
    message = {"type": "error", "error": {"code": "beta_api_shape_disabled", "message": "The Realtime Beta API is no longer supported."}}
    with pytest.raises(ASRProviderError, match="beta_api_shape_disabled"):
        recv_event(fake_sockets, message)


def test_recv_ignores_housekeeping_events(fake_sockets):
    assert recv_event(fake_sockets, {"type": "session.updated", "session": {}}) is None


def test_recv_ignores_non_json_messages(fake_sockets):
    assert recv_event(fake_sockets, "not-json") is None


def test_send_audio_passthrough_at_24k(fake_sockets):
    provider = make_provider(sample_rate=24000)
    provider.start()
    ws = fake_sockets[-1]
    pcm = np.arange(-100, 100, dtype="<i2").tobytes()
    provider.send_audio(pcm)
    message = json.loads(ws.sent[-1])
    assert message["type"] == "input_audio_buffer.append"
    assert base64.b64decode(message["audio"]) == pcm


def test_send_audio_resamples_16k_to_24k(fake_sockets):
    provider = make_provider(sample_rate=16000)
    provider.start()
    ws = fake_sockets[-1]
    total_in = 0
    rng = np.random.default_rng(42)
    for _ in range(10):
        chunk = rng.integers(-32768, 32767, size=160, dtype=np.int16)
        total_in += chunk.size
        provider.send_audio(chunk.astype("<i2").tobytes())
    total_out = sent_audio_samples(ws).size
    assert abs(total_out - total_in * 1.5) <= 2


def test_send_audio_skips_empty_chunk(fake_sockets):
    provider = make_provider(sample_rate=16000)
    provider.start()
    ws = fake_sockets[-1]
    provider.send_audio(b"")
    assert len(ws.sent) == 1  # only the session.update


def test_resample_upsample_16k_to_24k_ratio():
    pcm = np.zeros(160, dtype="<i2").tobytes()
    assert len(_resample_pcm16(pcm, 16000, 24000)) // 2 == 240


def test_resample_downsample_48k_to_24k_ratio():
    pcm = np.zeros(480, dtype="<i2").tobytes()
    assert len(_resample_pcm16(pcm, 48000, 24000)) // 2 == 240


def test_resample_non_integer_ratio():
    # 441 samples = 10 ms at 44.1 kHz -> 240 samples at 24 kHz.
    pcm = np.zeros(441, dtype="<i2").tobytes()
    assert len(_resample_pcm16(pcm, 44100, 24000)) // 2 == 240


def test_resample_preserves_constant_signal():
    pcm = np.full(160, 1234, dtype="<i2").tobytes()
    out = np.frombuffer(_resample_pcm16(pcm, 16000, 24000), dtype="<i2")
    assert np.all(out == 1234)


def test_resample_extreme_values_stay_in_int16_range():
    pcm = np.tile(np.array([32767, -32768], dtype="<i2"), 100).tobytes()
    out = np.frombuffer(_resample_pcm16(pcm, 16000, 24000), dtype="<i2")
    assert out.size > 0
    assert out.max() <= 32767
    assert out.min() >= -32768
