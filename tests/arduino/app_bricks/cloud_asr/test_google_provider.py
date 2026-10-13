# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time
from collections.abc import Iterator

import pytest
from google.cloud.speech import (
    SpeechRecognitionAlternative,
    StreamingRecognitionResult,
    StreamingRecognizeRequest,
    StreamingRecognizeResponse,
)

from arduino.app_bricks.cloud_asr.providers import google as google_provider
from arduino.app_bricks.cloud_asr.providers.google import GoogleSpeech
from arduino.app_bricks.cloud_asr.providers.types import ASRProviderEvent


class FakeSpeechClient:
    """SpeechClient stand-in that records the requests of each stream and answers the first one."""

    def __init__(self, *, client_options=None):
        self.client_options = client_options
        self.streams: list[list[StreamingRecognizeRequest]] = []

    def streaming_recognize(self, requests: Iterator[StreamingRecognizeRequest]) -> Iterator[StreamingRecognizeResponse]:
        sent: list[StreamingRecognizeRequest] = []
        self.streams.append(sent)
        for request in requests:
            sent.append(request)
            if len(self.streams) == 1 and len(sent) == 2:
                # The configuration and the first audio chunk have arrived: answer with a final transcript
                yield StreamingRecognizeResponse(
                    results=[StreamingRecognitionResult(alternatives=[SpeechRecognitionAlternative(transcript="hello")], is_final=True)]
                )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpeechClient]:
    clients: list[FakeSpeechClient] = []

    def make_client(**kwargs):
        client = FakeSpeechClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(google_provider, "SpeechClient", make_client)
    return clients


def _recv_until(provider: GoogleSpeech, event_type: str, timeout: float = 2.0) -> ASRProviderEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = provider.recv()
        if event is not None and event.type == event_type:
            return event
    raise AssertionError(f"No '{event_type}' event within {timeout}s")


def test_stream_sends_the_configuration_before_the_audio(fake_client: list[FakeSpeechClient]):
    provider = GoogleSpeech(api_key="key", language="en")
    provider.start()
    try:
        provider.send_audio(b"\x01\x02")
        event = _recv_until(provider, "text")
    finally:
        provider.stop()

    assert event.data == "hello"
    client = fake_client[0]
    assert client.client_options is not None and client.client_options.api_key == "key"
    config_request, audio_request = client.streams[0][:2]
    assert config_request.streaming_config.config.language_code == "en-US"
    assert config_request.audio_content == b""
    assert audio_request.audio_content == b"\x01\x02"
