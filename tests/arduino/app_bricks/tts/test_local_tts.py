# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading

import numpy as np

from arduino.app_bricks.tts import TextToSpeech
from arduino.app_bricks.tts.local_tts import TTS_MAX_BYTES
from arduino.app_peripherals.speaker import BaseSpeaker, FormatPlain, FormatPacked
from arduino.app_utils import App


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data


class BlockingSpeaker(BaseSpeaker):
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = 4,
        auto_reconnect: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, channels=channels, format=format, buffer_size=buffer_size, auto_reconnect=auto_reconnect)
        self.chunks_written = []
        self.first_chunk_written = threading.Event()
        self.release_first_chunk = threading.Event()
        self.close_called = False

    def _open_speaker(self):
        pass

    def _close_speaker(self):
        self.close_called = True

    def _write_audio(self, audio_chunk: np.ndarray):
        self.chunks_written.append(audio_chunk.copy())
        if len(self.chunks_written) == 1:
            self.first_chunk_written.set()
            self.release_first_chunk.wait(timeout=2)


def make_tts(monkeypatch, speaker, post_response):
    models = [
        {
            "name": "melo-tts-en",
            "voices": [
                {
                    "language": "en",
                    "name": "default",
                    "sample_rate": 44100,
                }
            ],
        }
    ]

    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.requests.get", lambda url: FakeResponse(json_data=models))
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.requests.post", post_response)

    tts = TextToSpeech(speaker=speaker)
    App.unregister(tts)
    speaker.start()
    return tts


def test_cancel_without_active_speech_keeps_speaker_running(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    tts.cancel()

    assert speaker.is_started() is True
    assert speaker.close_called is False
    assert speaker.chunks_written == []


def test_chunk_text_splits_on_sentence_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 1000}. {'b' * 1000}"

    chunks = tts.chunk_text(text)

    assert chunks == [f"{'a' * 1000}.", "b" * 1000]
    assert all(len(chunk.encode("utf-8")) <= TTS_MAX_BYTES for chunk in chunks)


def test_chunk_text_preserves_utf8_boundaries(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    chunks = tts.chunk_text("é" * 600)

    assert chunks == ["é" * 512, "é" * 88]
    assert all(len(chunk.encode("utf-8")) <= TTS_MAX_BYTES for chunk in chunks)


def test_speak_synthesizes_text_chunks(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []
    text = f"{'a' * 1000}. {'b' * 1000}"

    def post_response(url, json):
        post_calls.append(json["text"])
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)
    expected_chunks = tts.chunk_text(text)

    tts.speak(text)

    assert post_calls == expected_chunks
    assert len(speaker.chunks_written) == len(expected_chunks)


def test_cancel_stops_playback_without_stopping_speaker(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json: FakeResponse(content=pcm_audio.tobytes()))

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)
    tts.cancel()
    speaker.release_first_chunk.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert len(speaker.chunks_written) == 1
    np.testing.assert_array_equal(speaker.chunks_written[0], pcm_audio[:4])
    assert speaker.is_started() is True
    assert speaker.close_called is False


def test_cancel_during_synthesis_skips_playback(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    synthesis_started = threading.Event()
    release_synthesis = threading.Event()
    pcm_audio = np.arange(12, dtype=np.int16)

    def post_response(url, json):
        synthesis_started.set()
        release_synthesis.wait(timeout=2)
        return FakeResponse(content=pcm_audio.tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert synthesis_started.wait(timeout=2)
    tts.cancel()
    release_synthesis.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert speaker.chunks_written == []
    assert speaker.is_started() is True
    assert speaker.close_called is False


def test_synthesize_pcm_serializes_requests(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    first_synthesis_started = threading.Event()
    second_synthesis_started = threading.Event()
    release_first_synthesis = threading.Event()
    post_calls = []
    results = []
    errors = []

    def post_response(url, json):
        post_calls.append(json["text"])
        if json["text"] == "first":
            first_synthesis_started.set()
            release_first_synthesis.wait(timeout=2)
        else:
            second_synthesis_started.set()
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    def synthesize(text):
        try:
            results.append(tts.synthesize_pcm(text))
        except Exception as e:
            errors.append(e)

    first_thread = threading.Thread(target=synthesize, args=("first",), daemon=True)
    second_thread = threading.Thread(target=synthesize, args=("second",), daemon=True)

    first_thread.start()
    assert first_synthesis_started.wait(timeout=2)
    second_thread.start()

    assert second_synthesis_started.wait(timeout=0.1) is False
    assert post_calls == ["first"]

    release_first_synthesis.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
    assert errors == []
    assert len(results) == 2
    assert post_calls == ["first", "second"]
