# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading

import numpy as np
import pytest

from arduino.app_bricks.tts import TextToSpeech, TTSBusyError
from arduino.app_bricks.tts.local_tts import TTS_MAX_CHARS
from arduino.app_peripherals.speaker import BaseSpeaker, FormatPlain, FormatPacked
from arduino.app_utils import App


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self._chunks = chunks
        self.close_called = False

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=1):
        if self._chunks is not None:
            yield from self._chunks
            return

        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def close(self):
        self.close_called = True


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


def make_tts(monkeypatch, speaker, post_response, cancel_response=None):
    models = [
        {
            "name": "piper-tts-en",
            "voices": [
                {
                    "language": "en",
                    "name": "default",
                    "sample_rate": 44100,
                }
            ],
        }
    ]

    def post(url, json=None, **kwargs):
        if url.endswith("/tts/cancel"):
            if cancel_response is not None:
                return cancel_response(url, json, **kwargs)
            return FakeResponse(content=b"cancelled")
        return post_response(url, json, **kwargs)

    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.requests.get", lambda url: FakeResponse(json_data=models))
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.requests.post", post)

    tts = TextToSpeech(speaker=speaker)
    App.unregister(tts)
    speaker.start()
    return tts


def test_cancel_without_active_speech_keeps_speaker_running(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    tts.cancel()

    assert speaker.is_started() is True
    assert speaker.close_called is False
    assert speaker.chunks_written == []


def test_chunk_text_splits_on_sentence_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 1000}. {'b' * 1000}"

    chunks = tts._chunk_text(text)

    assert chunks == [f"{'a' * 1000}.", "b" * 1000]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_newline_when_no_sentence_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 900}\n{'b' * 300}"

    chunks = tts._chunk_text(text)

    assert chunks == ["a" * 900, "b" * 300]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_space_when_no_sentence_or_newline_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 900} {'b' * 300}"

    chunks = tts._chunk_text(text)

    assert chunks == ["a" * 900, "b" * 300]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_character_limit(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    chunks = tts._chunk_text("é" * 1200)

    assert chunks == ["é" * TTS_MAX_CHARS, "é" * 176]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_speak_synthesizes_text_chunks(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []
    text = f"{'a' * 1000}. {'b' * 1000}"

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        assert json["keep_alive"] is True
        assert kwargs["stream"] is True
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)
    expected_chunks = tts._chunk_text(text)

    tts.speak(text)

    assert post_calls == expected_chunks
    assert len(speaker.chunks_written) == len(expected_chunks)


def test_synthesize_pcm_stream_yields_response_chunks(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    audio_chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]

    def post_response(url, json, **kwargs):
        assert kwargs["stream"] is True
        assert json["keep_alive"] is False
        return FakeResponse(chunks=audio_chunks)

    tts = make_tts(monkeypatch, speaker, post_response)

    assert list(tts.synthesize_pcm_stream("hello")) == audio_chunks
    assert tts.synthesize_pcm("hello") == b"".join(audio_chunks)


def test_speak_plays_first_stream_chunk_before_response_completes(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    first_chunk_sent = threading.Event()
    release_second_chunk = threading.Event()
    audio_chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]

    def stream_chunks():
        first_chunk_sent.set()
        yield audio_chunks[0]
        release_second_chunk.wait(timeout=2)
        yield audio_chunks[1]

    def post_response(url, json, **kwargs):
        return FakeResponse(chunks=stream_chunks())

    tts = make_tts(monkeypatch, speaker, post_response)
    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)

    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)
    assert first_chunk_sent.is_set()
    assert speak_thread.is_alive() is True
    np.testing.assert_array_equal(speaker.chunks_written[0], np.arange(4, dtype=np.int16))

    speaker.release_first_chunk.set()
    release_second_chunk.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert len(speaker.chunks_written) == 2


def test_cancel_stops_playback_without_stopping_speaker(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    cancel_calls = []
    tts = make_tts(
        monkeypatch,
        speaker,
        lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()),
        cancel_response=lambda url, json=None, **kwargs: cancel_calls.append(url) or FakeResponse(content=b"cancelled"),
    )

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
    assert cancel_calls == [tts.api_base_url + "/tts/cancel"]


def test_cancel_during_synthesis_skips_playback(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    synthesis_started = threading.Event()
    release_synthesis = threading.Event()
    pcm_audio = np.arange(12, dtype=np.int16)

    def post_response(url, json, **kwargs):
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


def test_synthesize_pcm_raises_when_busy(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    first_synthesis_started = threading.Event()
    release_first_synthesis = threading.Event()
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        if json["text"] == "first":
            first_synthesis_started.set()
            release_first_synthesis.wait(timeout=2)
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    first_thread = threading.Thread(target=tts.synthesize_pcm, args=("first",), daemon=True)
    first_thread.start()
    assert first_synthesis_started.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.synthesize_pcm("second")

    assert post_calls == ["first"]

    release_first_synthesis.set()
    first_thread.join(timeout=2)

    assert first_thread.is_alive() is False
    assert post_calls == ["first"]


def test_speak_raises_when_busy(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.speak("second")

    speaker.release_first_chunk.set()
    tts.cancel()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
