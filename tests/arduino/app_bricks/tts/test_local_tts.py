# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import contextlib
import json
import threading
import time

import numpy as np
import pytest

from arduino.app_bricks.tts import TextToSpeech, TTSBusyError
from arduino.app_bricks.tts.local_tts import TTS_MAX_CHARS
from arduino.app_peripherals.speaker import BaseSpeaker, FormatPlain, FormatPacked
from arduino.app_utils import App


class FakeResponse:
    """A runner response: read whole (status_code, json()) or streamed (status_code, iter_content(), json())."""

    def __init__(self, status_code=200, json_data=None, content=b"", chunks=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self._chunks = chunks
        self.close_called = False

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size):
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


def make_tts(monkeypatch, speaker, post_response, cancel_response=None, **tts_kwargs):
    models = [
        {
            # The runner reports the name from the model's config.json (underscore
            # style), while the brick is configured with the catalog id (piper-tts-en).
            "name": "pipertts_en",
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
        assert url.endswith("/tts/cancel"), f"{url}: the synthesis must be streamed"
        if cancel_response is not None:
            return cancel_response(url, json, **kwargs)
        return FakeResponse(content=b"cancelled")

    @contextlib.contextmanager
    def stream(method, url, json=None, **kwargs):
        assert (method, url.rsplit("/", 2)[-2:]) == ("POST", ["tts", "synthesize"])
        response = post_response(url, json, **kwargs)
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr("arduino.app_utils.http.get", lambda url, **kwargs: FakeResponse(json_data=models))
    monkeypatch.setattr("arduino.app_utils.http.post", post)
    monkeypatch.setattr("arduino.app_utils.http.stream", stream)

    tts = TextToSpeech(speaker=speaker, **tts_kwargs)
    App.unregister(tts)
    speaker.start()
    return tts


def test_resolve_voice_matches_runner_name_with_different_separators(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=b""))

    # Configured catalog id `piper-tts-en` resolves to the runner-reported
    # `pipertts_en`, which is what synthesis requests must send.
    assert tts._voice["model"] == "pipertts_en"
    assert tts._voice["language"] == "en"


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


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_speak_non_blocking_returns_before_playback_completes(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)

    assert speaker.first_chunk_written.wait(timeout=2)
    assert tts.is_speaking() is True

    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert len(speaker.chunks_written) == 3
    np.testing.assert_array_equal(np.concatenate(speaker.chunks_written), pcm_audio)


def test_speak_non_blocking_queues_sequential_playback(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    tts.speak("first", block=False)
    worker_thread = tts._speak_thread
    tts.speak("second", block=False)

    assert tts._speak_thread is worker_thread

    assert wait_until(lambda: not tts.is_speaking())
    assert post_calls == ["first", "second"]
    assert len(speaker.chunks_written) == 2


def test_speak_non_blocking_raises_when_queue_full(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()), max_queue_size=1)

    tts.speak("first", block=False)
    # The worker dequeues "first" and blocks on the first written chunk, leaving the queue empty.
    assert speaker.first_chunk_written.wait(timeout=2)
    tts.speak("second", block=False)

    with pytest.raises(TTSBusyError):
        tts.speak("third", block=False)

    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert len(speaker.chunks_written) == 6  # "first" and "second" played, "third" rejected


def test_max_queue_size_must_be_positive(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)

    with pytest.raises(ValueError):
        make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=b""), max_queue_size=0)


def test_speak_blocking_raises_when_background_playback_active(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)
    assert speaker.first_chunk_written.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.speak("second")

    tts.cancel()
    speaker.release_first_chunk.set()
    assert wait_until(lambda: not tts.is_speaking())


def test_speak_non_blocking_cancel_stops_playback_and_drops_queued_speech(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=pcm_audio.tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    tts.speak("first", block=False)
    assert speaker.first_chunk_written.wait(timeout=2)
    tts.speak("second", block=False)

    tts.cancel()
    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert post_calls == ["first"]
    assert len(speaker.chunks_written) == 1
    assert speaker.is_started() is True


def test_speak_non_blocking_logs_synthesis_errors(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    error_response = FakeResponse(status_code=500, json_data={"error": {"message": "synthesis boom"}})
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: error_response)

    logged_errors = []
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.logger.error", lambda msg, *args, **kwargs: logged_errors.append(msg))

    tts.speak("hello", block=False)

    assert wait_until(lambda: not tts.is_speaking())
    assert speaker.chunks_written == []
    assert any("synthesis boom" in msg for msg in logged_errors)


def test_stop_terminates_background_worker(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)
    worker_thread = tts._speak_thread

    assert speaker.first_chunk_written.wait(timeout=2)
    speaker.release_first_chunk.set()
    tts.stop()

    worker_thread.join(timeout=2)
    assert worker_thread.is_alive() is False
    assert tts._speak_thread is None
    assert tts.is_speaking() is False
    assert speaker.close_called is True


def test_synthesis_streams_from_a_real_http_server(monkeypatch):
    """No mock of the HTTP client: each HTTP chunk is yielded as it arrives, and a JSON error body becomes the error."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from arduino.app_utils import http

    real_stream = http.stream
    chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]
    send_second_chunk = threading.Event()

    class Runner(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request["text"] == "fail":
                body = json.dumps({"error": {"message": "synthesis boom"}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for index, chunk in enumerate(chunks):
                if index == 1:
                    send_second_chunk.wait(5)
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Runner)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        speaker = BlockingSpeaker(buffer_size=4)  # Reads of 4 int16 samples: one chunk at a time
        tts = make_tts(monkeypatch, speaker, post_response=None)
        monkeypatch.setattr(http, "stream", real_stream)
        tts.api_base_url = f"http://127.0.0.1:{server.server_address[1]}/audio-analytics/v1/api"

        audio = iter(tts.synthesize_pcm_stream("hello"))
        assert next(audio) == chunks[0]  # Before the runner sends the rest
        send_second_chunk.set()
        assert list(audio) == chunks[1:]

        with pytest.raises(RuntimeError, match="synthesis boom"):
            list(tts.synthesize_pcm_stream("fail"))
    finally:
        server.shutdown()
        server.server_close()
