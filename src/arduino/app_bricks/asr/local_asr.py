# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from collections.abc import Generator, Iterator
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from types import TracebackType
from contextlib import AbstractContextManager
from typing import Literal

import numpy as np
import requests
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from arduino.app_internal.core import resolve_address
from arduino.app_internal.core.module import get_brick_config, get_brick_configured_model
from arduino.app_peripherals.microphone import BaseMicrophone, Microphone, PauseDetector, chunk_level
from arduino.app_utils import AppError, Logger, brick

logger = Logger("ASR")


class ASRError(AppError):
    """Base class for ASR errors."""


class ASRBusyError(ASRError):
    """Raised when this ASR instance already has an active transcription session."""


class ASRServiceBusyError(ASRError):
    """Raised when the inference server rejects session creation because it is serving another client."""


class ASRUnavailableError(ASRError):
    """Raised when the inference service is unreachable or the connection drops unexpectedly."""


class AudioSourceExhausted(Exception):
    """
    Raised by finite-source adapters (WAV/ndarray) to signal end-of-data.
    Never raised by real BaseMicrophone implementations.
    """


def _dtype_to_pcm_format(dtype: np.dtype, is_packed: bool = False) -> str:
    """Map a numpy dtype to an API PCM format string (e.g. 'pcm_s16le')."""
    import sys

    byteorder = dtype.byteorder
    if byteorder in ("=", "|"):
        byteorder = "<" if sys.byteorder == "little" else ">"
    endian = "le" if byteorder == "<" else "be"
    kind = dtype.kind
    size = dtype.itemsize

    if kind == "i":
        if size == 1:
            return "pcm_s8"
        elif size == 2:
            return f"pcm_s16{endian}"
        elif size == 4:
            return f"pcm_s24{endian}" if is_packed else f"pcm_s32{endian}"
    elif kind == "u":
        if size == 1:
            return "pcm_u8"
        elif size == 2:
            return f"pcm_u16{endian}"
        elif size == 4:
            return f"pcm_u32{endian}"
    elif kind == "f":
        if size == 4:
            return f"pcm_f32{endian}"
        elif size == 8:
            return f"pcm_f64{endian}"

    raise ValueError(f"Unsupported numpy dtype for PCM format: {dtype}")


class _PauseFlushPolicy:
    """
    Decides when to ask the server for an early transcript of the speech buffered so far.

    Whisper only transcribes a segment once the server-side VAD closes it, which can
    take many seconds of continuous speech. Flushing earlier gives near-streaming output,
    but the server transcribes exactly what it has, so a flush in the middle of a word
    garbles that word. The policy therefore waits for a short pause, found by a
    :class:`PauseDetector`, and only forces a cut when the speaker does not pause at all.
    """

    def __init__(self, min_s: float, max_s: float, pause_s: float, min_voiced_s: float) -> None:
        self.min_s = min_s
        self.max_s = max_s
        self.min_voiced_s = min_voiced_s
        self._detector = PauseDetector(pause_s=pause_s)
        self.reset()

    def reset(self) -> None:
        """Start a new segment: called after a flush and when the server closes a segment itself."""
        self._in_segment = False
        self._segment_s = 0.0
        self._voiced_s = 0.0

    def update(self, level: float, duration_s: float) -> bool:
        """Account one chunk of ``duration_s`` seconds; returns True when a flush is due."""
        quiet = self._detector.update_level(level, duration_s)
        if not self._in_segment:
            if quiet:
                return False
            self._in_segment = True
        self._segment_s += duration_s
        if not quiet:
            self._voiced_s += duration_s
        if self._voiced_s < self.min_voiced_s:
            return False
        at_pause = self._segment_s >= self.min_s and self._detector.paused
        if at_pause or self._segment_s >= self.max_s:
            self.reset()
            return True
        return False


@dataclass(frozen=True)
class ASREvent:
    type: Literal["partial_text", "full_text"]
    data: str


class TranscriptionStream[T](AbstractContextManager["TranscriptionStream[T]"], Iterator[T]):
    """Iterator wrapper that guarantees proper teardown on context exit."""

    def __init__(self, generator: Generator[T]) -> None:
        self._generator = generator

    def __enter__(self) -> "TranscriptionStream[T]":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def __iter__(self) -> "TranscriptionStream[T]":
        return self

    def __next__(self) -> T:
        return next(self._generator)

    def close(self) -> None:
        self._generator.close()


@dataclass
class SessionInfo:
    session_id: str
    duration: int
    start_time: float
    result_queue: queue.Queue[ASREvent]
    chunk_queue: queue.Queue[bytes | object]  # object is for _END_SENTINEL
    cancelled: threading.Event
    language: str | None = None
    reader_thread: threading.Thread | None = None
    # Set by the reader when the flush policy asks for an early transcript
    flush_requested: threading.Event = field(default_factory=threading.Event)
    # Set by the receiver when the server closes a segment on its own (VAD end)
    segment_closed: threading.Event = field(default_factory=threading.Event)


_END_SENTINEL = object()  # Sentinel value to signal end of audio stream in the chunk queue
_FLUSH_MARK = object()  # Chunk-queue marker: request an early transcript once the audio before it is sent


class BaseASR:
    """
    Shared logic for ASR bricks. Subclasses bind the audio source
    via :meth:`_build_source` and add their own public ``transcribe*`` surface.

    Not decorated with ``@brick`` — only the concrete subclasses register.
    """

    _APP_SERVICE_NAME = "audio-analytics-runner"
    # Early transcripts (see _PauseFlushPolicy): cut at the first pause of at least
    # _FLUSH_PAUSE_SECONDS once a segment is _FLUSH_MIN_SECONDS long, or anyway at
    # _FLUSH_MAX_SECONDS, well inside Whisper's 30 s window.
    _FLUSH_MIN_SECONDS = 3.0
    _FLUSH_MAX_SECONDS = 10.0
    _FLUSH_PAUSE_SECONDS = 0.25
    _FLUSH_MIN_VOICED_SECONDS = 0.5
    _DEFAULT_VAD_MS = 700
    # Session creation loads the model onto the NPU; the server retries a failed
    # load several times, so a create can legitimately take well over 10 s.
    _CREATE_TIMEOUT_SECONDS = 60

    def __init__(self, source: object, language: str | None = None, translate: bool = False) -> None:
        # API configuration
        self.api_host = resolve_address(self._APP_SERVICE_NAME)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"
        self.ws_url = f"ws://{self.api_host}:{self.api_port}/stream"

        # Load the model configured at bricks level
        brick_config = get_brick_config(self.__class__)
        app_configured_model = get_brick_configured_model(brick_config.get("id") if brick_config else None)
        if app_configured_model:
            self.model = app_configured_model
        else:
            self.model = brick_config.get("model", None)

        self.language = language
        self.translate = translate

        self._source, self._owns_source = self._build_source(source)

        self._pcm_format = _dtype_to_pcm_format(
            self._source.format,
            self._source.format_is_packed,
        )

        self._worker_loop: Future[asyncio.AbstractEventLoop] = Future()
        self._stop_worker = threading.Event()

        self._active_session_lock = threading.Lock()
        self._active_session: SessionInfo | None = None

    def start(self) -> None:
        """Prepare the ASR for transcription. Starts the owned mic if applicable."""
        logger.debug("Starting ASR and preparing resources...")
        self._stop_worker.clear()
        if self._worker_loop.done():
            self._worker_loop = Future()
        if self._owns_source:
            self._source.start()
        self._warmup()

    def stop(self) -> None:
        """Stop the ASR and clean up resources. Stops the owned mic if applicable."""
        logger.debug("Stopping ASR and cleaning up resources...")
        self._stop_worker.set()
        self._worker_loop.cancel()
        self.cancel()
        if self._owns_source:
            self._source.stop()
        logger.debug("Stopped ASR and cleaned up resources.")

    def cancel(self) -> None:
        """Cancel the active transcription session, if any."""
        active = self._active_session
        if active is None:
            logger.debug("No active session to cancel")
            return
        logger.debug(f"Cancelling session {active.session_id}")
        active.cancelled.set()

    def is_transcribing(self) -> bool:
        """
        Tells if a transcription session is currently active on this instance.

        Returns:
            bool: True if a session is active, False otherwise.
        """
        return self._active_session is not None

    def _build_source(self, source: object) -> tuple:
        """Bind the audio source. Subclasses must override."""
        raise NotImplementedError("Subclasses must override _build_source")

    def _ensure_source_started(self) -> None:
        if not self._source.is_started():
            raise RuntimeError("Audio source must be started before transcription.")

    def _collect_transcription(self, stream: TranscriptionStream[ASREvent]) -> str:
        """
        Drain an event stream into a single transcription string.

        The server sends a sentence as consecutive ``partial_text`` pieces, one
        per early flush, then a ``full_text`` with the whole sentence. Sentences
        are concatenated from their ``full_text``; the pieces of a sentence the
        session ended on before its ``full_text`` (e.g. stopped right after the
        speech) are appended as they are. Returns ``""`` if no speech was detected.
        """
        pending: list[str] = []  # pieces of the sentence not closed by a full_text yet
        final_text = ""

        with stream:
            for chunk in stream:
                if not chunk.data.strip():
                    continue
                if chunk.type == "partial_text":
                    pending.append(chunk.data)
                elif chunk.type == "full_text":
                    final_text += chunk.data  # already contains its pieces
                    pending.clear()

        if pending:
            logger.debug("Session ended before the last full_text, using its partial_text pieces")
            final_text += "".join(pending)
        return final_text if final_text.strip() else ""

    @brick.execute
    def _asyncio_loop(self) -> None:
        """Dedicated thread for the asyncio event loop hosting session coroutines."""
        logger.debug("Asyncio event loop starting")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop.set_result(loop)

        async def keep_alive() -> None:
            while not self._stop_worker.is_set():
                await asyncio.sleep(0.1)

        try:
            loop.run_until_complete(keep_alive())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            logger.debug("Asyncio event loop stopped")

    def _warmup(self) -> None:
        """Best-effort warmup: create and immediately close a transcription session so the
        inference container loads the ASR model before the first real transcription."""
        if self._stop_worker.is_set():
            return
        started_at = time.perf_counter()
        try:
            session_id = self._create_transcription_session(language=self.language, translate=self.translate)
        except Exception as e:
            logger.warning(f"ASR warmup failed during session creation: {e}")
            return
        try:
            self._close_transcription_session(session_id)
        except Exception as e:
            logger.warning(f"ASR warmup failed during closing session {session_id}: {e}")
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"ASR warmup completed in {elapsed_ms:.2f} ms")

    def _transcribe_stream(self, duration: int = 0, vad_ms: int | None = None) -> Generator[ASREvent]:
        if self._stop_worker.is_set():
            raise RuntimeError("Brick is stopping or already stopped")
        try:
            worker_loop = self._worker_loop.result(timeout=5)
        except TimeoutError:
            raise RuntimeError("Worker loop is not initialized. Call start() first.") from None
        except CancelledError:
            raise RuntimeError("Brick is stopping or already stopped") from None
        if self._stop_worker.is_set():
            raise RuntimeError("Brick is stopping or already stopped")

        if not self._active_session_lock.acquire(blocking=False):
            active_id = self._active_session.session_id if self._active_session else "unknown"
            raise ASRBusyError(
                f"A transcription session (id={active_id}) is already active on this instance. "
                f"Create a separate ASR instance for concurrent transcriptions."
            )

        session_info: SessionInfo | None = None
        future = None

        try:
            # Snapshot current language and translate flag for the session
            session_language = self.language
            session_translate = self.translate
            session_id = self._create_transcription_session(vad_ms=vad_ms, language=session_language, translate=session_translate)
            session_info = SessionInfo(
                session_id=session_id,
                duration=duration,
                start_time=time.time(),
                result_queue=queue.Queue(),
                chunk_queue=queue.Queue(maxsize=100),
                language=session_language,
                cancelled=threading.Event(),
            )
            self._active_session = session_info

            future = asyncio.run_coroutine_threadsafe(
                self._transcription_session_handler(session_info),
                worker_loop,
            )

            while not future.done():
                try:
                    yield session_info.result_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

            while True:
                try:
                    yield session_info.result_queue.get_nowait()
                except queue.Empty:
                    break

            future.result()

        except GeneratorExit:
            logger.debug(f"Transcription interrupted by user for session {session_info.session_id if session_info else '?'}")
            if session_info:
                session_info.cancelled.set()
            if future and not future.done():
                future.cancel()
                try:
                    future.result(timeout=2)
                except Exception:
                    pass
            raise

        except TimeoutError:
            raise

        except ASRError:
            raise

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        finally:
            if session_info is not None:
                session_info.cancelled.set()
            self._active_session = None
            self._active_session_lock.release()

    def _create_transcription_session(self, vad_ms: int | None = None, language: str | None = None, translate: bool = False) -> str:
        sampling_rate = str(self._source.sample_rate)
        channels = str(self._source.channels)

        hangover_ms = str(vad_ms if vad_ms is not None else self._DEFAULT_VAD_MS)

        create_url = f"{self.api_base_url}/transcriptions/create"
        create_data = {
            "model": self.model,
            "stream": True,
            "translate": translate,
            "parameters": json.dumps([
                {"key": "sampling_rate", "value": sampling_rate},
                {"key": "channels", "value": channels},
                {"key": "format", "value": self._pcm_format},
                {"key": "vad", "value": hangover_ms},
            ]),
        }
        if language is not None:
            create_data["language"] = language

        try:
            start = time.monotonic()
            response = requests.post(url=create_url, json=create_data, timeout=self._CREATE_TIMEOUT_SECONDS)
            elapsed = time.monotonic() - start
            if elapsed > 5:
                logger.warning(f"Session creation took {elapsed:.1f}s")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ASRUnavailableError(f"Inference service unreachable: {e}") from None

        # The server answers 409 (code "conflict") when another session is active
        if response.status_code in (400, 409):
            try:
                err = response.json().get("error", {})
                msg = err.get("message", "")
            except Exception:
                msg = response.text or ""
            if response.status_code == 409 or "transcription session is already active" in msg:
                raise ASRServiceBusyError(msg or "Inference server is serving another client")
            raise ASRError(msg or f"Failed to create transcription session: {response.status_code}")

        if response.status_code != 200:
            msg = f"Failed to create transcription session: {response.status_code}"
            try:
                err = response.json().get("error", {})
                msg = err.get("message", msg)
            except Exception:
                pass
            raise ASRError(msg)

        result = response.json()
        session_id = result.get("session_id")
        if not session_id:
            raise ASRError("No session ID returned from transcription API")

        state = result.get("state")
        if state != "asr_initialized":
            raise ASRError(f"Unexpected session state: {state}")

        return session_id

    async def _transcription_session_handler(self, session_info: SessionInfo) -> None:
        session_id = session_info.session_id

        reader = threading.Thread(
            target=self._reader_thread_body,
            args=(session_info,),
            daemon=True,
            name=f"ASRReader-{session_id}",
        )
        session_info.reader_thread = reader
        reader.start()

        try:
            try:
                async with (
                    websockets.connect(
                        self.ws_url,
                        ping_interval=10,
                        ping_timeout=5,
                        close_timeout=5,
                    ) as write_ws,
                    websockets.connect(
                        self.ws_url,
                        ping_interval=10,
                        ping_timeout=5,
                        close_timeout=5,
                    ) as read_ws,
                ):
                    await self._await_connection_established(write_ws, "write_ws")
                    await self._await_connection_established(read_ws, "read_ws")

                    send_task = asyncio.create_task(self._send_pcm_stream(websocket=write_ws, session_info=session_info))
                    receive_task = asyncio.create_task(self._receive_transcription(websocket=read_ws, session_info=session_info))
                    drain_write_ws_task = asyncio.create_task(self._drain_websocket(write_ws, session_info, "write_ws"))
                    flush_task = asyncio.create_task(self._periodic_flush(session_info))

                    try:
                        while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                            done, _ = await asyncio.wait(
                                {send_task, receive_task, drain_write_ws_task},
                                timeout=0.1,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                continue
                            for task in done:
                                exc = task.exception()
                                if exc:
                                    raise exc
                            break

                    finally:
                        for task in (flush_task, send_task):
                            if task and not task.done():
                                task.cancel()

                        await asyncio.gather(flush_task, send_task, return_exceptions=True)

                        # Server protocol: close session BEFORE tearing down WebSockets
                        try:
                            await asyncio.to_thread(self._close_transcription_session, session_id)
                        except Exception as e:
                            logger.error(f"Failed to close session {session_id} during teardown: {e}")

                        session_info.cancelled.set()

                        for task in (receive_task, drain_write_ws_task):
                            if task and not task.done():
                                task.cancel()

                        await asyncio.gather(receive_task, drain_write_ws_task, return_exceptions=True)

            except OSError as e:
                raise ASRUnavailableError(f"Failed to connect to inference service: {e}") from None

        finally:
            session_info.cancelled.set()
            join_timeout = 2.0
            await asyncio.to_thread(reader.join, join_timeout)
            if reader.is_alive():
                logger.warning(f"Reader thread for session {session_id} did not exit within {join_timeout}s; leaking as daemon")

    def _reader_thread_body(self, session_info: SessionInfo) -> None:
        session_id = session_info.session_id
        start_time = session_info.start_time
        duration = session_info.duration
        policy = _PauseFlushPolicy(
            min_s=self._FLUSH_MIN_SECONDS,
            max_s=self._FLUSH_MAX_SECONDS,
            pause_s=self._FLUSH_PAUSE_SECONDS,
            min_voiced_s=self._FLUSH_MIN_VOICED_SECONDS,
        )
        frames_per_second = self._source.sample_rate * self._source.channels
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                if duration > 0 and (time.time() - start_time) >= duration:
                    logger.debug(f"Session {session_id} duration limit reached: {duration}s")
                    break
                try:
                    chunk = self._source.capture()
                except AudioSourceExhausted:
                    logger.debug(f"Session {session_id} audio source exhausted")
                    break
                except Exception as e:
                    logger.error(f"Reader thread capture error for session {session_id}: {e}")
                    break
                if chunk is None:
                    continue  # transient (paused/underrun) — keep going
                if session_info.segment_closed.is_set():
                    session_info.segment_closed.clear()
                    policy.reset()
                flush_due = policy.update(chunk_level(chunk, self._source.format_is_packed), chunk.size / frames_per_second)
                self._enqueue(session_info, chunk.tobytes())
                if flush_due:
                    # Queued behind the audio, so the server cuts exactly at this pause
                    self._enqueue(session_info, _FLUSH_MARK)
        finally:
            # Block until the end sentinel is enqueued so the sender always sees it.
            # This is required if exit condition is duration or WAV exhaustion.
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                try:
                    session_info.chunk_queue.put(_END_SENTINEL, timeout=0.2)
                    break
                except queue.Full:
                    continue
            logger.debug(f"Reader thread exited for session {session_id}")

    def _enqueue(self, session_info: SessionInfo, item: bytes | object) -> None:
        """Queue an item for the sender: live mics drop on overflow, finite sources wait."""
        try:
            session_info.chunk_queue.put_nowait(item)
        except queue.Full:
            if not isinstance(self._source, BaseMicrophone):
                try:
                    session_info.chunk_queue.put(item)
                except queue.Full:
                    logger.warning(f"Send queue full for session {session_info.session_id}, dropping chunk")
            else:
                logger.warning(f"Send queue full for session {session_info.session_id}, dropping chunk")

    async def _await_connection_established(self, websocket: websockets.ClientConnection, label: str) -> None:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except (TimeoutError, ConnectionClosed) as e:
            raise ASRUnavailableError(f"{label} handshake failed: {e}") from None
        msg = json.loads(raw)
        if msg.get("state") != "connection_established":
            raise RuntimeError(f"{label} expected connection_established, got {msg}")

    async def _send_pcm_stream(self, websocket: websockets.ClientConnection, session_info: SessionInfo) -> int:
        session_id = session_info.session_id
        chunks_sent = 0
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                try:
                    item = await asyncio.to_thread(session_info.chunk_queue.get, True, 0.2)
                except queue.Empty:
                    continue
                if item is _END_SENTINEL:
                    break
                if item is _FLUSH_MARK:
                    session_info.flush_requested.set()
                    continue

                assert isinstance(item, bytes), f"Expected bytes, got {type(item)}"
                message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": base64.b64encode(item).decode("utf-8"),
                }
                await websocket.send(json.dumps(message))
                chunks_sent += 1
                if chunks_sent % 20 == 0:
                    logger.debug(f"Session {session_id}: sent {chunks_sent} audio chunks")

            logger.debug(f"Finished sending PCM stream for session {session_id}, chunks_sent={chunks_sent}")
            return chunks_sent

        except asyncio.CancelledError:
            logger.debug(f"PCM stream sending cancelled for session {session_id}")
            raise
        except ConnectionClosedOK:
            logger.debug(f"WebSocket closed as expected while sending PCM stream for session {session_id}")
            return chunks_sent
        except ConnectionClosed as e:
            raise ASRUnavailableError(f"WebSocket connection lost while sending for session {session_id}: {e}") from None

    async def _receive_transcription(self, websocket: websockets.ClientConnection, session_info: SessionInfo) -> None:
        session_id = session_info.session_id
        result_queue = session_info.result_queue

        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except TimeoutError:
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse WebSocket message: {message}")
                    continue

                message_session_id = data.get("session_id")
                if message_session_id is not None and message_session_id != session_id:
                    logger.warning(f"Ignoring WebSocket message for session {message_session_id}; current session is {session_id}. Message: {data}")
                    continue

                logger.debug(f"Received WebSocket message for session {session_id}. Message: {data}")

                evt_type = data.get("type") or data.get("message_type")
                evt_state = data.get("state")
                evt_text = data.get("text", "")

                if evt_state == "connection_established":
                    continue
                elif evt_type == "transcript.text.delta":
                    logger.debug(f"Session {session_id} putting partial transcription: {evt_text}")
                    result_queue.put(ASREvent("partial_text", evt_text))
                    continue
                elif evt_type == "transcript.text.done":
                    logger.debug(f"Session {session_id} putting full transcription: {evt_text}")
                    session_info.segment_closed.set()
                    result_queue.put(ASREvent("full_text", evt_text))
                    continue
                elif evt_type == "transcript.event":
                    if evt_state == "asr_initialized":
                        logger.debug(f"ASR initialized for session {session_id}")
                        continue
                    elif evt_state == "speech_start":
                        logger.debug(f"Speech started for session {session_id}")
                        continue
                    elif evt_state == "speech_end":
                        logger.debug(f"Speech ended for session {session_id}")
                        continue
                    else:
                        logger.debug(f"Unknown transcript.event for session {session_id}: state={evt_state!r}, text={evt_text!r}")
                        continue
                elif evt_type == "error":
                    error_msg = data.get("message", "Unknown ASR error")
                    raise RuntimeError(error_msg)
                elif evt_type == "connection_close":
                    logger.warning(f"WebSocket connection closed for session {session_id}")
                    break
                else:
                    logger.warning(f"Unknown message type received for session {session_id}: type={evt_type!r}, msg={data}")
                    continue

        except asyncio.CancelledError:
            logger.debug(f"Receive task cancelled for session {session_id}")
            raise
        except ConnectionClosedOK:
            logger.debug(f"WebSocket closed as expected while receiving transcription for session {session_id}")
            return
        except ConnectionClosed as e:
            raise ASRUnavailableError(f"WebSocket connection lost while receiving for session {session_id}: {e}") from None

    async def _drain_websocket(self, websocket: websockets.ClientConnection, session_info: SessionInfo, label: str) -> None:
        session_id = session_info.session_id

        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except TimeoutError:
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.debug(f"Drained non-JSON WebSocket message from {label}: {message}")
                    continue

                message_session_id = data.get("session_id")
                if message_session_id is not None and message_session_id != session_id:
                    logger.debug(
                        f"Drained WebSocket message from {label} for session {message_session_id}; current session is {session_id}. Message: {data}"
                    )
                    continue

                logger.debug(f"Drained WebSocket message from {label} for session {session_id}: {data}")

        except asyncio.CancelledError:
            logger.debug(f"Drain task cancelled for {label}, session {session_id}")
            raise
        except ConnectionClosedOK:
            logger.debug(f"WebSocket {label} closed as expected while draining for session {session_id}")
        except ConnectionClosed as e:
            logger.debug(f"WebSocket {label} closed while draining for session {session_id}: {e}")

    async def _periodic_flush(self, session_info: SessionInfo) -> None:
        """Send the flushes the reader thread requests (see _PauseFlushPolicy)."""
        session_id = session_info.session_id
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                await asyncio.sleep(0.05)
                if not session_info.flush_requested.is_set():
                    continue
                session_info.flush_requested.clear()
                await asyncio.to_thread(self._flush_transcription_session, session_id)
        except asyncio.CancelledError:
            logger.debug(f"Periodic flush cancelled for session {session_id}")
            raise

    def _flush_transcription_session(self, session_id: str) -> None:
        logger.debug(f"Flushing transcription session {session_id}")
        url = f"{self.api_base_url}/transcriptions/flush"
        try:
            response = requests.post(url, json={"session_id": session_id}, timeout=3)
        except Exception as e:
            logger.warning(f"Failed to flush session {session_id}: {e}")
            return
        if response.status_code != 200:
            logger.warning(f"Failed to flush session {session_id}: status {response.status_code}: {response.text}")
            return
        logger.debug(f"Session {session_id} flushed successfully")

    def _close_transcription_session(self, session_id: str) -> None:
        logger.debug(f"Closing transcription session {session_id}")
        url = f"{self.api_base_url}/transcriptions/close"
        try:
            response = requests.post(url, json={"session_id": session_id}, timeout=20)
        except Exception:
            raise
        if response.status_code != 200:
            raise RuntimeError(f"HTTP status {response.status_code}: {response.text}")
        logger.debug(f"Session {session_id} closed successfully")


@brick
class AutomaticSpeechRecognition(BaseASR):
    """ASR brick for live audio transcription from a microphone."""

    def __init__(
        self,
        mic: BaseMicrophone | None = None,
        language: str | None = None,
        translate: bool = False,
    ) -> None:
        """
        ASR brick that transcribes a live audio stream from a microphone.

        Args:
            mic: Microphone to be captured for transcription. One of:
                BaseMicrophone: used as-is; the caller owns its
                    lifecycle (ASR never calls start()/stop() on it).
                None: ASR constructs a default Microphone() and owns its
                    lifecycle (started on start(), stopped on stop()).
                Default: None.
            language (str): Language code for the ASR model (e.g. "en" for
                English). This is typically auto-detected by the model,
                but can be overridden here if needed. It is exposed as
                the public ``language`` attribute and may be reassigned at
                runtime; the new value takes effect on the next session.
            translate (bool): If ``True``, speech is translated to English instead
                of being transcribed in the language it was spoken in. It is valid
                only for models that support translation, so it costs no extra
                model: the ASR model itself does the translating. The Whisper
                models this brick runs, including the default
                ``whisper-small-quantized``, support it, and their translate task
                always targets English. Any of their source languages can be
                translated, but English is the only possible target. Set
                ``language`` as well to skip source auto-detection. It is exposed
                as the public ``translate`` attribute and may be reassigned at
                runtime; the new value takes effect on the next session.
                Default: ``False``.

        Note:
            Only one transcription can be active at a time.
        """
        super().__init__(source=mic, language=language, translate=translate)

    def _build_source(self, source: object) -> tuple:
        if source is None:
            return Microphone(0), True  # First plugged mic, shared with other consumers
        if isinstance(source, BaseMicrophone):
            return source, False
        raise TypeError(f"Unsupported source type: {type(source)!r}")

    def transcribe(self, duration: int = 60) -> str:
        """
        Transcribe audio for a duration and return the final text.

        Args:
            duration (int): Maximum recording time in seconds. ``0`` means unbounded.
                Default: ``60``.

        Returns:
            str: The transcribed text, or an empty string if no speech was detected.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the microphone has not been started.
        """
        return self._collect_transcription(self.transcribe_stream(duration=duration))

    def transcribe_stream(self, duration: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio for a duration and yield intermediate transcription events.

        Args:
            duration (int): Maximum recording time in seconds. ``0`` means unbounded.
                Default: ``0``.

        Yields:
            ASREvent: objects representing transcription events.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the microphone has not been started.
        """
        self._ensure_source_started()
        return TranscriptionStream(self._transcribe_stream(duration=duration))

    def transcribe_sentence(self, timeout: int = 0) -> str:
        """
        Transcribe a sentence returning the full text.

        Runs until the sentence boundary is detected, the timeout elapses
        without one.

        Args:
            timeout (int): Maximum recording time in seconds. ``0`` means no timeout.
                Default: ``0``.

        Returns:
            str: The transcribed text, or an empty string if no speech was detected.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the connection drops mid-session.
            RuntimeError: If the microphone has not been started.
        """
        return self._collect_transcription(self.transcribe_sentence_stream(timeout=timeout))

    def transcribe_sentence_stream(self, timeout: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe a sentence and yield the intermediate transcription events.

        The stream ends after the sentence boundary is detected, the timeout
        elapses without one.

        Args:
            timeout (int): Maximum recording time in seconds. ``0`` means no timeout.
                Default: ``0``.

        Yields:
            ASREvent: objects representing transcription events.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the microphone has not been started.
        """
        self._ensure_source_started()

        def sentence_gen() -> Generator[ASREvent]:
            inner = self._transcribe_stream(duration=timeout)
            try:
                for event in inner:
                    yield event
                    if event.type == "full_text" and event.data.strip():
                        return
            finally:
                inner.close()

        return TranscriptionStream(sentence_gen())

    def transcribe_until_cancelled(self) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio indefinitely and yield intermediate transcription events.

        The stream ends only when :meth:`cancel` is called.

        Yields:
            ASREvent: objects representing transcription events.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the microphone has not been started.
        """
        self._ensure_source_started()
        return TranscriptionStream(self._transcribe_stream(duration=0))
