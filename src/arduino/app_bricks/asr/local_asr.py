# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import io
import json
import queue
import threading
import time
import wave
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import ContextManager, Generic, Literal, TypeVar

import numpy as np
import requests
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from arduino.app_internal.core import resolve_address
from arduino.app_internal.core.module import get_brick_config, get_brick_configured_model
from arduino.app_peripherals.microphone import BaseMicrophone, Microphone
from arduino.app_utils import Logger, brick

logger = Logger("ASR")


class ASRError(Exception):
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


@dataclass(frozen=True)
class ASREvent:
    type: Literal["partial_text", "full_text"]
    data: str


T = TypeVar("T")


class TranscriptionStream(Generic[T], ContextManager["TranscriptionStream[T]"], Iterator[T]):
    """Iterator wrapper that guarantees proper teardown on context exit."""

    def __init__(self, generator: Generator[T, None, None]):
        self._generator = generator

    def __enter__(self) -> "TranscriptionStream[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> "TranscriptionStream[T]":
        return self

    def __next__(self) -> T:
        return next(self._generator)

    def close(self) -> None:
        self._generator.close()


class InMemoryAudioSource:
    """
    Audio source wrapping WAV bytes or a raw PCM ndarray.

    Exposes only the subset of BaseMicrophone attributes/methods that ASR uses,
    so it can be used uniformly. ``capture()`` raises ``AudioSourceExhausted``
    when the underlying buffer is drained.
    """

    _DEFAULT_SAMPLING_RATE = 16000
    _DEFAULT_CHANNELS = 1
    _DEFAULT_BUFFER_SIZE = 1024

    def __init__(self, samples: bytes | np.ndarray):
        if isinstance(samples, (bytes, bytearray)):
            with wave.open(io.BytesIO(bytes(samples)), "rb") as wf:
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            # Derive numpy dtype from WAV sample width (signed int, little-endian — WAV convention)
            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            if sample_width not in dtype_map:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")
            self.format = np.dtype(dtype_map[sample_width])
            self._samples = np.frombuffer(frames, dtype=self.format)
        elif isinstance(samples, np.ndarray):
            self.sample_rate = self._DEFAULT_SAMPLING_RATE
            self.channels = self._DEFAULT_CHANNELS
            self.format = samples.dtype
            self._samples = samples
        else:
            raise TypeError(f"Unsupported in-memory audio source type: {type(samples)!r}")

        self.format_is_packed = False
        self.buffer_size = self._DEFAULT_BUFFER_SIZE
        self._started = True  # It's started by default, this is not a real device
        self._cursor = 0

    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def capture(self) -> np.ndarray:
        step = self.buffer_size * self.channels
        if self._cursor >= len(self._samples):
            raise AudioSourceExhausted()

        chunk = self._samples[self._cursor : self._cursor + step]
        self._cursor += step

        return chunk


@dataclass
class SessionInfo:
    session_id: str
    duration: int
    start_time: float
    result_queue: queue.Queue[ASREvent]
    chunk_queue: queue.Queue[bytes | object]  # object is for _END_SENTINEL
    cancelled: threading.Event
    reader_thread: threading.Thread | None = None


_END_SENTINEL = object()  # Sentinel value to signal end of audio stream in the chunk queue


@brick
class AutomaticSpeechRecognition:
    _APP_SERVICE_NAME = "audio-analytics-runner"
    _FLUSH_INTERVAL_SECONDS = 5
    _DEFAULT_VAD = "700"

    def __init__(
        self,
        source: BaseMicrophone | np.ndarray | bytes | None = None,
        language: str | None = None,
    ):
        """
        ASR brick that uses a local audio analytics service to decode audio streams.

        Args:
            source: Audio source for transcription. One of:
                BaseMicrophone: used as-is; the caller owns its
                    lifecycle (ASR never calls start()/stop() on it).
                bytes: treated as a WAV container and wrapped internally.
                np.ndarray: treated as raw PCM samples at 16 kHz mono
                    (dtype inferred) and wrapped internally.
                None: ASR constructs a default Microphone() and owns its
                    lifecycle (started on start(), stopped on stop()).
                Default: None.
            language (str): Language code for the ASR model (e.g. "en" for English).

        Note:
            Only one transcription can be active per instance at a time. For
            concurrent transcriptions on different mics, create multiple ASR
            instances.
        """
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

        if source is None:
            self._source = Microphone()
            self._owns_source = True
        elif isinstance(source, BaseMicrophone):
            self._source = source
            self._owns_source = False
        elif isinstance(source, (bytes, bytearray, np.ndarray)):
            self._source = InMemoryAudioSource(source)
            self._owns_source = False
        else:
            raise TypeError(f"Unsupported source type: {type(source)!r}")

        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_ready = threading.Event()
        self._stop_worker = threading.Event()

        self._active_session_lock = threading.Lock()
        self._active_session: SessionInfo | None = None

    def start(self):
        """Prepare the ASR for transcription. Starts the owned mic if applicable."""
        logger.debug("Starting ASR and preparing resources...")
        self._stop_worker.clear()
        if self._owns_source:
            self._source.start()

    def stop(self):
        """Stop the ASR and clean up resources. Stops the owned mic if applicable."""
        logger.debug("Stopping ASR and cleaning up resources...")
        self._stop_worker.set()
        self.cancel()
        if self._owns_source:
            self._source.stop()
        logger.debug("Stopped ASR and cleaned up resources.")

    def cancel(self):
        """Cancel the active transcription session, if any."""
        active = self._active_session
        if active is None:
            logger.debug("No active session to cancel")
            return
        logger.debug(f"Cancelling session {active.session_id}")
        active.cancelled.set()

    def transcribe(self, duration: int = 0) -> str:
        """
        Transcribe audio from the configured source and return the final text.

        Args:
            duration (int): Maximum recording time in seconds. ``0`` means unbounded.
                Ignored for finite sources (WAV/ndarray), which are consumed
                to completion regardless. Default: ``0``.

        Returns:
            str: The transcribed text, or an empty string if no speech was detected.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the audio source has not been started.
        """
        last_partial = ""
        final_text = ""

        with self.transcribe_stream(duration=duration) as stream:
            for chunk in stream:
                if chunk.type == "partial_text" and chunk.data.strip():
                    last_partial = chunk.data
                elif chunk.type == "full_text" and chunk.data.strip():
                    final_text += chunk.data

        if final_text.strip():
            return final_text
        if last_partial.strip():
            logger.warning("ASR returned empty full_text, falling back to last partial_text")
            return last_partial
        logger.info("ASR returned no speech / empty transcription")
        return ""

    def transcribe_stream(self, duration: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio from the configured source and stream events.

        Args:
            duration (int): Maximum recording time in seconds. ``0`` means unbounded.
                Ignored for finite sources (WAV/ndarray). Default: ``0``.

        Yields:
            ASREvent: objects representing transcription events.

        Raises:
            ASRBusyError: If this instance already has an active session.
            ASRServiceBusyError: If no more concurrent sessions are available.
            ASRUnavailableError: If the inference service is unreachable or the
                connection drops mid-session.
            RuntimeError: If the audio source has not been started.
        """
        if not self._source.is_started():
            raise RuntimeError("Audio source must be started before transcription.")
        return TranscriptionStream(self._transcribe_stream(duration=duration))

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
            response = requests.post(url, json={"session_id": session_id}, timeout=5)
        except Exception:
            raise
        if response.status_code != 200:
            raise RuntimeError(f"HTTP status {response.status_code}: {response.text}")
        logger.debug(f"Session {session_id} closed successfully")

    def _create_transcription_session(self) -> str:
        sampling_rate = str(self._source.sample_rate)
        channels = str(self._source.channels)
        pcm_format = _dtype_to_pcm_format(self._source.format, getattr(self._source, "format_is_packed", False))

        create_url = f"{self.api_base_url}/transcriptions/create"
        create_data = {
            "model": self.model,
            "stream": True,
            "parameters": json.dumps([
                {"key": "sampling_rate", "value": sampling_rate},
                {"key": "channels", "value": channels},
                {"key": "format", "value": pcm_format},
                {"key": "vad", "value": self._DEFAULT_VAD},
            ]),
        }
        if self.language is not None:
            create_data["language"] = self.language

        try:
            response = requests.post(url=create_url, json=create_data, timeout=5)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ASRUnavailableError(f"Inference service unreachable: {e}") from None

        if response.status_code == 400:
            try:
                err = response.json().get("error", {})
                msg = err.get("message", "")
            except Exception:
                msg = response.text or ""
            if "transcription session is already active" in msg:
                raise ASRServiceBusyError(msg or "Inference server is serving another client")
            raise ASRError(msg or f"Failed to create transcription session: 400")

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
            logger.warning(f"ASR session {session_id} created but not initialized (state={state})")

        return session_id

    def _transcribe_stream(self, duration: int = 0) -> Generator[ASREvent, None, None]:
        if not self._worker_ready.wait(timeout=5):
            raise RuntimeError("Worker loop not initialized. Call start() first.")
        if self._stop_worker.is_set():
            raise RuntimeError("ASR is stopping or stopped")

        if not self._active_session_lock.acquire(blocking=False):
            active_id = self._active_session.session_id if self._active_session else "unknown"
            raise ASRBusyError(
                f"A transcription session (id={active_id}) is already active on this instance. "
                f"Create a separate AutomaticSpeechRecognition instance for concurrent transcriptions."
            )

        session_info: SessionInfo | None = None
        future = None

        try:
            logger.debug(f"Creating transcription session with model={self.model}, language={self.language}")

            session_id = self._create_transcription_session()

            session_info = SessionInfo(
                session_id=session_id,
                duration=duration,
                start_time=time.time(),
                result_queue=queue.Queue(),
                chunk_queue=queue.Queue(maxsize=100),
                cancelled=threading.Event(),
            )
            self._active_session = session_info

            future = asyncio.run_coroutine_threadsafe(
                self._transcription_session_handler(session_info),
                self._worker_loop,
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

        except (TimeoutError, asyncio.TimeoutError):
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

    @brick.execute
    def _asyncio_loop(self):
        """Dedicated thread for the asyncio event loop hosting session coroutines."""
        logger.debug("Asyncio event loop starting")
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)
        self._worker_ready.set()

        async def keep_alive():
            while not self._stop_worker.is_set():
                await asyncio.sleep(0.1)

        try:
            self._worker_loop.run_until_complete(keep_alive())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            pending = asyncio.all_tasks(self._worker_loop)
            for task in pending:
                task.cancel()
            if pending:
                self._worker_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._worker_ready.clear()
            self._worker_loop.close()
            self._worker_loop = None
            logger.debug("Asyncio event loop stopped")

    async def _await_connection_established(self, websocket, label):
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except (asyncio.TimeoutError, ConnectionClosed) as e:
            raise ASRUnavailableError(f"{label} handshake failed: {e}") from None
        msg = json.loads(raw)
        if msg.get("state") != "connection_established":
            raise RuntimeError(f"{label} expected connection_established, got {msg}")

    async def _periodic_flush(self, session_info: SessionInfo) -> None:
        session_id = session_info.session_id
        has_duration = session_info.duration > 0
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                await asyncio.sleep(self._FLUSH_INTERVAL_SECONDS)
                if self._stop_worker.is_set() or session_info.cancelled.is_set():
                    break
                await asyncio.to_thread(self._flush_transcription_session, session_id)
                if has_duration:
                    remaining = session_info.duration - (time.time() - session_info.start_time)
                    if remaining < self._FLUSH_INTERVAL_SECONDS:
                        logger.debug(f"No more flushes for session {session_id}: only {remaining:.1f}s remaining")
                        break
        except asyncio.CancelledError:
            logger.debug(f"Periodic flush cancelled for session {session_id}")
            raise

    def _reader_thread_body(self, session_info: SessionInfo) -> None:
        session_id = session_info.session_id
        start_time = session_info.start_time
        duration = session_info.duration
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
                try:
                    session_info.chunk_queue.put_nowait(chunk.tobytes())
                except queue.Full:
                    logger.warning(f"Send queue full for session {session_id}, dropping chunk")
        finally:
            try:
                session_info.chunk_queue.put_nowait(_END_SENTINEL)
            except queue.Full:
                pass
            logger.debug(f"Reader thread exited for session {session_id}")

    async def _transcription_session_handler(self, session_info: SessionInfo):
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
                        ping_interval=3,
                        ping_timeout=2,
                        close_timeout=1,
                    ) as write_ws,
                    websockets.connect(
                        self.ws_url,
                        ping_interval=3,
                        ping_timeout=2,
                        close_timeout=1,
                    ) as read_ws,
                ):
                    await self._await_connection_established(write_ws, "write_ws")
                    await self._await_connection_established(read_ws, "read_ws")

                    send_task = asyncio.create_task(self._send_pcm_stream(websocket=write_ws, session_info=session_info))
                    receive_task = asyncio.create_task(self._receive_transcription(websocket=read_ws, session_info=session_info))
                    flush_task = asyncio.create_task(self._periodic_flush(session_info))

                    try:
                        while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                            done, _ = await asyncio.wait(
                                {send_task, receive_task},
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
                        session_info.cancelled.set()

                        if flush_task and not flush_task.done():
                            flush_task.cancel()
                        await asyncio.gather(flush_task, return_exceptions=True)

                        # Server protocol: close session BEFORE tearing down WebSockets
                        try:
                            await asyncio.to_thread(self._close_transcription_session, session_id)
                        except Exception as e:
                            logger.error(f"Failed to close session {session_id} during teardown: {e}")

                        for task in (send_task, receive_task):
                            if task and not task.done():
                                task.cancel()
                        await asyncio.gather(send_task, receive_task, return_exceptions=True)
            except OSError as e:
                raise ASRUnavailableError(f"Failed to connect to inference service: {e}") from None

        finally:
            session_info.cancelled.set()
            join_timeout = 2.0
            await asyncio.to_thread(reader.join, join_timeout)
            if reader.is_alive():
                logger.warning(f"Reader thread for session {session_id} did not exit within {join_timeout}s; leaking as daemon")

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
                except asyncio.TimeoutError:
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
                    result_queue.put(ASREvent("partial_text", evt_text))
                    continue
                elif evt_type == "transcript.text.done":
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
