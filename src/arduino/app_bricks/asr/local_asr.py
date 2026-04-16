# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from collections.abc import AsyncGenerator, Generator, Iterator
from dataclasses import dataclass
from typing import ContextManager, Generic, Literal, TypeVar

import numpy as np
import requests
import websockets
from websockets.exceptions import ConnectionClosedOK

from arduino.app_internal.core import resolve_address
from arduino.app_peripherals.microphone import BaseMicrophone
from arduino.app_utils import Logger, brick

logger = Logger("LocalASR")

_DEFAULT_SAMPLING_RATE = 16000
_DEFAULT_CHANNELS = 1
_DEFAULT_PCM_FORMAT = "pcm_s16le"
_DEFAULT_VAD = "700"


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


@dataclass(frozen=True)
class MicSessionInfo:
    session_id: str
    mic: BaseMicrophone
    duration: int
    start_time: float
    result_queue: queue.Queue[ASREvent]
    cancelled: threading.Event


@dataclass(frozen=True)
class WAVSessionInfo:
    session_id: str
    wav_audio: bytes
    result_queue: queue.Queue[ASREvent]
    cancelled: threading.Event


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


class AudioStreamRouter:
    """Routes audio streams from microphones to per-session subscribers."""

    def __init__(self):
        self._subscribers: dict[int, dict[str, queue.Queue]] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    def subscribe(self, mic: BaseMicrophone, subscriber_id: str, audio_queue: queue.Queue) -> None:
        mic_id = id(mic)
        with self._lock:
            self._subscribers.setdefault(mic_id, {})[subscriber_id] = audio_queue
            logger.debug(f"Subscriber {subscriber_id} registered for mic {mic_id}")

    def unsubscribe(self, mic: BaseMicrophone, subscriber_id: str) -> None:
        mic_id = id(mic)
        with self._lock:
            subscribers = self._subscribers.get(mic_id)
            if not subscribers:
                return

            if subscriber_id in subscribers:
                del subscribers[subscriber_id]
                logger.debug(f"Subscriber {subscriber_id} unregistered from mic {mic_id}")

            if not subscribers:
                del self._subscribers[mic_id]

    def publish(self, mic: BaseMicrophone, audio_chunk) -> None:
        mic_id = id(mic)
        with self._lock:
            subscribers = dict(self._subscribers.get(mic_id, {}))

        for subscriber_id, audio_queue in subscribers.items():
            try:
                audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                logger.warning(f"Audio queue full for subscriber {subscriber_id}, dropping chunk")

    def has_subscribers(self, mic: BaseMicrophone) -> bool:
        mic_id = id(mic)
        with self._lock:
            return bool(self._subscribers.get(mic_id))

    def unregister_thread(self, mic: BaseMicrophone) -> None:
        mic_id = id(mic)
        with self._lock:
            self._threads.pop(mic_id, None)

    def ensure_thread(self, mic: BaseMicrophone, thread_factory) -> threading.Thread:
        mic_id = id(mic)
        with self._lock:
            thread = self._threads.get(mic_id)
            if thread is not None and thread.is_alive():
                return thread

            thread = thread_factory()
            self._threads[mic_id] = thread
            thread.start()
            return thread


@brick
class AutomaticSpeechRecognition:
    _APP_SERVICE_NAME = "audio-analytics-runner"
    _FLUSH_INTERVAL_SECONDS = 5

    def __init__(self, language: str | None = None):
        """ASR implementation that uses a local audio analytics service to decode audio streams.

        Arguments:
            language: The language code for the ASR model (e.g., "en" for English).

        """
        self.max_concurrent_transcriptions = 3
        self.api_host = resolve_address(self._APP_SERVICE_NAME)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"
        self.ws_url = f"ws://{self.api_host}:{self.api_port}/stream"

        self.model = "whisper-small"
        self.language = language

        self._worker_loop = None
        self._worker_ready = threading.Event()
        self._stop_worker = threading.Event()
        self._audio_stream_router = AudioStreamRouter()
        self._session_semaphore = threading.Semaphore(self.max_concurrent_transcriptions)
        self._active_sessions: dict[str, threading.Event] = {}
        self._active_sessions_lock = threading.Lock()

    def start(self):
        """Prepare the ASR for transcription."""
        self._stop_worker.clear()

    def stop(self):
        """Stop the ASR and clean up resources."""
        logger.debug("Stopping ASR and cleaning up resources...")
        self._stop_worker.set()

    def cancel(self):
        """Cancel all active transcription sessions."""
        with self._active_sessions_lock:
            sessions = dict(self._active_sessions)

        if not sessions:
            logger.info("No active sessions to cancel")
            return

        logger.info(f"Cancelling {len(sessions)} active session(s): {list(sessions.keys())}")
        for session_id, cancelled_event in sessions.items():
            cancelled_event.set()
            logger.debug(f"Cancelled session {session_id}")

    def _flush_transcription_session(self, session_id: str) -> None:
        logger.debug(f"Flushing transcription session {session_id}")
        url = f"{self.api_base_url}/transcriptions/flush"
        payload = {"session_id": session_id}

        try:
            response = requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Failed to flush session {session_id}: {e}")
            return

        if response.status_code != 200:
            logger.warning(f"Failed to flush session {session_id}: flush returned status {response.status_code}: {response.text}")
            return

        logger.debug(f"Session {session_id} flushed successfully")

    def _close_transcription_session(self, session_id: str) -> None:
        logger.debug(f"Closing transcription session {session_id}")
        url = f"{self.api_base_url}/transcriptions/close"
        payload = {"session_id": session_id}

        try:
            response = requests.post(url, json=payload, timeout=15)
        except Exception as e:
            raise RuntimeError(f"Failed to close session {session_id}: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(f"Failed to close session {session_id}: close returned status {response.status_code}: {response.text}")

        logger.debug(f"Session {session_id} closed successfully")

    def transcribe_mic(self, mic: BaseMicrophone, duration: int = 0) -> str:
        """
        Transcribe audio data from the microphone and return the transcribed text.
        """
        if not mic.is_started():
            raise RuntimeError("Microphone must be started before transcription. Call mic.start() first.")

        final_text = ""

        with self.transcribe_mic_stream(mic=mic, duration=duration) as stream:
            for chunk in stream:
                if chunk.type == "partial_text":
                    continue
                elif chunk.type == "full_text" and chunk.data.strip():
                    final_text += chunk.data

        if final_text.strip():
            return final_text
        else:
            logger.info("ASR returned no speech / empty transcription")
            return ""

    def transcribe_mic_stream(self, mic: BaseMicrophone, duration: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio data from the microphone and stream the results as soon as they are available.
        """
        if not mic.is_started():
            raise RuntimeError("Microphone must be started before transcription. Call mic.start() first.")

        return TranscriptionStream(self._transcribe_stream(duration=duration, audio_source=mic))

    def transcribe_wav(self, wav_data: np.ndarray | bytes) -> str:
        """
        Transcribe audio from WAV data and return the transcribed text.
        """
        last_partial = ""
        final_text = ""

        with self.transcribe_wav_stream(wav_data) as stream:
            for chunk in stream:
                if chunk.type == "partial_text" and chunk.data.strip():
                    last_partial = chunk.data
                elif chunk.type == "full_text":
                    final_text = chunk.data

        if final_text.strip():
            return final_text

        if last_partial.strip():
            logger.warning("ASR returned empty full_text, falling back to last partial_text")
            return last_partial
        else:
            logger.info("ASR returned no speech / empty transcription")
            return ""

    def transcribe_wav_stream(self, wav_data: np.ndarray | bytes) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio from WAV data and stream the results.
        """
        data = wav_data.tobytes() if isinstance(wav_data, np.ndarray) else wav_data
        return TranscriptionStream(self._transcribe_stream(audio_source=data))

    def _transcribe_stream(
        self,
        duration: int = 0,
        audio_source: BaseMicrophone | bytes | None = None,
    ) -> Generator[ASREvent, None, None]:
        if not self._worker_ready.wait(timeout=5):
            raise RuntimeError("Worker loop not initialized. Call start() first.")

        if self._stop_worker.is_set():
            raise RuntimeError("ASR is stopping or stopped")

        if not self._session_semaphore.acquire(blocking=False):
            raise RuntimeError(
                f"Maximum concurrent transcriptions ({self.max_concurrent_transcriptions}) reached. Wait for an existing transcription to complete."
            )

        session_id = None
        cancelled = threading.Event()
        future = None

        try:
            logger.debug(f"Creating transcription session with model={self.model}, language={self.language}")

            if isinstance(audio_source, BaseMicrophone):
                sampling_rate = str(audio_source.sample_rate)
                channels = str(audio_source.channels)
                pcm_format = _dtype_to_pcm_format(audio_source.format, audio_source.format_is_packed)
            else:
                sampling_rate = str(_DEFAULT_SAMPLING_RATE)
                channels = str(_DEFAULT_CHANNELS)
                pcm_format = _DEFAULT_PCM_FORMAT

            create_url = f"{self.api_base_url}/transcriptions/create"
            create_data = {
                "model": self.model,
                "stream": True,
                "parameters": json.dumps([
                    {"key": "sampling_rate", "value": sampling_rate},
                    {"key": "channels", "value": channels},
                    {"key": "format", "value": pcm_format},
                    {"key": "vad", "value": _DEFAULT_VAD},
                ]),
            }
            if self.language is not None:
                create_data["language"] = self.language

            response = requests.post(url=create_url, json=create_data, timeout=5)

            if response.status_code != 200:
                error_msg = f"Failed to create transcription session: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except Exception:
                    pass
                raise RuntimeError(error_msg)

            result = response.json()

            session_id = result.get("session_id")
            if not session_id:
                raise RuntimeError("No session ID returned from transcription API")

            with self._active_sessions_lock:
                self._active_sessions[session_id] = cancelled

            state = result.get("state")
            if state != "asr_initialized":
                logger.warning(f"ASR session {session_id} created but not initialized (state={state})")

            result_queue = queue.Queue[ASREvent]()

            if isinstance(audio_source, BaseMicrophone):
                session_info: MicSessionInfo | WAVSessionInfo = MicSessionInfo(
                    session_id=session_id,
                    mic=audio_source,
                    duration=duration,
                    start_time=time.time(),
                    result_queue=result_queue,
                    cancelled=cancelled,
                )
            elif isinstance(audio_source, bytes):
                session_info = WAVSessionInfo(
                    session_id=session_id,
                    wav_audio=audio_source,
                    result_queue=result_queue,
                    cancelled=cancelled,
                )
            else:
                raise RuntimeError("audio_source must be either a BaseMicrophone or bytes")

            future = asyncio.run_coroutine_threadsafe(
                self._transcription_session_handler(session_info),
                self._worker_loop,
            )

            while not future.done():
                try:
                    yield result_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

            while True:
                try:
                    yield result_queue.get_nowait()
                except queue.Empty:
                    break

            future.result()

        except GeneratorExit:
            logger.debug(f"Transcription interrupted by user for session {session_id}")
            cancelled.set()
            if future and not future.done():
                future.cancel()
                try:
                    future.result(timeout=2)
                except Exception:
                    pass
            raise

        except (TimeoutError, asyncio.TimeoutError):
            raise

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        finally:
            cancelled.set()
            if session_id:
                with self._active_sessions_lock:
                    self._active_sessions.pop(session_id, None)
            self._session_semaphore.release()

    @brick.execute
    def _asyncio_loop(self):
        """
        Dedicated thread for running the asyncio event loop.
        Manages transcription sessions posted via run_coroutine_threadsafe.
        """
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
        msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5.0))
        if msg.get("state") != "connection_established":
            raise RuntimeError(f"{label} expected connection_established, got {msg}")

    async def _periodic_flush(self, session_info: MicSessionInfo | WAVSessionInfo) -> None:
        """Periodically flush the transcription session to force partial results.

        If the session has a finite duration and the remaining time until it ends
        is less than ``_FLUSH_INTERVAL_SECONDS``, the flush is skipped because the
        session will close shortly anyway.
        """
        session_id = session_info.session_id
        has_duration = isinstance(session_info, MicSessionInfo) and session_info.duration > 0
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                await asyncio.sleep(self._FLUSH_INTERVAL_SECONDS)
                if self._stop_worker.is_set() or session_info.cancelled.is_set():
                    break
                await asyncio.to_thread(self._flush_transcription_session, session_id)
                if has_duration:
                    remaining = session_info.duration - (time.time() - session_info.start_time)
                    if remaining < self._FLUSH_INTERVAL_SECONDS:
                        logger.debug(f"No more flushes for session {session_id}: only {remaining:.1f}s remaining (< {self._FLUSH_INTERVAL_SECONDS}s)")
                        break
        except asyncio.CancelledError:
            logger.debug(f"Periodic flush cancelled for session {session_id}")
            raise

    async def _transcription_session_handler(self, session_info: MicSessionInfo | WAVSessionInfo):
        """
        One transcription session uses two WebSocket connections:
        - write_ws: send audio
        - read_ws: receive events

        The session supports multiple utterances (do not stop on first done).
        """

        session_id = session_info.session_id

        async with websockets.connect(self.ws_url) as write_ws, websockets.connect(self.ws_url) as read_ws:
            # Handshake
            await self._await_connection_established(write_ws, "write_ws")
            await self._await_connection_established(read_ws, "read_ws")

            # Source
            if isinstance(session_info, MicSessionInfo):
                pcm_chunks = self._iter_mic_pcm_chunks(session_info)
            else:
                pcm_chunks = self._iter_wav_pcm_chunks(session_info)

            send_task = asyncio.create_task(
                self._send_pcm_stream(
                    websocket=write_ws,
                    session_id=session_id,
                    pcm_chunks=pcm_chunks,
                )
            )

            receive_task = asyncio.create_task(
                self._receive_transcription(
                    websocket=read_ws,
                    session_info=session_info,
                )
            )

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
                if flush_task and not flush_task.done():
                    flush_task.cancel()
                await asyncio.gather(flush_task, return_exceptions=True)

                # Close the session first, then disconnect WebSockets (server protocol requirement)
                await asyncio.to_thread(self._close_transcription_session, session_id)

                session_info.cancelled.set()

                for task in (send_task, receive_task):
                    if task and not task.done():
                        task.cancel()

                await asyncio.gather(send_task, receive_task, return_exceptions=True)

    async def _send_pcm_stream(
        self,
        websocket: websockets.ClientConnection,
        session_id: str,
        pcm_chunks: AsyncGenerator[bytes, None],
    ) -> int:
        chunks_sent = 0
        try:
            async for audio_bytes in pcm_chunks:
                if self._stop_worker.is_set():
                    break

                message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
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

    async def _iter_mic_pcm_chunks(self, session_info: MicSessionInfo) -> AsyncGenerator[bytes, None]:
        session_id = session_info.session_id
        mic = session_info.mic
        duration = session_info.duration
        start_time = session_info.start_time
        audio_queue: queue.Queue = queue.Queue(maxsize=100)

        self._audio_stream_router.subscribe(mic, session_id, audio_queue)

        def make_reader_thread() -> threading.Thread:
            return threading.Thread(
                target=self._mic_reader_loop,
                args=(mic,),
                daemon=True,
                name=f"AudioReader-{id(mic)}",
            )

        self._audio_stream_router.ensure_thread(mic, make_reader_thread)

        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                if duration > 0 and (time.time() - start_time) >= duration:
                    logger.debug(f"Session {session_id} duration limit reached: {duration}s")
                    break

                try:
                    loop = asyncio.get_running_loop()
                    audio_chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, audio_queue.get, True, 0.1),
                        timeout=0.2,
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    continue

                yield audio_chunk.tobytes()

        finally:
            self._audio_stream_router.unsubscribe(mic, session_id)
            logger.debug(f"Session {session_id} mic chunk iterator cleanup completed")

    def _mic_reader_loop(self, mic: BaseMicrophone):
        """
        Single reader thread per microphone.
        It continuously captures audio and fans it out to active subscribers.
        """
        mic_id = id(mic)
        logger.debug(f"Audio reader thread starting for mic {mic_id}")

        try:
            while not self._stop_worker.is_set():
                if not self._audio_stream_router.has_subscribers(mic):
                    logger.debug(f"No more subscribers for mic {mic_id}, stopping reader thread")
                    break

                audio_chunk = mic.capture()

                if self._audio_stream_router.has_subscribers(mic):
                    self._audio_stream_router.publish(mic, audio_chunk)

        except Exception as e:
            logger.error(f"Audio reader thread error for mic {mic_id}: {e}")

        finally:
            self._audio_stream_router.unregister_thread(mic)
            logger.debug(f"Audio reader thread stopped for mic {mic_id}")

    async def _iter_wav_pcm_chunks(self, session_info: WAVSessionInfo) -> AsyncGenerator[bytes, None]:
        import io
        import wave

        session_id = session_info.session_id
        wav_audio = session_info.wav_audio

        with wave.open(io.BytesIO(wav_audio), "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        logger.debug(f"WAV format for session {session_id} - Sample Rate: {sample_rate}, Channels: {num_channels}, Sample Width: {sample_width}")

        chunk_duration = 0.5
        chunk_size = int(chunk_duration * sample_rate * num_channels * sample_width)

        for i in range(0, len(frames), chunk_size):
            if self._stop_worker.is_set() or session_info.cancelled.is_set():
                break
            yield frames[i : i + chunk_size]

    async def _receive_transcription(self, websocket: websockets.ClientConnection, session_info: MicSessionInfo | WAVSessionInfo) -> None:
        """
        Receive transcription events for one session over its dedicated websocket.
        """
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
                    logger.error(f"Transcription error for session {session_id}: {error_msg}")
                    raise RuntimeError(error_msg)

                elif evt_type == "connection_close":
                    logger.warning(f"WebSocket connection closed for session {session_id}")
                    break

                else:
                    logger.warning(f"Unknown message type received: {evt_type}")
                    raise RuntimeError(f"Unknown message type received: {evt_type}")

        except asyncio.CancelledError:
            logger.debug(f"Receive task cancelled for session {session_id}")
            raise

        except ConnectionClosedOK:
            logger.debug(f"WebSocket closed as expected while receiving transcription for session {session_id}")
            return

        except Exception as e:
            logger.error(f"Error receiving transcription for {session_id}: {e}")
            raise
