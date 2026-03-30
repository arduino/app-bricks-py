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

from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_peripherals.microphone import BaseMicrophone
from arduino.app_utils import Logger, brick

logger = Logger("LocalASR")


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
    session_closed: threading.Event


@dataclass(frozen=True)
class WAVSessionInfo:
    session_id: str
    wav_audio: bytes
    result_queue: queue.Queue[ASREvent]
    cancelled: threading.Event
    session_closed: threading.Event


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
    def __init__(self, language: str = "en"):
        """ASR implementation that uses a local audio analytics service to decode audio streams.

        Arguments:
            language: The language code for the ASR model (e.g., "en" for English).

        """
        self.max_concurrent_transcriptions = 3

        self.api_host = "localhost"
        infra = load_brick_compose_file(self.__class__) or {}
        for k, _ in infra["services"].items():
            self.api_host = k
            break
        self.api_host = resolve_address(self.api_host)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"
        self.ws_url = f"ws://{self.api_host}:{self.api_port}/stream"

        self.model = "whisper-small"
        self.language = "it"

        self._worker_loop = None
        self._stop_worker = threading.Event()
        self._audio_stream_router = AudioStreamRouter()
        self._session_semaphore = threading.Semaphore(self.max_concurrent_transcriptions)

    def start(self):
        """Prepare the ASR for transcription."""
        self._stop_worker.clear()

    def stop(self):
        """Stop the ASR and clean up resources."""
        self._stop_worker.set()

    def _close_transcription_session(self, session_id: str) -> None:
        url = f"{self.api_base_url}/transcriptions/close"
        payload = {"session_id": session_id}

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug(f"Session {session_id} closed successfully")
            else:
                logger.warning(f"Session close returned status {response.status_code} for session {session_id}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to close session {session_id}: {e}")

    def transcribe_mic(self, mic: BaseMicrophone, duration: int = 0) -> str:
        """
        Transcribe audio data from the microphone and return the transcribed text.
        """
        if not mic.is_started():
            raise RuntimeError("Microphone must be started before transcription. Call mic.start() first.")

        last_partial = ""
        final_text = ""

        with self.transcribe_mic_stream(mic=mic, duration=duration) as stream:
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
        if self._worker_loop is None:
            raise RuntimeError("Worker loop not initialized. Call start() first.")

        if not self._session_semaphore.acquire(blocking=False):
            raise RuntimeError(
                f"Maximum concurrent transcriptions ({self.max_concurrent_transcriptions}) reached. Wait for an existing transcription to complete."
            )

        session_id = None
        cancelled = threading.Event()
        session_closed = threading.Event()
        future = None

        try:
            logger.debug(f"Creating transcription session with model={self.model}, language={self.language}")

            create_url = f"{self.api_base_url}/transcriptions/create"
            create_data = {
                "model": self.model,
                "stream": True,
                "language": self.language,
                "parameters": json.dumps([
                    {"key": "sampling_rate", "value": "16000"},
                    {"key": "channels", "value": "1"},
                    {"key": "format", "value": "pcm_s16le"},
                    {"key": "vad", "value": "700"},
                ]),
            }

            response = requests.post(url=create_url, json=create_data, timeout=3)
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

            result_queue = queue.Queue[ASREvent]()

            if isinstance(audio_source, BaseMicrophone):
                session_info: MicSessionInfo | WAVSessionInfo = MicSessionInfo(
                    session_id=session_id,
                    mic=audio_source,
                    duration=duration,
                    start_time=time.time(),
                    result_queue=result_queue,
                    cancelled=cancelled,
                    session_closed=session_closed,
                )
            elif isinstance(audio_source, bytes):
                session_info = WAVSessionInfo(
                    session_id=session_id,
                    wav_audio=audio_source,
                    result_queue=result_queue,
                    cancelled=cancelled,
                    session_closed=session_closed,
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
            if session_id and not session_closed.is_set():
                self._close_transcription_session(session_id)
                session_closed.set()
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
            self._worker_loop.close()
            self._worker_loop = None
            logger.debug("Asyncio event loop stopped")

    async def _transcription_session_handler(self, session_info: MicSessionInfo | WAVSessionInfo):
        """
        One websocket per transcription session.
        Audio is streamed while results are received on the same websocket.
        When transcript.text.done arrives, close the server-side session and then
        exit the websocket context so the connection is closed too.
        """
        session_id = session_info.session_id
        send_task: asyncio.Task | None = None
        receive_task: asyncio.Task | None = None

        async with websockets.connect(self.ws_url) as websocket:
            logger.debug(f"WebSocket connected for session {session_id}")

            if isinstance(session_info, MicSessionInfo):
                pcm_chunks = self._iter_mic_pcm_chunks(session_info)
            else:
                pcm_chunks = self._iter_wav_pcm_chunks(session_info)

            send_task = asyncio.create_task(self._send_pcm_stream(websocket, session_id, pcm_chunks))
            receive_task = asyncio.create_task(self._receive_transcription(websocket, session_info))

            tasks = {send_task, receive_task}

            try:
                while True:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                    if send_task in done:
                        send_exc = send_task.exception()
                        if send_exc is not None:
                            raise send_exc
                        tasks.discard(send_task)

                    if receive_task in done:
                        receive_exc = receive_task.exception()
                        if receive_exc is not None:
                            raise receive_exc
                        break

                    if not pending and receive_task not in tasks:
                        break

                session_info.cancelled.set()

            finally:
                session_info.cancelled.set()

                if send_task is not None and not send_task.done():
                    send_task.cancel()

                if receive_task is not None and not receive_task.done():
                    receive_task.cancel()

                await asyncio.gather(
                    *(task for task in (send_task, receive_task) if task is not None),
                    return_exceptions=True,
                )

                if not session_info.session_closed.is_set():
                    await asyncio.to_thread(self._close_transcription_session, session_id)
                    session_info.session_closed.set()

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

        logger.info(f"WAV format for session {session_id} - Sample Rate: {sample_rate}, Channels: {num_channels}, Sample Width: {sample_width}")

        chunk_duration = 0.5
        chunk_size = int(chunk_duration * sample_rate * num_channels * sample_width)

        for i in range(0, len(frames), chunk_size):
            if self._stop_worker.is_set() or session_info.cancelled.is_set():
                break
            yield frames[i : i + chunk_size]

    async def _receive_transcription(
        self,
        websocket: websockets.ClientConnection,
        session_info: MicSessionInfo | WAVSessionInfo,
    ) -> None:
        """
        Receive transcription events for one session over its dedicated websocket.
        The session ends only when transcript.text.done arrives, or on error/close.
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

                logger.info(f"Received WebSocket message for session {session_id}. Message: {data}")

                msg_type = data.get("message_type") or data.get("type")

                if msg_type == "transcript.text.delta":
                    result_queue.put(ASREvent("partial_text", data.get("text", "")))

                elif msg_type == "transcript.text.done":
                    result_queue.put(ASREvent("full_text", data.get("text", "")))
                    break

                elif msg_type == "transcript.event":
                    continue

                elif msg_type == "connection_established":
                    continue

                elif msg_type == "connection_close":
                    logger.warning(f"WebSocket connection closed for session {session_id}")
                    break

                elif "error" in data:
                    error_msg = data["error"].get("message", "Unknown error")
                    logger.error(f"Transcription error for {session_id}: {error_msg}")
                    raise RuntimeError(error_msg)

                else:
                    logger.warning(f"Unknown message type received: {msg_type}")
                    raise RuntimeError(f"Unknown message type received: {msg_type}")

        except asyncio.CancelledError:
            logger.debug(f"Receive task cancelled for session {session_id}")
            raise

        except ConnectionClosedOK:
            logger.debug(f"WebSocket closed as expected while receiving transcription for session {session_id}")
            return

        except Exception as e:
            logger.error(f"Error receiving transcription for {session_id}: {e}")
            raise
