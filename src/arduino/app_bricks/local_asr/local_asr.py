# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import time
import threading
from typing import ContextManager, Literal, TypeVar, Generic
from collections.abc import Generator, Iterator
from dataclasses import dataclass

import numpy as np
import requests
import websockets

from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_peripherals.microphone import BaseMicrophone
from arduino.app_utils import brick, Logger

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
    """Routes audio streams from multiple microphones to grouped subscribers."""
    
    def __init__(self):
        self._subscribers = {}  # mic_id -> {subscriber_id -> queue}
        self._threads = {}  # mic_id -> thread
        self._lock = threading.Lock()
    
    def subscribe(self, mic: BaseMicrophone, subscriber_id: str, audio_queue: queue.Queue):
        """Register a subscriber to receive audio chunks from a specific microphone."""
        mic_id = id(mic)
        with self._lock:
            if mic_id not in self._subscribers:
                self._subscribers[mic_id] = {}
            self._subscribers[mic_id][subscriber_id] = audio_queue
            logger.debug(f"Subscriber {subscriber_id} registered for mic {mic_id}")
    
    def unsubscribe(self, mic: BaseMicrophone, subscriber_id: str):
        """Unregister a subscriber from a specific microphone."""
        mic_id = id(mic)
        with self._lock:
            if mic_id in self._subscribers and subscriber_id in self._subscribers[mic_id]:
                del self._subscribers[mic_id][subscriber_id]
                logger.debug(f"Subscriber {subscriber_id} unregistered from mic {mic_id}")
                # Clean up empty mic groups
                if not self._subscribers[mic_id]:
                    del self._subscribers[mic_id]
    
    def publish(self, mic: BaseMicrophone, audio_chunk):
        """Publish audio chunk to all subscribers of a specific microphone."""
        mic_id = id(mic)
        with self._lock:
            if mic_id not in self._subscribers:
                return
            for subscriber_id, audio_queue in list(self._subscribers[mic_id].items()):
                try:
                    audio_queue.put_nowait(audio_chunk)
                except queue.Full:
                    logger.warning(f"Audio queue full for subscriber {subscriber_id}, dropping chunk")
    
    def has_subscribers(self, mic: BaseMicrophone) -> bool:
        """Check if there are any active subscribers for a specific microphone."""
        mic_id = id(mic)
        with self._lock:
            return mic_id in self._subscribers and len(self._subscribers[mic_id]) > 0
    
    def register_thread(self, mic: BaseMicrophone, thread: threading.Thread):
        """Register a reader thread for a microphone."""
        mic_id = id(mic)
        with self._lock:
            self._threads[mic_id] = thread
    
    def unregister_thread(self, mic: BaseMicrophone):
        """Unregister a reader thread for a microphone."""
        mic_id = id(mic)
        with self._lock:
            if mic_id in self._threads:
                del self._threads[mic_id]
    
    def get_thread(self, mic: BaseMicrophone) -> threading.Thread | None:
        """Get the reader thread for a microphone."""
        mic_id = id(mic)
        with self._lock:
            return self._threads.get(mic_id)


@brick
class LocalASR:
    def __init__(self):
        self.max_concurrent_transcriptions = 3
        
        # API configuration
        self.api_host = "localhost"
        infra = load_brick_compose_file(self.__class__) or {}
        for k, _ in infra["services"].items():
            self.api_host = k
            break  # Only one service is expected
        self.api_host = resolve_address(self.api_host)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")
        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"
        self.ws_url = f"ws://{self.api_host}:{self.api_port}/stream"
        
        # ASR configuration
        self.model = "whisper-small"
        self.language = "en"
        
        # Worker thread management
        self._worker_loop = None
        self._stop_worker = threading.Event()
        
        # Audio distribution
        self._audio_stream_router = AudioStreamRouter()
        
        # Limit concurrency
        self._session_semaphore = threading.Semaphore(self.max_concurrent_transcriptions)
    
    def start(self):
        """
        Prepare the ASR for transcription.
        """
        self._stop_worker.clear()
    
    def stop(self):
        """
        Stop the ASR and clean up resources.
        """
        self._stop_worker.set()

    def transcribe_mic(self, mic: BaseMicrophone, duration: int = 0) -> str:
        """
        Transcribe audio data from the microphone and return the transcribed text.

        Args:
            mic (BaseMicrophone): The microphone instance to capture audio from.
            duration (int): Duration in seconds to record audio. If 0, records until silence.

        Returns:
            str: The transcribed text.
        
        Raises:
            RuntimeError: If transcription fails, microphone not started, or other errors occur.
        """
        if not mic.is_started():
            raise RuntimeError("Microphone must be started before transcription. Call mic.start() first.")
        
        transcription = ""
        with self.transcribe_mic_stream(mic=mic, duration=duration) as stream:
            for chunk in stream:
                transcription = chunk.data
        
        if transcription:
            return transcription
        
        raise RuntimeError("Transcription did not return any text")

    def transcribe_mic_stream(self, mic: BaseMicrophone, duration: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio data from the microphone and stream the results as soon
        as they are available.

        Partial results are yielded as they arrive, and the final text is yielded
        when the transcription is complete. Partial chunks are of temporary nature
        and may be updated by the final full text.

        Args:
            mic (BaseMicrophone): The microphone instance to capture audio from.
            duration (int): Duration in seconds to record audio. If 0, records until silence.

        Returns:
            TranscriptionStream[ASREvent]: iterable context manager emitting ASREvent items.

        Raises:
            RuntimeError: If transcription fails, microphone not started, or other errors occur.
        """
        if not mic.is_started():
            raise RuntimeError("Microphone must be started before transcription. Call mic.start() first.")
        
        return TranscriptionStream(self._transcribe_stream(duration=duration, audio_source=mic))

    def transcribe_wav(self, wav_data: np.ndarray | bytes) -> str:
        """
        Transcribe audio from WAV data and return the transcribed text.

        Args:
            wav_data (np.ndarray | bytes): WAV file data (including header) as numpy array or bytes.

        Returns:
            str: The transcribed text.
        
        Raises:
            RuntimeError: If transcription fails or other errors occur.
        """
        transcription = ""
        with self.transcribe_wav_stream(wav_data) as stream:
            for chunk in stream:
                transcription = chunk.data
        
        if transcription:
            return transcription
        
        raise RuntimeError("Transcription did not return any text")

    def transcribe_wav_stream(self, wav_data: np.ndarray | bytes) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio from WAV data and stream the results.

        Args:
            wav_data (np.ndarray | bytes): WAV file data (including header) as numpy array or bytes.

        Returns:
            TranscriptionStream[ASREvent]: iterable context manager emitting ASREvent items.

        Raises:
            RuntimeError: If transcription fails or other errors occur.
        """
        data = wav_data.tobytes() if isinstance(wav_data, np.ndarray) else wav_data
        return TranscriptionStream(self._transcribe_stream(audio_source=data))

    def _transcribe_stream(self, duration: int = 0, audio_source: BaseMicrophone | bytes | None = None) -> Generator[ASREvent, None, None]:
        if self._worker_loop is None:
            raise RuntimeError("Worker loop not initialized. Call start() first.")
        if not self._session_semaphore.acquire(blocking=False):
            raise RuntimeError(
                f"Maximum concurrent transcriptions ({self.max_concurrent_transcriptions}) reached. "
                "Wait for an existing transcription to complete."
            )

        session_id = None
        cancelled = threading.Event()
        future = None

        try:
            logger.debug(f"Creating transcription session with model={self.model}, language={self.language}")
            # Create transcription session
            url = f"{self.api_base_url}/transcriptions/create"
            data = {
                "model": self.model,
                "stream": True,
                "language": self.language
            }
            response = requests.post(
                url=url,
                json=data,
                timeout=3
            )
            if response.status_code != 200:
                error_msg = f"Failed to create transcription session: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)
            result = response.json()
            session_id = result.get("session_id")
            if not session_id:
                raise RuntimeError("No session ID returned from transcription API")
            
            result_queue = queue.Queue[ASREvent]()
            
            # Create session info
            if isinstance(audio_source, BaseMicrophone):
                session_info = MicSessionInfo(
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
                self._worker_loop
            )
            
            # Yield results until the session is done
            while not future.done():
                try:
                    yield result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            
            # Drain queue
            while True:
                try:
                    yield result_queue.get_nowait()
                except queue.Empty:
                    break
            
            future.result()  # Will raise if an exception occurred
            
        except GeneratorExit:
            # User broke out of the generator loop, signal cancellation
            logger.debug(f"Transcription interrupted by user for session {session_id}")
            cancelled.set()
            if future and not future.done():
                try:
                    future.result(timeout=2)
                except Exception:
                    pass  # Ignore any errors during cleanup
            raise
            
        except (TimeoutError, asyncio.TimeoutError):
            raise

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        finally:
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
            # Clean up the event loop
            pending = asyncio.all_tasks(self._worker_loop)
            for task in pending:
                task.cancel()
            if pending:
                self._worker_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._worker_loop.close()
            self._worker_loop = None
            logger.debug("Asyncio event loop stopped")
    
    async def _transcription_session_handler(self, session_info: MicSessionInfo | WAVSessionInfo):
        """
        Manages WebSocket connection, sends audio chunks, and receives
        transcription results.
        """
        session_id = session_info.session_id
        
        async with websockets.connect(self.ws_url) as websocket:
            logger.debug(f"WebSocket connected for session {session_id}")
            
            # Start audio sending and transcription receiving tasks
            if isinstance(session_info, MicSessionInfo):
                send_task = asyncio.create_task(
                    self._send_mic_audio(websocket, session_info)
                )
            else:
                send_task = asyncio.create_task(
                    self._send_wav_audio(websocket, session_info)
                )
            receive_task = asyncio.create_task(
                self._receive_transcription(websocket, session_info)
            )
            
            await asyncio.gather(send_task, receive_task)
    
    async def _send_mic_audio(self, websocket: websockets.ClientConnection, session_info: MicSessionInfo):
        """Sends audio chunks from mic queue to WebSocket"""
        session_id = session_info.session_id
        mic = session_info.mic
        duration = session_info.duration
        start_time = session_info.start_time
        
        # Start reader thread for this mic if not already running
        existing_thread = self._audio_stream_router.get_thread(mic)
        if existing_thread is None or not existing_thread.is_alive():
            reader_thread = threading.Thread(
                target=self._mic_reader_loop,
                args=(mic,),
                daemon=True,
                name=f"AudioReader-{id(mic)}"
            )
            self._audio_stream_router.register_thread(mic, reader_thread)
            reader_thread.start()
        
        # Create audio queue and subscribe
        audio_queue = queue.Queue(maxsize=100)
        self._audio_stream_router.subscribe(mic, session_id, audio_queue)
        
        try:
            while not self._stop_worker.is_set() and not session_info.cancelled.is_set():
                # Check duration limit (if duration > 0)
                if duration > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.debug(f"Session {session_id} duration limit reached: {duration}s")
                        break
                
                try:
                    audio_chunk = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, audio_queue.get, True, 0.1),
                        timeout=0.1
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    continue
                
                audio_bytes = audio_chunk.tobytes()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": audio_base64
                }
                
                await websocket.send(json.dumps(message))
            
        except asyncio.CancelledError:
            logger.debug(f"Audio sending cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error sending audio chunks for session {session_id}: {e}")
            raise
        finally:
            # Unsubscribe from audio as soon as we're done sending
            self._audio_stream_router.unsubscribe(mic, session_id)
            logger.debug(f"Session {session_id} unsubscribed from audio")

    def _mic_reader_loop(self, mic: BaseMicrophone):
        """
        Reader thread function for capturing audio from a specific microphone
        and publishing to its subscribers.
        """
        mic_id = id(mic)
        logger.debug(f"Audio reader thread starting for mic {mic_id}")
        
        try:
            while not self._stop_worker.is_set():
                # Only capture audio if there are active subscribers for this mic
                if self._audio_stream_router.has_subscribers(mic):
                    audio_chunk = mic.capture()
                    self._audio_stream_router.publish(mic, audio_chunk)
                else:
                    # No subscribers left, exit the thread
                    logger.debug(f"No more subscribers for mic {mic_id}, stopping reader thread")
                    break
                    
        except Exception as e:
            logger.error(f"Audio reader thread error for mic {mic_id}: {e}")
            raise
        finally:
            self._audio_stream_router.unregister_thread(mic)
            logger.debug(f"Audio reader thread stopped for mic {mic_id}")
    
    async def _send_wav_audio(self, websocket: websockets.ClientConnection, session_info: WAVSessionInfo):
        """Sends audio chunks from WAV numpy array to WebSocket"""
        import wave
        import io

        session_id = session_info.session_id
        wav_audio = session_info.wav_audio

        try:
            # Parse WAV file to get format information and calculate header size
            with wave.open(io.BytesIO(wav_audio), 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                num_frames = wf.getnframes()
            
            frames_size = num_frames * num_channels * sample_width
            header_size = len(wav_audio) - frames_size
            
            header = wav_audio[:header_size]
            frames = wav_audio[header_size:]

            header_base64 = base64.b64encode(header).decode('utf-8')
            header_message = {
                "message_type": "transcriptions_session_audio",
                "message_source": "audio_analytics_api",
                "session_id": session_id,
                "type": "input_audio",
                "data": header_base64
            }
            await websocket.send(json.dumps(header_message))
        
            # Calculate chunk size for ~100ms of audio
            chunk_duration = 0.1
            chunk_size = int(chunk_duration * sample_rate * num_channels * sample_width)

            for i in range(0, len(frames), chunk_size):
                iteration_start = time.perf_counter()

                if self._stop_worker.is_set() or session_info.cancelled.is_set():
                    break
                    
                audio_chunk = frames[i:i + chunk_size]
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                chunk_message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": audio_base64
                }
                await websocket.send(json.dumps(chunk_message))
                
                # Account for processing time to maintain real-time simulation
                elapsed = time.perf_counter() - iteration_start
                sleep_time = max(0, chunk_duration - elapsed)
                await asyncio.sleep(sleep_time)
            
        except asyncio.CancelledError:
            logger.debug(f"Array audio sending cancelled for session {session_id}")
            raise

        except Exception as e:
            logger.error(f"Error sending array audio for session {session_id}: {e}")
            raise
    
    async def _receive_transcription(self, websocket: websockets.ClientConnection, session_info: MicSessionInfo | WAVSessionInfo):
        """Receives transcriptions and puts them in the result queue"""
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
                
                # Handle different message types
                msg_type = data.get("type")
                
                if msg_type == "transcript.text.delta":
                    # Partial transcription result
                    text = data.get("text", "")
                    result_queue.put(ASREvent("partial_text", text))
                
                elif msg_type == "transcript.text.done":
                    # Final transcription result
                    text = data.get("text", "")
                    result_queue.put(ASREvent("full_text", text))
                    break

                elif msg_type == "connection_established":
                    continue

                elif "error" in data:
                    # Error message from server
                    error_msg = data["error"].get("message", "Unknown error")
                    logger.error(f"Transcription error for {session_id}: {error_msg}")
                    raise RuntimeError(error_msg)

                else:
                    logger.warning(f"Unknown message type received: {msg_type}")
                    raise RuntimeError(f"Unknown message type received: {msg_type}")
        
        except asyncio.CancelledError:
            logger.debug(f"Receive task cancelled for session {session_id}")
            raise

        except Exception as e:
            logger.error(f"Error receiving transcription for {session_id}: {e}")
            raise
