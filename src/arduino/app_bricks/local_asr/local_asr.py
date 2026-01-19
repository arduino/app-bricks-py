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
import websockets

from arduino.app_peripherals.microphone import BaseMicrophone, ALSAMicrophone
from arduino.app_utils import brick, Logger, HttpClient

logger = Logger("LocalASR")


@dataclass(frozen=True)
class ASREvent:
    type: Literal["partial_text", "full_text"]
    data: str


@dataclass(frozen=True)
class MicSessionInfo:
    session_id: str
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


class AudioSubscribers:
    """Manages pub-sub for audio distribution to multiple consumers."""
    
    def __init__(self):
        self._subscribers = {}  # subscriber_id -> queue
        self._lock = threading.Lock()
    
    def subscribe(self, subscriber_id: str, audio_queue: queue.Queue):
        """Register a subscriber to receive audio chunks."""
        with self._lock:
            self._subscribers[subscriber_id] = audio_queue
            logger.debug(f"Subscriber {subscriber_id} registered")
    
    def unsubscribe(self, subscriber_id: str):
        """Unregister a subscriber."""
        with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                logger.debug(f"Subscriber {subscriber_id} unregistered")
    
    def publish(self, audio_chunk):
        """Publish audio chunk to all subscribers."""
        with self._lock:
            for subscriber_id, audio_queue in list(self._subscribers.items()):
                try:
                    audio_queue.put_nowait(audio_chunk)
                except queue.Full:
                    logger.warning(f"Audio queue full for subscriber {subscriber_id}, dropping chunk")
    
    def has_subscribers(self) -> bool:
        """Check if there are any active subscribers."""
        with self._lock:
            return len(self._subscribers) > 0


@brick
class LocalASR:
    def __init__(
        self,
        mic: BaseMicrophone | None = None,
        api_host: str = "localhost",
        api_port: int = 8085,
        model: str = "whisper-small",
        language: str = "en"
    ):
        if mic is None:
            mic = ALSAMicrophone()
        self.mic: BaseMicrophone = mic

        self.max_concurrent_transcriptions = 3
        
        # API configuration
        self.api_host = api_host
        self.api_port = api_port
        self.api_base_url = f"http://{api_host}:{api_port}/audio-analytics/v1/api"
        self.ws_url = f"ws://{api_host}:{api_port}/stream"
        
        # ASR configuration
        self.model = model
        self.language = language
        
        # State management
        self._transcribing = False
        self._state_lock = threading.Lock()
        
        # HTTP client for requests
        self._http_client = HttpClient()
        
        # Worker thread management
        self._worker_loop = None
        self._stop_worker = threading.Event()
        
        # Audio distribution
        self._audio_subscribers = AudioSubscribers()
        
        # Session management - only for limiting concurrency
        self._session_semaphore = threading.Semaphore(self.max_concurrent_transcriptions)
    
    def start(self):
        """
        Prepare the ASR for transcription.

        This starts the microphone device too.
        """
        self.mic.start()
        
        self._stop_worker.clear()
    
    def stop(self):
        """
        Stop the ASR.

        This stops the microphone device too.
        """
        self._stop_worker.set()
        
        self.mic.stop()

    def transcribe_mic(self, duration: int = 0) -> str:
        """
        Transcribe audio data from the microphone and return the transcribed text.

        Args:
            duration (int): Duration in seconds to record audio. If 0, records until silence.

        Returns:
            str: The transcribed text.
        
        Raises:
            RuntimeError: If transcription fails or other errors occur.
        """
        transcription = ""
        with self.transcribe_mic_stream(duration=duration) as stream:
            for chunk in stream:
                transcription = chunk.data
        
        if transcription:
            return transcription
        
        raise RuntimeError("Transcription did not return any text")

    def transcribe_mic_stream(self, duration: int = 0) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio data from the microphone and stream the results as soon
        as they are available.

        Partial results are yielded as they arrive, and the final text is yielded
        when the transcription is complete. Partial chunks are of temporary nature
        and may be updated by the final full text.

        Args:
            duration (int): Duration in seconds to record audio. If 0, records until silence.

        Returns:
            TranscriptionStream[ASREvent]: iterable context manager emitting ASREvent items.

        Raises:
            RuntimeError: If transcription fails or other errors occur.
        """
        return TranscriptionStream(self._transcribe_stream(duration))

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

    def _transcribe_stream(self, duration: int = 0, audio_source: bytes | None = None) -> Generator[ASREvent, None, None]:
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
            logger.info(f"Creating transcription session with model={self.model}, language={self.language}")
            # Create transcription session
            url = f"{self.api_base_url}/transcriptions/create"
            data = {
                "model": self.model,
                "stream": True,
                "language": self.language
            }
            response = self._http_client.request_with_retry(
                url=url,
                method="POST",
                json=data,
                timeout=3
            )
            
            if response is None:
                raise RuntimeError("Failed to create transcription session")
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
            if audio_source is None:
                session_info = MicSessionInfo(
                    session_id=session_id,
                    duration=duration,
                    start_time=time.time(),
                    result_queue=result_queue,
                    cancelled=cancelled,
                )
            else:
                session_info = WAVSessionInfo(
                    session_id=session_id,
                    wav_audio=audio_source,
                    result_queue=result_queue,
                    cancelled=cancelled,
                )
            
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
            
        except TimeoutError:
            raise

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        finally:
            self._session_semaphore.release()

    @brick.execute
    def _audio_loop(self):
        """
        Dedicated thread for reading audio from the microphone and publishing
        to subscribers.
        """
        logger.info("Audio reader thread starting")
        
        try:
            while not self._stop_worker.is_set():
                # Only capture audio if there are active subscribers
                if self._audio_subscribers.has_subscribers():
                    audio_chunk = self.mic.capture()
                    self._audio_subscribers.publish(audio_chunk)
                else:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Audio reader thread error: {e}")
            raise
        finally:
            logger.info("Audio reader thread stopped")

    @brick.execute
    def _asyncio_loop(self):
        """
        Dedicated thread for running the asyncio event loop.
        Manages transcription sessions posted via run_coroutine_threadsafe.
        """
        logger.info("Asyncio event loop starting")
        
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
            logger.info("Asyncio event loop stopped")
    
    async def _transcription_session_handler(self, session_info: MicSessionInfo | WAVSessionInfo):
        """
        Manages WebSocket connection, sends audio chunks, and receives
        transcription results.
        """
        session_id = session_info.session_id
        
        async with websockets.connect(self.ws_url) as websocket:
            logger.info(f"WebSocket connected for session {session_id}")
            
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
        duration = session_info.duration
        start_time = session_info.start_time
        
        # Create audio queue and subscribe
        audio_queue = queue.Queue(maxsize=100)
        self._audio_subscribers.subscribe(session_id, audio_queue)
        
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
            self._audio_subscribers.unsubscribe(session_id)
            logger.debug(f"Session {session_id} unsubscribed from audio")
    
    async def _send_wav_audio(self, websocket: websockets.ClientConnection, session_info: WAVSessionInfo):
        """Sends audio chunks from WAV numpy array to WebSocket"""
        import wave
        import io

        session_id = session_info.session_id
        wav_audio = session_info.wav_audio

        with wave.open(io.BytesIO(wav_audio), 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        # Calculate chunk size for ~100ms of audio
        chunk_duration = 0.1
        chunk_size = int(chunk_duration * sample_rate * num_channels * sample_width)

        try:
            for i in range(0, len(frames), chunk_size):
                iteration_start = time.perf_counter()

                if self._stop_worker.is_set() or session_info.cancelled.is_set():
                    break
                    
                audio_chunk = frames[i:i + chunk_size]
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": audio_base64
                }
                
                await websocket.send(json.dumps(message))
                
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
