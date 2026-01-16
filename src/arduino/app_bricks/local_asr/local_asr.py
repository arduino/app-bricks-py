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

import websockets

from arduino.app_peripherals.microphone import BaseMicrophone, ALSAMicrophone
from arduino.app_utils import brick, Logger, HttpClient

logger = Logger("LocalASR")


@dataclass(frozen=True)
class ASREvent:
    type: Literal["partial_text", "full_text"]
    data: str


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    duration: int
    timeout: int
    result_queue: queue.Queue[ASREvent | Exception]


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
        self._worker_thread = None
        self._worker_loop = None
        self._stop_worker = threading.Event()
        self._new_session_queue = queue.Queue()  # Queue for posting new sessions
        
        # Session management - only for limiting concurrency
        self._session_semaphore = threading.Semaphore(self.max_concurrent_transcriptions)
    
    def start(self):
        """
        Prepare the ASR for transcription.

        This starts the microphone device too.
        """
        self.mic.start()
        
        # Start worker thread if not already running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_worker.clear()
            self._worker_thread = threading.Thread(
                target=self._work_dispatcher,
                name="LocalASR-Worker",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Worker thread started")
    
    def stop(self):
        """
        Stop the ASR.

        This stops the microphone device too.
        """
        # Signal worker thread to stop
        self._stop_worker.set()
        
        # Wake up worker by putting sentinel in queue
        try:
            self._new_session_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for worker thread to finish
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")
        
        self.mic.stop()

    def transcribe(self, duration: int = 0, timeout: int = 30) -> str:
        """
        Transcribe audio data from the microphone and return the transcribed text.

        Args:
            duration (int): Duration in seconds to record audio. If 0, records until silence.
            timeout (int): Maximum time in seconds to wait for a full transcription.

        Returns:
            str: The transcribed text.
        
        Raises:
            TimeoutError: If transcription does not complete within the specified timeout.
            RuntimeError: If already transcribing or other errors occur.
        """
        transcription = ""
        with self.transcribe_stream(duration=duration, timeout=timeout) as stream:
            for chunk in stream:
                transcription = chunk.data
        
        if transcription:
            return transcription
        
        raise RuntimeError("Transcription did not return any text")

    def transcribe_stream(self, duration: int = 0, timeout: int = 30) -> TranscriptionStream[ASREvent]:
        """
        Transcribe audio data from the microphone and stream the results as soon
        as they are available.

        Partial results are yielded as they arrive, and the final text is yielded
        when the transcription is complete. Partial chunks are of temporary nature
        and may be updated by the final full text.

        Args:
            duration (int): Duration in seconds to record audio. If 0, records until silence.
            timeout (int): Maximum time in seconds to wait for a full transcription.

        Returns:
            TranscriptionStream[ASREvent]: iterable context manager emitting ASREvent items.

        Raises:
            TimeoutError: If transcription does not complete within the specified timeout.
            RuntimeError: If already transcribing or other errors occur.
        """
        return TranscriptionStream(self._transcribe_stream(duration=duration, timeout=timeout))

    def _transcribe_stream(self, duration: int = 0, timeout: int = 30) -> Generator[ASREvent, None, None]:
        # Acquire semaphore to limit concurrent transcriptions
        if not self._session_semaphore.acquire(blocking=False):
            raise RuntimeError(
                f"Maximum concurrent transcriptions ({self.max_concurrent_transcriptions}) reached. "
                "Wait for an existing transcription to complete."
            )
        
        try:
            # Create transcription session
            url = f"{self.api_base_url}/transcriptions/create"
            data = {
                "model": self.model,
                "stream": True,
                "language": self.language
            }
            
            logger.info(f"Creating streaming transcription session with model={self.model}, language={self.language}")
            response = self._http_client.request_with_retry(
                url=url,
                method="POST",
                json=data,
                timeout=timeout
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
            
            # Create result queue for this session
            result_queue = queue.Queue[ASREvent | Exception]()
            
            # Create session info and post to worker queue
            session_info = SessionInfo(
                session_id=session_id,
                duration=duration,
                timeout=timeout,
                result_queue=result_queue
            )
            self._new_session_queue.put(session_info)
            
            # Yield results from the queue
            while True:
                try:
                    event = result_queue.get(timeout=1.0)
                    if isinstance(event, Exception):
                        raise event

                    yield event
                    if event.type == "full_text":
                        # Final text received, end of transcription
                        break
                    
                except queue.Empty:
                    # Timeout waiting for results, keep trying
                    continue
            
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            # Release semaphore
            self._session_semaphore.release()

    @brick.execute
    def _work_dispatcher(self):
        """
        Waits for the transciption process to be requested and dispatches work to both the
        consumer that reads from the microphone and the producers that send data to the ASR
        service via sessions. Can dispatch work for up to max_concurrent_transcriptions
        simultaneous transcriptions. The microphone consumer is always one (if a transcription
        is requested), while the producers can be up to max_concurrent_transcriptions,
        implementing a pub-sub pattern.
        """
        # Create a new event loop for this thread
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)
        
        logger.info("Work dispatcher started")
        
        try:
            # Run the main dispatcher loop
            self._worker_loop.run_until_complete(self._async_work_dispatcher())
        except Exception as e:
            logger.error(f"Work dispatcher error: {e}")
        finally:
            # Clean up the event loop
            self._worker_loop.close()
            logger.info("Work dispatcher stopped")
    
    async def _async_work_dispatcher(self):
        """
        Async main loop for the work dispatcher.
        Manages microphone reading and distributes audio to active sessions.
        """
        # Create audio subscribers registry
        audio_subscribers = AudioSubscribers()
        
        audio_reader_task = None
        
        try:
            while not self._stop_worker.is_set():
                # Check for new sessions from the queue (non-blocking in async)
                try:
                    # Run blocking queue.get in executor to not block event loop
                    loop = asyncio.get_event_loop()
                    session_info = await loop.run_in_executor(None, self._new_session_queue.get, True, 0.1)
                    
                    # Check for sentinel value (stop signal)
                    if session_info is None:
                        logger.info("Received stop signal")
                        break
                    
                    logger.info(f"New session received: {session_info.session_id}")
                    
                    # Start session producer task (loop will track it)
                    asyncio.create_task(
                        self._session_producer(session_info, audio_subscribers)
                    )
                    logger.info(f"Session producer task started for {session_info.session_id}")
                    
                except queue.Empty:
                    pass  # No new session, continue
                
                # Start audio reader if needed and not running
                if audio_subscribers.has_subscribers():
                    if audio_reader_task is None or audio_reader_task.done():
                        audio_reader_task = asyncio.create_task(
                            self._audio_reader(audio_subscribers)
                        )
                        logger.info("Audio reader task started")
                else:
                    # No subscribers, stop audio reader
                    if audio_reader_task is not None and not audio_reader_task.done():
                        audio_reader_task.cancel()
                        try:
                            await audio_reader_task
                        except asyncio.CancelledError:
                            pass
                        audio_reader_task = None
                        logger.info("Audio reader task stopped (no subscribers)")
        
        finally:
            # Cancel all tasks in the loop except the current one (this dispatcher)
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
            
            for task in tasks:
                task.cancel()
            
            # Wait for all tasks to finish
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _audio_reader(self, audio_subscribers: AudioSubscribers):
        """
        Reads audio from the microphone and publishes to all subscribers.
        Runs in the worker thread's event loop.
        
        Args:
            audio_subscribers: The subscribers registry to publish to
        """
        logger.info("Audio reader starting")
        
        try:
            # Run mic.stream() in a thread pool executor to avoid blocking
            for audio_chunk in self.mic.stream():
                if self._stop_worker.is_set():
                    break
                
                # Check if there are still subscribers
                if not audio_subscribers.has_subscribers():
                    break
                
                # Publish to all subscribers
                audio_subscribers.publish(audio_chunk)
                
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
        
        except Exception as e:
            logger.error(f"Audio reader error: {e}")
            raise
        finally:
            logger.info("Audio reader stopped")
    
    async def _session_producer(self, session_info: SessionInfo, audio_subscribers: AudioSubscribers):
        """
        Producer task that manages audio subscription and transcription for a session.
        
        Args:
            session_info: Information about the session
            audio_subscribers: The subscribers registry to subscribe to
        """
        session_id = session_info.session_id
        result_queue = session_info.result_queue
        
        logger.info(f"Session producer starting for {session_id}")
        
        # Create audio queue and subscribe
        audio_queue = queue.Queue(maxsize=100)
        audio_subscribers.subscribe(session_id, audio_queue)
        
        try:
            # Start transcription task (manages WebSocket and audio sending)
            await self._receive_transcription_results(session_info, audio_queue)
            
        except Exception as e:
            logger.error(f"Session {session_id} producer error: {e}")
            # Put error in result queue
            result_queue.put(RuntimeError(str(e)))
        finally:
            # Unsubscribe from audio
            audio_subscribers.unsubscribe(session_id)
            logger.info(f"Session producer stopped for {session_id}")
    
    async def _receive_transcription_results(self, session_info: SessionInfo, audio_queue: queue.Queue):
        """
        Manages WebSocket connection, sends audio chunks, and receives transcription results.
        
        Args:
            session_info: Information about the session
            audio_queue: Queue to get audio chunks from
        """
        session_id = session_info.session_id
        duration = session_info.duration
        timeout = session_info.timeout
        result_queue = session_info.result_queue
        start_time = time.time()
        
        async with websockets.connect(self.ws_url) as websocket:
            logger.info(f"WebSocket connected for session {session_id}")
            
            # Start audio sending task
            send_task = asyncio.create_task(
                self._send_audio_chunks(websocket, session_id, audio_queue, duration, start_time)
            )
            
            try:
                while not self._stop_worker.is_set():
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        result_queue.put(TimeoutError(f"Transcription did not complete within {timeout} seconds"))
                        break
                    
                    # Receive message with timeout
                    remaining_timeout = timeout - elapsed
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=min(remaining_timeout, 1.0))
                    except asyncio.TimeoutError:
                        continue
                    
                    # Parse message
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
                    
                    elif "error" in data:
                        # Error message
                        error_msg = data["error"].get("message", "Unknown error")
                        logger.error(f"Transcription error for {session_id}: {error_msg}")
                        result_queue.put(RuntimeError(error_msg))
                        break

                    else:
                        logger.warning(f"Unknown message type received: {msg_type}")
                        result_queue.put(RuntimeError(f"Unknown message type received: {msg_type}"))
                        break
            
            except asyncio.CancelledError:
                logger.debug(f"Receive task cancelled for session {session_id}")
                raise
            except Exception as e:
                logger.error(f"Error receiving transcription results for {session_id}: {e}")
                result_queue.put(RuntimeError(str(e)))
            finally:
                # Cancel send task
                if not send_task.done():
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass
    
    async def _send_audio_chunks(self, websocket, session_id: str, audio_queue: queue.Queue, duration: int, start_time: float):
        """
        Sends audio chunks from queue to WebSocket.
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID
            audio_queue: Queue to get audio chunks from
            duration: Duration in seconds to record audio. If 0, records until silence (VAD).
            start_time: The start time of recording
        """
        try:
            while not self._stop_worker.is_set():
                # Check duration limit (if duration > 0)
                if duration > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.debug(f"Session {session_id} duration limit reached: {duration}s")
                        break
                
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, audio_queue.get, True, 0.1),
                        timeout=0.2
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    continue
                
                # Convert audio chunk to bytes
                audio_bytes = audio_chunk.tobytes()
                
                # Encode as base64
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Create message
                message = {
                    "message_type": "transcriptions_session_audio",
                    "message_source": "audio_analytics_api",
                    "session_id": session_id,
                    "type": "input_audio",
                    "data": audio_base64
                }
                
                # Send message
                await websocket.send(json.dumps(message))
            
        except asyncio.CancelledError:
            logger.debug(f"Audio sending cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error sending audio chunks for session {session_id}: {e}")
            raise
