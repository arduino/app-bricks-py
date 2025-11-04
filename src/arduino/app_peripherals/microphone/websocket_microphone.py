# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import threading
import queue
import time
import numpy as np
import websockets
import asyncio
from typing import Optional

from arduino.app_utils import Logger

from .base_microphone import BaseMicrophone
from .config import RATE_16K, MONO, FORMAT_S16_LE, BALANCED_CHUNK
from .errors import MicrophoneOpenError

logger = Logger("WebSocketMicrophone")


class WebSocketMicrophone(BaseMicrophone):
    """
    WebSocket Microphone implementation that hosts a WebSocket server.

    This microphone acts as a WebSocket server that receives audio chunks from connected clients.
    Only one client can be connected at a time.

    Clients must send audio data in one of these formats:
    - Binary audio data (raw PCM)
    - Base64 encoded audio
    - JSON messages with audio data
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: int = 10,
        audio_format: str = "binary",
        sample_rate: int = RATE_16K,
        channels: int = MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = BALANCED_CHUNK,
    ):
        """
        Initialize WebSocket microphone server.

        Args:
            host (str): Host address to bind the server to (default: "0.0.0.0")
            port (int): Port to bind the server to (default: 8080)
            timeout (int): Connection timeout in seconds (default: 10)
            audio_format (str): Expected audio format from clients ("binary", "base64", "json") (default: "binary")
            sample_rate (int): Sample rate in Hz (default: 16000)
            channels (int): Number of audio channels (default: 1)
            format (str): Audio format (default: "S16_LE")
            chunk_size (int): Number of frames per chunk (default: 1024)
        """
        super().__init__(sample_rate, channels, format, chunk_size)

        self.host = host
        self.port = port
        self.timeout = timeout
        self.audio_format = audio_format
        self.logger = logger

        # Determine numpy dtype based on format
        self._dtype = self._get_dtype_for_format(format)

        self._audio_queue = queue.Queue(10)
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: Optional[websockets.ServerConnection] = None
        self._client_lock = asyncio.Lock()

    def _get_dtype_for_format(self, format: str) -> np.dtype:
        """Get numpy dtype for audio format."""
        format_map = {
            "S8": np.int8,
            "U8": np.uint8,
            "S16_LE": np.int16,
            "S16_BE": ">i2",
            "U16_LE": np.uint16,
            "U16_BE": ">u2",
            "S32_LE": np.int32,
            "S32_BE": ">i4",
            "U32_LE": np.uint32,
            "U32_BE": ">u4",
            "FLOAT_LE": np.float32,
            "FLOAT_BE": ">f4",
            "FLOAT64_LE": np.float64,
            "FLOAT64_BE": ">f8",
        }
        return format_map.get(format, np.int16)

    def _open_microphone(self) -> None:
        """Start the WebSocket server."""
        # Start server in separate thread with its own event loop
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        start_timeout = 10
        while self._server is None and time.time() - start_time < start_timeout:
            if self._server is not None:
                break
            time.sleep(0.1)

        if self._server is None:
            raise MicrophoneOpenError(f"Failed to start WebSocket server on {self.host}:{self.port}")

    def _start_server_thread(self) -> None:
        """Run WebSocket server in its own thread with event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server())
        except Exception as e:
            logger.error(f"WebSocket server thread error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self._stop_event.clear()

            self._server = await websockets.serve(
                self._ws_handler,
                self.host,
                self.port,
                open_timeout=self.timeout,
                ping_timeout=self.timeout,
                close_timeout=self.timeout,
                ping_interval=20,
            )

            logger.info(f"WebSocket microphone server started on {self.host}:{self.port}")

            await self._stop_event.wait()

        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            raise
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()

    async def _ws_handler(self, conn: websockets.ServerConnection) -> None:
        """Handle a connected WebSocket client. Only one client allowed at a time."""
        client_addr = f"{conn.remote_address[0]}:{conn.remote_address[1]}"

        async with self._client_lock:
            if self._client is not None:
                # Reject the new client
                logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
                try:
                    await conn.send(json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000}))
                    await conn.close(code=1000, reason="Server busy - only one client allowed")
                except Exception as e:
                    logger.warning(f"Error sending rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        logger.info(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                await self._send_to_client({
                    "status": "connected",
                    "message": "You are now connected to the microphone server",
                    "audio_format": self.audio_format,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "format": self.format,
                })
            except Exception as e:
                logger.warning(f"Could not send welcome message to {client_addr}: {e}")

            async for message in conn:
                audio_chunk = await self._parse_message(message)
                if audio_chunk is not None:
                    # Drop old chunks until there's room for the new one
                    while True:
                        try:
                            self._audio_queue.put_nowait(audio_chunk)
                            break
                        except queue.Full:
                            try:
                                # Drop oldest chunk and try again
                                self._audio_queue.get_nowait()
                            except queue.Empty:
                                continue

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    logger.info(f"Client removed: {client_addr}")

    async def _parse_message(self, message) -> Optional[np.ndarray]:
        """Parse WebSocket message to extract audio chunk."""
        try:
            if self.audio_format == "binary":
                # Direct binary data
                if isinstance(message, bytes):
                    return np.frombuffer(message, dtype=self._dtype)
                else:
                    logger.warning("Expected binary message but got text")
                    return None

            elif self.audio_format == "base64":
                # Base64 encoded audio
                if isinstance(message, str):
                    audio_data = base64.b64decode(message)
                    return np.frombuffer(audio_data, dtype=self._dtype)
                else:
                    logger.warning("Expected text message for base64 but got binary")
                    return None

            elif self.audio_format == "json":
                # JSON with audio data
                if isinstance(message, str):
                    data = json.loads(message)
                    if "audio" in data:
                        audio_b64 = data["audio"]
                        audio_data = base64.b64decode(audio_b64)
                        return np.frombuffer(audio_data, dtype=self._dtype)
                    else:
                        logger.warning("JSON message missing 'audio' field")
                        return None
                else:
                    logger.warning("Expected text message for JSON but got binary")
                    return None

            else:
                logger.error(f"Unknown audio format: {self.audio_format}")
                return None

        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

    async def _send_to_client(self, message: dict) -> None:
        """Send a JSON message to the connected client."""
        if self._client is not None:
            try:
                await self._client.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"Error sending message to client: {e}")

    def _close_microphone(self) -> None:
        """Stop the WebSocket server."""
        if self._loop is not None and self._server is not None:
            try:
                # Signal the server to stop
                asyncio.run_coroutine_threadsafe(self._stop_event.set(), self._loop)

                # Wait for server thread to finish
                if self._server_thread is not None:
                    self._server_thread.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping WebSocket server: {e}")
            finally:
                self._server = None
                self._loop = None
                self._server_thread = None

    def _read_audio(self) -> Optional[np.ndarray]:
        """Read a single audio chunk from the WebSocket microphone."""
        try:
            # Non-blocking get with short timeout
            audio_chunk = self._audio_queue.get(timeout=0.1)
            return audio_chunk
        except queue.Empty:
            return None
