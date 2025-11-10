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
from typing import Literal, Optional
from concurrent.futures import CancelledError, TimeoutError

from .base_microphone import BaseMicrophone
from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneConfigError, MicrophoneOpenError
from arduino.app_utils import Logger

logger = Logger("WebSocketMicrophone")


class WebSocketMicrophone(BaseMicrophone):
    """
    WebSocket Microphone implementation that hosts a WebSocket server.

    This microphone exposes a WebSocket server that receives audio chunks from connected clients.
    Only one client can be connected at a time.

    Clients must send PCM audio data in one of these formats:
    - Binary (raw)
    - Base64 encoded
    - With JSON envelope

    Also, clients are expected to respect the sample rate, channels, format, and chunk size
    specified during initialization.
    """

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 10,
        audio_format: Literal["binary", "base64", "json"] = "binary",
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = CHUNK_BALANCED,
    ):
        """
        Initialize WebSocket microphone server.

        Args:
            port (int): Port to bind the server to (default: 8080)
            timeout (int): Connection timeout in seconds (default: 10)
            audio_format (str): Expected audio format from clients ("binary", "base64", "json") (default: "binary")
            sample_rate (int): Sample rate in Hz (default: 16000)
            channels (int): Number of audio channels (default: 1)
            format (str): Audio format (default: "S16_LE")
            chunk_size (int): Number of frames per chunk (default: 1024). This parameter is advisory,
                it's sent to clients to suggest an optimal chunk size but clients may ignore it.
        """
        super().__init__(sample_rate, channels, format, chunk_size)

        # Determine numpy dtype based on format
        self._dtype = self._resolve_dtype(format)
        if self._dtype is None:
            raise MicrophoneConfigError(f"Unsupported format: {format}")

        self.host = "0.0.0.0"
        self.port = port
        self.timeout = timeout
        self.audio_format = audio_format
        self.logger = logger

        self._audio_queue = queue.Queue(10)
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: Optional[websockets.ServerConnection] = None
        self._client_lock = asyncio.Lock()

    def _resolve_dtype(self, format: str) -> np.dtype | None:
        """Get numpy dtype for audio format."""
        # Mapping format string -> numpy dtype
        format_map = {
            "S8": np.int8,
            "U8": np.uint8,
            "S16_LE": "<i2",
            "S16_BE": ">i2",
            "U16_LE": "<u2",
            "U16_BE": ">u2",
            "S24_LE": "<i4",  # 24-bit packed in 32-bit container
            "S24_BE": ">i4",  # 24-bit packed in 32-bit container
            "S32_LE": "<i4",
            "S32_BE": ">i4",
            "U32_LE": "<u4",
            "U32_BE": ">u4",
            "FLOAT_LE": "<f4",
            "FLOAT_BE": ">f4",
            "FLOAT64_LE": "<f8",
            "FLOAT64_BE": ">f8",
        }
        nf = format_map.get(format)
        return np.dtype(nf) if nf else None

    def _get_format_details(self, format: str) -> dict | None:
        """Get detailed format information for clients by introspecting the numpy dtype."""
        dtype = self._resolve_dtype(format)
        if dtype is None:
            return None

        # Ensure we have a dtype for introspection
        dt = np.dtype(dtype)

        # Determine bit depth
        bit_depth_override = {  # Special cases where bit depth differs from dtype size
            "S24_LE": 24,  # 24-bit packed in 32-bit container
            "S24_BE": 24,  # 24-bit packed in 32-bit container
        }
        if format in bit_depth_override:
            bit_depth = bit_depth_override[format]
        else:
            bit_depth = dt.itemsize * 8

        # Determine sample format from dtype kind
        if dt.kind == "i":  # signed integer
            sample_format = "signed_integer"
        elif dt.kind == "u":  # unsigned integer
            sample_format = "unsigned_integer"
        elif dt.kind == "f":  # floating point
            sample_format = "float"
        else:
            sample_format = "unknown"

        # Determine byte order
        if dt.byteorder == "<":
            byte_order = "little_endian"
        elif dt.byteorder == ">":
            byte_order = "big_endian"
        elif dt.byteorder == "|":
            # Not applicable (single byte types like int8, uint8)
            byte_order = "n/a"
        else:
            # Native byte order ('=') or other - determine from system
            import sys

            byte_order = "little_endian" if sys.byteorder == "little" else "big_endian"

        return {
            "bit_depth": bit_depth,
            "sample_format": sample_format,
            "byte_order": byte_order,
        }

    def _open_microphone(self) -> None:
        """Start the WebSocket server."""
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        start_timeout = 2.0
        while self._server is None and time.time() - start_time < start_timeout:
            time.sleep(0.01)

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

            # If port was 0, update with the actual bound port
            if self.port == 0 and self._server.sockets:
                actual_port = self._server.sockets[0].getsockname()[1]
                self.port = actual_port
                logger.info(f"WebSocket microphone server started on {self.host}:{self.port} (auto-assigned)")
            else:
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
                format_details = self._get_format_details(self.format)
                welcome_msg = {
                    "status": "connected",
                    "message": "You are now connected to the microphone server",
                    "audio_format": self.audio_format,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "format": self.format,
                    "chunk_size": self.chunk_size,
                }

                # Add universal format details if available
                if format_details:
                    welcome_msg.update(format_details)

                await self._send_to_client(welcome_msg)
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

    def _close_microphone(self) -> None:
        """Stop the WebSocket server."""
        if self._loop is not None and self._server is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._stop_and_disconnect_client(), self._loop)
                future.result(timeout=1.0)
            except CancelledError:
                logger.debug(f"Error stopping WebSocket server: CancelledError")
            except TimeoutError:
                logger.debug(f"Error stopping WebSocket server: TimeoutError")
            except Exception as e:
                logger.warning(f"Error stopping WebSocket server: {e}")

        # Wait for server thread to finish
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=10.0)

        # Clear frame queue
        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass

        # Reset state
        self._server = None
        self._loop = None
        self._client = None

    async def _stop_and_disconnect_client(self):
        """Set the async stop event and close the client connection."""
        # Send goodbye message and close the client connection
        if self._client:
            try:
                # Send goodbye message before closing
                await self._send_to_client({
                    "status": "disconnecting",
                    "message": "Server is shutting down. Connection will be closed.",
                })
                # Give a brief moment for the message to be sent
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing client in stop event: {e}")
            finally:
                await self._client.close()

        self._stop_event.set()

    def _read_audio(self) -> Optional[np.ndarray]:
        """
        Read a single audio chunk from the WebSocket microphone.

        Returns audio chunks as received from clients.
        """
        try:
            # Get chunk with short timeout to avoid blocking
            audio_chunk = self._audio_queue.get(timeout=0.1)
            return audio_chunk
        except queue.Empty:
            return None

    async def _send_to_client(self, message: dict) -> None:
        """Send a message to the connected client."""
        if self._client is None:
            raise ConnectionError("No client connected to send message to")

        if isinstance(message, dict):
            message = json.dumps(message)

        try:
            await self._client.send(message)
        except Exception as e:
            logger.warning(f"Error sending message to client: {e}")
