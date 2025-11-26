# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
import os
import threading
import queue
import time
import numpy as np
import cv2
import websockets
import asyncio
from collections.abc import Callable

from arduino.app_internal.core.peripherals import BPPCodec
from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraOpenError

logger = Logger("WebSocketCamera")


class WebSocketCamera(BaseCamera):
    """
    WebSocket Camera implementation that hosts a WebSocket server.

    This camera acts as a WebSocket server that receives frames from connected clients.
    Only one client can be connected at a time.

    Clients must encode video frames in one of these formats:
    - JPEG
    - PNG
    - WebP
    - BMP
    - TIFF

    Secure communication with the WebSocket server is supported in three security modes:
    - Security disabled (empty secret)
    - Authenticated (secret + enable_encryption=False) - HMAC-SHA256
    - Authenticated + Encrypted (secret + enable_encryption=True) - ChaCha20-Poly1305

    The frames can be serialized in one of the formats supported by BPPCodec.
    """

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 3,
        secret: str = "",
        enable_encryption: bool = False,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize WebSocket camera server with security options.

        Args:
            port: Port to bind the server to
            timeout: Connection timeout in seconds
            frame_format: Expected frame format ("binary", "json")
            secret: Secret key for authentication/encryption (empty = security disabled)
            enable_encryption: Enable encryption (only effective if secret is provided)
            resolution: Resolution as (width, height)
            fps: Frames per second to capture
            adjustments: Function to adjust frames
            auto_reconnect: Enable automatic reconnection on failure
        """
        super().__init__(resolution, fps, adjustments, auto_reconnect)

        self.protocol = "ws"
        self.port = port
        self.timeout = timeout
        self.codec = BPPCodec(secret, enable_encryption)
        self.secret = secret
        self.enable_encryption = enable_encryption
        self.logger = logger

        host_ip = os.getenv("HOST_IP")
        self._bind_ip = "0.0.0.0"
        self.ip = host_ip if host_ip is not None else self._bind_ip

        self._frame_queue = queue.Queue(1)
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: websockets.ServerConnection | None = None
        self._client_lock = asyncio.Lock()

    @property
    def url(self) -> str:
        """Return the WebSocket server address."""
        return f"{self.protocol}://{self.ip}:{self.port}"

    @property
    def security_mode(self) -> str:
        """Return current security mode for logging/debugging."""
        if not self.secret:
            return "none"
        elif self.enable_encryption:
            return "encrypted (ChaCha20-Poly1305)"
        else:
            return "authenticated (HMAC-SHA256)"

    def _open_camera(self) -> None:
        """Start the WebSocket server."""
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if self._server is not None:
                self.logger.info(f"WebSocket camera server started on {self.url}, security: {self.security_mode}")
                return
            time.sleep(0.1)

        # Cleanup server thread if it failed to start in time
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

        raise CameraOpenError(f"Failed to start WebSocket server on {self.url}")

    def _start_server_thread(self) -> None:
        """Run WebSocket server in its own thread with event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_server())
        except Exception as e:
            self.logger.error(f"WebSocket server thread error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self._stop_event.clear()

            self._server = await asyncio.wait_for(
                websockets.serve(
                    self._ws_handler,
                    self._bind_ip,
                    self.port,
                    open_timeout=self.timeout,
                    ping_timeout=self.timeout,
                    close_timeout=self.timeout,
                    ping_interval=20,
                    max_size=5 * 1024 * 1024,  # Limit max message size for security
                ),
                timeout=self.timeout,
            )

            # Get the actual port if OS assigned one (i.e. when port=0)
            if self.port == 0:
                server_socket = list(self._server.sockets)[0]
                self.port = server_socket.getsockname()[1]

            await self._stop_event.wait()

        except TimeoutError as e:
            self.logger.error(f"Failed to start WebSocket server within {self.timeout}s: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

    async def _ws_handler(self, conn: websockets.ServerConnection) -> None:
        """Handle a connected WebSocket client. Only one client allowed at a time."""
        client_addr = f"{conn.remote_address[0]}:{conn.remote_address[1]}"

        async with self._client_lock:
            if self._client is not None:
                # Reject the new client
                self.logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
                try:
                    rejection = json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000})
                    await self._send_to_client(rejection, client=conn)
                    await conn.close(code=1000, reason="Server busy")
                except Exception as e:
                    self.logger.warning(f"Failed to send rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        self._set_status("connected", {"client_address": client_addr})
        self.logger.debug(f"Client connected: {client_addr}")

        try:
            try:
                # Send welcome message
                welcome = json.dumps({
                    "status": "connected",
                    "message": "Connected to camera server",
                    "resolution": self.resolution,
                    "fps": self.fps,
                    "security_mode": self.security_mode,
                })
                await self._send_to_client(welcome)
            except Exception as e:
                self.logger.warning(f"Failed to send welcome message: {e}")

            # Handle incoming messages
            async for message in conn:
                frame = self._parse_message(message)
                if frame is None:
                    continue

                # Drop old frames until there's room for the new one
                while True:
                    try:
                        self._frame_queue.put_nowait(frame)
                        break
                    except queue.Full:
                        try:
                            # Drop oldest frame and try again
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            continue

        except websockets.exceptions.ConnectionClosed:
            self.logger.debug(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            async with self._client_lock:
                if self._client == conn:
                    self._client = None
                    self._set_status("disconnected", {"client_address": client_addr})

    def _parse_message(self, message: websockets.Data) -> np.ndarray | None:
        if isinstance(message, str):
            try:
                message = base64.b64decode(message)
            except Exception as e:
                self.logger.warning(f"Failed to decode string message using base64: {e}")
                return None

        decoded = self.codec.decode(message)
        if decoded is None:
            self.logger.warning("Failed to decode/authenticate message")
            return None

        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return frame

    def _close_camera(self):
        """Stop the WebSocket server."""
        # Only attempt cleanup if the event loop is running
        if self._loop and not self._loop.is_closed() and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._stop_and_disconnect_client(), self._loop)
                future.result(1.0)
            except Exception as e:
                self.logger.warning(f"Failed to stop WebSocket server cleanly: {e}")

        # Wait for server thread to finish
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=10.0)

        # Clear frame queue
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        # Reset state
        self._server = None
        self._loop = None
        self._client = None

    async def _stop_and_disconnect_client(self):
        """Cleanly disconnect client with goodbye message."""
        if self._client:
            try:
                self.logger.debug("Disconnecting client...")
                goodbye = json.dumps({"status": "disconnecting", "message": "Server is shutting down"})
                await self._send_to_client(goodbye)
            except Exception as e:
                self.logger.warning(f"Failed to send goodbye message: {e}")
            finally:
                if self._client:
                    await self._client.close()
                    self.logger.debug("Client connection closed")

        self._stop_event.set()

    def _read_frame(self) -> np.ndarray | None:
        """Read a frame from the queue."""
        try:
            return self._frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    async def _send_to_client(self, data: bytes | str, client: websockets.ServerConnection | None = None):
        """Send secure message to connected client."""
        if isinstance(data, str):
            data = data.encode()

        encoded = self.codec.encode(data)

        # Keep a ref to current client to avoid locking
        client = client or self._client
        if client is None:
            raise ConnectionError("No client connected")

        try:
            await client.send(encoded)
        except websockets.ConnectionClosedOK:
            self.logger.warning("Client has already closed the connection")
        except websockets.ConnectionClosedError as e:
            self.logger.warning(f"Client has already closed the connection with error: {e}")
        except Exception:
            raise
