# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import os
import threading
import queue
import time
from typing import Literal
import numpy as np
import cv2
import websockets
import asyncio
from collections.abc import Callable
from concurrent.futures import CancelledError

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraConfigError, CameraOpenError

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

    The frames can be serialized in one of the following formats:
    - Binary image data
    - Base64 encoded images
    - JSON messages with image data
    """

    def __init__(
        self,
        port: int = 8080,
        timeout: int = 3,
        frame_format: Literal["binary", "base64", "json"] = "binary",
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """
        Initialize WebSocket camera server.

        Args:
            host (str): Host address to bind the server to (default: "0.0.0.0")
            port (int): Port to bind the server to (default: 8080)
            timeout (int): Connection timeout in seconds (default: 10)
            frame_format (str): Expected frame format from clients ("binary", "base64", "json") (default: "binary")
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int): Frames per second to capture from the camera.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None
        """
        super().__init__(resolution, fps, adjustments)

        self.protocol = "ws"
        self.port = port
        self.timeout = timeout
        if frame_format not in ["binary", "base64", "json"]:
            raise CameraConfigError(f"Invalid frame format: {frame_format}")
        self.frame_format = frame_format
        self.logger = logger

        host_ip = os.getenv("HOST_IP")
        self._bind_ip = "0.0.0.0"
        self._external_ip = host_ip if host_ip is not None else self._bind_ip

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
        return f"{self.protocol}://{self._external_ip}:{self.port}"

    def _open_camera(self) -> None:
        """Start the WebSocket server."""
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        start_time = time.time()
        start_timeout = self.timeout
        while time.time() - start_time < start_timeout:
            if self._server is not None:
                logger.info(f"WebSocket camera server started on {self.url}")
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
                ),
                timeout=self.timeout,
            )

            # Get the actual port if OS assigned one (i.e. when port=0)
            if self.port == 0:
                server_socket = list(self._server.sockets)[0]
                self.port = server_socket.getsockname()[1]

            await self._stop_event.wait()

        except TimeoutError as e:
            self.logger.error(f"Failed to start WebSocket server in a time ({self.timeout}s): {e}")
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
                    await conn.send(json.dumps({"error": "Server busy", "message": "Only one client connection allowed at a time", "code": 1000}))
                    await conn.close(code=1000, reason="Server busy - only one client allowed")
                except Exception as e:
                    self.logger.warning(f"Error sending rejection message to {client_addr}: {e}")
                return

            # Accept the client
            self._client = conn

        self._set_status("connected", {"client_address": client_addr})

        self.logger.debug(f"Client connected: {client_addr}")

        try:
            # Send welcome message
            try:
                await self._send_to_client({
                    "status": "connected",
                    "message": "You are now connected to the camera server",
                    "frame_format": self.frame_format,
                    "resolution": self.resolution,
                    "fps": self.fps,
                })
            except Exception as e:
                self.logger.warning(f"Could not send welcome message to {client_addr}: {e}")

            async for message in conn:
                frame = self._parse_message(message)
                if frame is not None:
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
                    self.logger.debug(f"Client disconnected: {client_addr}")

    def _parse_message(self, message: str | bytes) -> np.ndarray | None:
        """Parse WebSocket message to extract frame."""
        try:
            if self.frame_format == "binary":
                # Expect raw binary image data
                if isinstance(message, str):
                    # Use latin-1 encoding to preserve binary data
                    image_data = message.encode("latin-1")
                else:
                    image_data = message

                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                return frame

            elif self.frame_format == "base64":
                # Expect base64 encoded image
                if isinstance(message, str):
                    image_data = base64.b64decode(message)
                else:
                    image_data = base64.b64decode(message.decode())

                # Decode image
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                return frame

            elif self.frame_format == "json":
                # Expect JSON with image data
                if isinstance(message, bytes):
                    message = message.decode()

                data = json.loads(message)

                if "image" in data:
                    image_data = base64.b64decode(data["image"])
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    return frame

                elif "frame" in data:
                    # Handle different frame data formats
                    frame_data = data["frame"]
                    if isinstance(frame_data, str):
                        image_data = base64.b64decode(frame_data)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        return frame

            else:
                logger.error(f"Unknown video format: {self.frame_format}")
                return None

        except Exception as e:
            logger.warning(f"Error parsing message: {e}")
            return None

    def _close_camera(self):
        """Stop the WebSocket server."""
        # Only attempt cleanup if the event loop is running
        if self._loop and not self._loop.is_closed() and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._stop_and_disconnect_client(), self._loop)
                future.result(1.0)
            except CancelledError:
                self.logger.debug(f"Error stopping WebSocket server: CancelledError")
            except TimeoutError:
                self.logger.debug(f"Error stopping WebSocket server: TimeoutError")
            except Exception as e:
                self.logger.warning(f"Error stopping WebSocket server: {e}")

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
        """Set the async stop event and close the client connection."""
        # Send goodbye message and close the client connection
        if self._client:
            try:
                self.logger.debug("Disconnecting client...")
                # Send goodbye message before closing
                await self._send_to_client({
                    "status": "disconnecting",
                    "message": "Server is shutting down. Connection will be closed.",
                })
            except Exception as e:
                self.logger.warning(f"Failed to send 'disconnecting' event to closing client: {e}")
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

    async def _send_to_client(self, message: str | bytes | dict) -> None:
        """Send a message to the connected client."""
        if isinstance(message, dict):
            message = json.dumps(message)

        async with self._client_lock:
            if self._client is None:
                raise ConnectionError("No client connected to send message to")

            try:
                await self._client.send(message)
            except websockets.ConnectionClosedOK:
                self.logger.warning("Client has already closed the connection")
            except Exception:
                raise
