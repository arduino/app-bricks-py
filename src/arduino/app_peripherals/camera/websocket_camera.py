# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
import base64
import threading
import queue
import time
from typing import Optional, Union
import numpy as np
import cv2
import websockets
import asyncio

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraOpenError

logger = Logger("WebSocketCamera")


class WebSocketCamera(BaseCamera):
    """
    WebSocket Camera implementation that hosts a WebSocket server.
    
    This camera acts as a WebSocket server that receives frames from connected clients.
    Clients can send frames in various formats:
    - Base64 encoded images
    - JSON messages with image data
    - Binary image data
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, timeout: int = 10, 
                 frame_format: str = "base64", **kwargs):
        """
        Initialize WebSocket camera server.

        Args:
            host: Host address to bind the server to (default: "0.0.0.0")
            port: Port to bind the server to (default: 8080)
            timeout: Connection timeout in seconds (default: 10)
            frame_format: Expected frame format from clients ("base64", "json", "binary") (default: "base64")
            **kwargs: Additional camera parameters propagated to BaseCamera
        """
        super().__init__(**kwargs)
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.frame_format = frame_format
        
        self._frame_queue = queue.Queue(1)
        self._server = None
        self._loop = None
        self._server_thread = None
        self._stop_event = asyncio.Event()
        self._client: Optional[websockets.ServerConnection] = None

    def _open_camera(self) -> None:
        """Start the WebSocket server."""
        # Start server in separate thread with its own event loop
        self._server_thread = threading.Thread(
            target=self._start_server_thread,
            daemon=True
        )
        self._server_thread.start()
        
        # Wait for server to start
        start_time = time.time()
        start_timeout = 10
        while self._server is None and time.time() - start_time < start_timeout:
            if self._server is not None:
                break
            time.sleep(0.1)
        
        if self._server is None:
            raise CameraOpenError(f"Failed to start WebSocket server on {self.host}:{self.port}")

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
            
            logger.info(f"WebSocket camera server started on {self.host}:{self.port}")
            
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
        
        if self._client is not None:
            # Reject the new client
            logger.warning(f"Rejecting client {client_addr}: only one client allowed at a time")
            try:
                await conn.send(json.dumps({
                    "error": "Server busy",
                    "message": "Only one client connection allowed at a time",
                    "code": 1000
                }))
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
                    "message": "You are now connected to the camera server",
                    "frame_format": self.frame_format,
                    "resolution": self.resolution,
                    "fps": self.fps,
                })
            except Exception as e:
                logger.warning(f"Could not send welcome message to {client_addr}: {e}")

            async for message in conn:
                frame = await self._parse_message(message)
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
                                break
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.warning(f"Error handling client {client_addr}: {e}")
        finally:
            if self._client == conn:
                self._client = None
                logger.info(f"Client removed: {client_addr}")

    async def _parse_message(self, message) -> Optional[np.ndarray]:
        """Parse WebSocket message to extract frame."""
        try:
            if self.frame_format == "base64":
                # Expect base64 encoded image
                if isinstance(message, str):
                    image_data = base64.b64decode(message)
                else:
                    image_data = base64.b64decode(message.decode())
                
                # Decode image
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            
            elif self.frame_format == "binary":
                # Expect raw binary image data
                if isinstance(message, str):
                    image_data = message.encode()
                else:
                    image_data = message
                
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            
            elif self.frame_format == "json":
                # Expect JSON with image data
                if isinstance(message, bytes):
                    message = message.decode()
                
                data = json.loads(message)
                
                if "image" in data:
                    image_data = base64.b64decode(data["image"])
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return frame
                
                elif "frame" in data:
                    # Handle different frame data formats
                    frame_data = data["frame"]
                    if isinstance(frame_data, str):
                        image_data = base64.b64decode(frame_data)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return frame
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing message: {e}")
            return None

    def _close_camera(self):
        """Stop the WebSocket server."""
        # Signal async stop event if it exists
        if self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._set_async_stop_event(), 
                self._loop
            )
            try:
                future.result(timeout=1.0)
            except Exception as e:
                logger.warning(f"Error setting async stop event: {e}")
        
        # Wait for server thread to finish
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=10.0)
        
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state
        self._server = None
        self._loop = None
        self._client = None

    async def _set_async_stop_event(self):
        """Set the async stop event and close the client connection."""
        self._stop_event.set()
        
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
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing client in stop event: {e}")

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the queue."""
        try:
            # Get frame with short timeout to avoid blocking
            frame = self._frame_queue.get(timeout=0.1)
            return frame
        except queue.Empty:
            return None

    def _send_message_to_client(self, message: Union[str, bytes, dict]) -> None:
        """
        Send a message to the connected client (if any).
        
        Args:
            message: Message to send to the client
            
        Raises:
            RuntimeError: If the event loop is not running or closed
            ConnectionError: If no client is connected
            Exception: For other communication errors
        """
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("WebSocket server event loop is not running")
        
        if self._client is None:
            raise ConnectionError("No client connected to send message to")
        
        # Schedule message sending in the server's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._send_to_client(message), 
            self._loop
        )
        
        try:
            future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            raise

    async def _send_to_client(self, message: Union[str, bytes, dict]) -> None:
        """Send message to a single client."""
        if isinstance(message, dict):
            message = json.dumps(message)
        
        try:
            await self._client.send(message)
        except Exception as e:
            logger.warning(f"Error sending to client: {e}")
            raise
