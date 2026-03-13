# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""WebSocket-based frame output sink."""

import asyncio
import base64
import json
import threading
from typing import Optional, Set

import numpy as np
import websockets

from aihub.base import OutputSink
from aihub.logging import logger


class WebSocketOutput(OutputSink):
    """
    WebSocket server that broadcasts frames to connected clients.

    Sends frames as JSON messages with base64-encoded JPEG data:
    {"frame": "<base64-encoded-jpeg>", "width": W, "height": H}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        **kwargs,
    ):
        """
        Initialize WebSocket output server.

        Args:
            host: Host to bind to.
            port: Server port.
        """
        self._host = host
        self._port = port
        self._jpeg_quality = 80

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._clients: Set = set()
        self._clients_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._server:
            self._server.close()

    def send_frame(self, frame: np.ndarray, metadata: dict) -> None:
        """
        Send a frame to all connected WebSocket clients.

        Args:
            frame: RGB np.ndarray frame.
            metadata: dict containing metadata about the frame.
        """
        if not self._clients or not self._loop:
            return

        message = self._encode_frame(frame, metadata)
        if message:
            asyncio.run_coroutine_threadsafe(self._broadcast_frame(message), self._loop)

    def _thread_target(self):
        self._loop = asyncio.new_event_loop()

        try:
            self._loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket output server error: {e}")
        finally:
            self._loop.close()

    async def _run_server(self):
        import logging

        logging.getLogger('websockets').setLevel(logging.ERROR)
        
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info(f"WebSocket output server listening on ws://{self._host}:{self._port}")

        await self._server.wait_closed()

    async def _handler(self, websocket, path=None):
        """Handle a WebSocket connection."""
        with self._clients_lock:
            self._clients.add(websocket)

        logger.debug(f"WebSocket output client connected (total: {len(self._clients)})")

        try:
            # Keep connection alive
            async for _ in websocket:
                pass
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            logger.debug(f"WebSocket output client disconnected (total: {len(self._clients)})")

    async def _broadcast_frame(self, message: str):
        """Broadcast message to all connected clients."""
        with self._clients_lock:
            clients = list(self._clients)

        if clients:
            # Use gather for concurrent sends
            await asyncio.gather(
                *[self._safe_send(client, message) for client in clients],
                return_exceptions=True,
            )

    async def _safe_send(self, client, message: str):
        """Send message to client, handling errors."""
        try:
            await client.send(message)
        except Exception:
            with self._clients_lock:
                self._clients.discard(client)

    def _encode_frame(self, frame: np.ndarray, metadata: dict) -> Optional[str]:
        """Encode frame to JSON message."""
        import cv2

        height, width = frame.shape[:2]

        # Convert RGB to BGR for OpenCV encoding
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(
            ".jpeg",
            bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            return None

        frame_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        return json.dumps(
            {
                "frame": frame_b64,
                "width": width,
                "height": height,
                "metadata": metadata,
            }
        )
