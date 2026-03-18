# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""WebSocket-based frame input source."""

import asyncio
import base64
import json
import queue
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import websockets

from aihub.base import InputSource
from aihub.logging import logger


class WebSocketInput(InputSource):
    """
    WebSocket server that receives frames from connecting clients.

    Expects frames as JSON messages with base64-encoded JPEG data:
    {"frame": "<base64-encoded-jpeg>", "width": W, "height": H}
    """

    def __init__(
        self,
        on_frame_cb: Callable[[np.ndarray], None],
        host: str = "0.0.0.0",
        port: int = 5000,
        **kwargs,
    ):
        """
        Initialize WebSocket input server.

        Args:
            on_frame_cb: Callback for each received frame.
            host: Host to bind the server to.
            port: Port for the WebSocket server.
            kwargs: Additional keyword arguments.
        """
        self._on_frame_cb = on_frame_cb
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)

        self._host = host
        self._port = port

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the WebSocket input server and process frames."""
        self._running = True
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()

        # Block here processing frames (like GStreamerInput)
        try:
            while self._running:
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                    self._on_frame_cb(frame)
                except queue.Empty:
                    continue
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._server:
            self._server.close()

        if self._thread:
            self._thread.join(timeout=2.0)

    def _thread_target(self) -> None:
        self._loop = asyncio.new_event_loop()

        try:
            self._loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket input server error: {e}")
        finally:
            self._loop.close()

    async def _run_server(self):
        import logging

        logging.getLogger("websockets").setLevel(logging.ERROR)

        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info(f"WebSocket input server listening on ws://{self._host}:{self._port}")

        await self._server.wait_closed()

    async def _handler(self, websocket, path=None):
        """Handle a WebSocket connection."""
        logger.debug("WebSocket input client connected")

        try:
            async for message in websocket:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    frame = self._decode_frame(data)
                    if frame is not None:
                        # Queue frame for main thread processing
                        try:
                            self._frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Drop frame if queue full

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from client: {e}")

        except Exception as e:
            logger.warning(f"WebSocket input client error: {e}")
        finally:
            logger.debug("WebSocket input client disconnected")

    def _decode_frame(self, data: dict) -> Optional[np.ndarray]:
        """Decode a frame from the message payload."""
        try:
            frame_data = base64.b64decode(data["frame"])

            arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            # Convert BGR to RGB (cv2.cvtColor returns a contiguous array)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.warning(f"Error decoding frame: {e}")
            return None
