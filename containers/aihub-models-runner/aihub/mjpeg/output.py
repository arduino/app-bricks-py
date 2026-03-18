# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""Flask-based MJPEG output."""

import threading
import time
from typing import Optional

import numpy as np

import cv2
from flask import Flask, Response
from waitress import serve

from aihub.base import OutputSink
from aihub.logging import logger


class MJPEGOutput(OutputSink):
    """
    Flask-based MJPEG streaming output.

    Provides a web interface for viewing the processed video stream.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5002,
        **kwargs,
    ):
        """
        Initialize MJPEG output.

        Args:
            host: Host to bind to.
            port: HTTP server port.
            kwargs: Additional keyword arguments.
        """
        self._host = host
        self._port = port
        self._jpeg_quality = 80

        self._app = Flask(__name__)
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the MJPEG server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        logger.info(f"MJPEG server listening on http://{self._host}:{self._port}")

    def stop(self) -> None:
        """Stop the MJPEG server."""
        self._running = False

    def send_frame(self, frame: np.ndarray, metadata: dict) -> None:
        """
        Send a frame to the MJPEG server.

        Args:
            frame: RGB np.ndarray frame.
            metadata: dict containing metadata about the frame (ignored for MJPEG output).
        """
        # Convert RGB to BGR for OpenCV encoding
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ok, jpeg = cv2.imencode(
            ".jpeg",
            bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )

        if ok:
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()

    def _run_server(self) -> None:
        """Run the Waitress server (called in background daemon thread)."""
        import logging

        logging.getLogger("waitress").setLevel(logging.ERROR)

        self._setup_routes()

        serve(
            self._app,
            host=self._host,
            port=self._port,
            threads=3,
        )

    def _setup_routes(self) -> None:
        """Configure Flask routes."""

        @self._app.route("/")
        def index():
            return """
            <html>
              <head><title>AIHub Stream</title></head>
              <body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh;">
                <img src="/stream" style="max-width:100%;max-height:100%;" />
              </body>
            </html>
            """

        @self._app.route("/stream")
        def stream():
            response = Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response

    def _mjpeg_generator(self):
        """Generate MJPEG frames for streaming."""
        while self._running:
            with self._lock:
                frame = self._latest_jpeg

            if frame is None:
                time.sleep(0.01)
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            # Cap at ~60fps
            time.sleep(1 / 60)
