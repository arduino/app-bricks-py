# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Drawing of the detections on the frames and the HTTP server streaming them as MJPEG.

Replaces the video pages the Edge Impulse runner container used to serve on port 4912. The root is the
multipart/x-mixed-replace stream, which browsers render like an image at its natural size. /embed is the
page the existing viewers load in an iframe: a bare document showing the whole stream frame scaled into the
iframe, so the iframe load event fires as it did with the old page, while the stream itself never
completes a document.
"""

import colorsys
import random
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from arduino.app_utils import Logger

logger = Logger("VideoObjectDetection")

BOUNDARY = b"frame"
EMBED_PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Video Object Detection</title>
<style>html,body{margin:0;height:100%}img{display:block;width:100%;height:100%;object-fit:contain}</style></head>
<body><img src="/" alt="video"></body></html>
"""
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)


class LabelColors:
    """A color per label, picked the first time the label is seen and kept for the whole run.

    The first hue is random, the following ones step around the color wheel by the golden ratio, so the
    labels of a run are always distinct from each other.
    """

    HUE_STEP = 0.618034

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._hue = self._random.random()
        self._colors: dict[str, tuple[int, int, int]] = {}
        self._lock = threading.Lock()

    def __getitem__(self, label: str) -> tuple[int, int, int]:
        with self._lock:
            if label not in self._colors:
                # Light, moderately saturated colors keep the white label text readable
                r, g, b = colorsys.hsv_to_rgb(self._hue, self._random.uniform(0.35, 0.6), 0.95)
                self._colors[label] = (int(b * 255), int(g * 255), int(r * 255))
                self._hue = (self._hue + self.HUE_STEP) % 1.0
            return self._colors[label]


def draw_detections(frame: np.ndarray, detections: dict[str, list[dict]], colors: LabelColors) -> np.ndarray:
    """A copy of the frame with a box around every detection and a filled label above it, "label" and "(score)".

    Args:
        frame (np.ndarray): HxWx3 BGR frame.
        detections (dict): The detections of the frame, as the brick passes them to the callbacks.
        colors (LabelColors): The color of every label.

    Returns:
        np.ndarray: The annotated copy.
    """
    image = frame.copy()
    scale = max(image.shape[0] / 480, 0.5)  # sizes tuned on 480p frames
    thickness = max(1, round(1.5 * scale))
    font_scale, font_thickness = 0.5 * scale, 1 if scale < 1.5 else 2
    line_h = cv2.getTextSize("Ag", FONT, font_scale, font_thickness)[0][1]
    pad = max(3, round(5 * scale))
    for label, label_detections in detections.items():
        color = colors[label]
        for detection in label_detections:
            x1, y1, x2, y2 = (int(v) for v in detection["bounding_box_xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            lines = [label, f"({detection['confidence']:.2f})"]
            chip_h = len(lines) * (line_h + pad) + pad
            chip_w = max(x2 - x1, max(cv2.getTextSize(line, FONT, font_scale, font_thickness)[0][0] for line in lines) + 2 * pad)
            top = y1 - chip_h if y1 - chip_h >= 0 else y1  # inside the box when there is no room above
            cv2.rectangle(image, (x1, top), (x1 + chip_w, top + chip_h), color, cv2.FILLED)
            for i, line in enumerate(lines):
                text_w = cv2.getTextSize(line, FONT, font_scale, font_thickness)[0][0]
                origin = (x1 + (chip_w - text_w) // 2, top + pad + (i + 1) * (line_h + pad) - pad // 2)
                cv2.putText(image, line, origin, FONT, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    return image


class VideoStreamServer:
    """Serves the frames published to it as an MJPEG stream, every client receives the latest one."""

    def __init__(self, addr: str = "0.0.0.0", port: int = 4912) -> None:
        self._addr, self._port = addr, port
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        self._clients = 0
        self._stopped = False
        self._server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        """The port the server listens on, useful when started on port 0."""
        return self._server.server_address[1] if self._server else self._port

    @property
    def has_clients(self) -> bool:
        """True while at least one client receives the stream, the only time frames are worth rendering."""
        return self._clients > 0

    def start(self) -> None:
        """Start serving in a background thread."""
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 method name fixed by BaseHTTPRequestHandler
                path = self.path.split("?", 1)[0]
                if path == "/":
                    stream._serve_stream(self)
                elif path == "/embed":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(EMBED_PAGE)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(EMBED_PAGE)
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args: object) -> None:
                logger.debug(f"video stream: {format % args}")

        ThreadingHTTPServer.allow_reuse_address = True
        self._server = ThreadingHTTPServer((self._addr, self._port), Handler)
        self._server.daemon_threads = True
        with self._cond:
            self._stopped = False
        threading.Thread(target=self._server.serve_forever, daemon=True, name="VideoObjectDetectionStream").start()
        logger.info(f"Video stream available on port {self.port}")

    def stop(self) -> None:
        """Stop serving and release the connected clients."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def publish(self, jpeg: bytes) -> None:
        """Make a JPEG frame the current one, the connected clients receive it."""
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    def _next_frame(self, after: int, client: socket.socket) -> tuple[int, bytes] | None:
        """The first frame published after sequence number ``after``, None once stopped or the client left."""
        while True:
            with self._cond:
                if self._stopped:
                    return None
                if self._frame is not None and self._seq > after:
                    return self._seq, self._frame
                self._cond.wait(0.5)
            if _client_left(client):
                return None

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        with self._cond:
            self._clients += 1
        seq = 0
        try:
            while (item := self._next_frame(seq, handler.connection)) is not None:
                seq, frame = item
                handler.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                handler.wfile.write(frame)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
        except OSError:
            pass  # the client went away
        finally:
            with self._cond:
                self._clients -= 1


def _client_left(client: socket.socket) -> bool:
    """True when the client closed its side, a stream client never sends anything after its request."""
    readable, _, _ = select.select([client], [], [], 0)
    if not readable:
        return False
    try:
        return client.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True
