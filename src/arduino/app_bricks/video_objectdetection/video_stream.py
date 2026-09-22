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
import time
from dataclasses import dataclass
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


@dataclass
class _Track:
    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    last_seen: float


class BoxStabilizer:
    """Steadies the boxes drawn on the video across results.

    A box appears when its score passes the threshold, then follows the matching box of the next results,
    its position and score smoothed. It keeps showing while the matching score stays within `margin` below
    the threshold, and for `hold` seconds after the last match, so scores hovering around the threshold and
    single missed results do not make it flicker. The smoothing depends on the size of the change relative to
    the box, so it behaves the same at every resolution and distance: the noise of a still object is ignored,
    small changes are followed slowly and large ones almost at once, so a still box stays still and a moving
    one keeps up. The callbacks of the brick see the raw detections, this only shapes what the viewers see.
    """

    MIN_HOLD = 0.25  # seconds a box outlives its last match, at least...
    HOLD_PERIODS = 2  # ...or this many times its inference took: a model that stops answering leaves no ghost boxes
    MARGIN = 0.15  # a box already shown survives scores this far below the threshold
    MIN_IOU = 0.3  # the overlap a box must have with a track of the same label to be its next position
    JITTER = 0.03  # an edge moving less than this fraction of the box size is noise on a still object and is ignored
    MOTION = 0.5  # an edge moving this fraction of the box size is movement and is followed at FAST_SMOOTHING
    SMOOTHING = 0.15  # weight of a change just above the jitter against the smoothed coordinate
    FAST_SMOOTHING = 0.8  # weight of a change at or beyond MOTION
    SCORE_SMOOTHING = 0.2  # weight of the new score against the displayed one

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._hold = self.MIN_HOLD
        self._lock = threading.Lock()

    def update(self, boxes: list, threshold: float, round_trip: float) -> None:
        """Feed the boxes of one result (objects with label, score, x, y, w, h) and the seconds it took."""
        now = time.monotonic()
        with self._lock:
            self._hold = max(self.MIN_HOLD, self.HOLD_PERIODS * round_trip)
            unmatched = sorted(boxes, key=lambda box: box.score, reverse=True)
            for track in self._tracks:
                match = self._best_match(track, unmatched)
                if match is None or match.score < threshold - self.MARGIN:
                    continue
                unmatched.remove(match)
                new = (match.x, match.y, match.x + match.w, match.y + match.h)
                width, height = track.box[2] - track.box[0], track.box[3] - track.box[1]
                track.box = tuple(self._follow(old, n, size) for old, n, size in zip(track.box, new, (width, height, width, height)))
                track.score += self.SCORE_SMOOTHING * (match.score - track.score)
                track.last_seen = now
            for box in unmatched:
                if box.score >= threshold:
                    self._tracks.append(_Track(box.label, (box.x, box.y, box.x + box.w, box.y + box.h), box.score, now))
            self._tracks = [track for track in self._tracks if now - track.last_seen <= self._hold]

    def visible(self) -> dict[str, list[dict]]:
        """The boxes to draw now, in the shape of the brick's detections."""
        now = time.monotonic()
        detections: dict[str, list[dict]] = {}
        with self._lock:
            for track in self._tracks:
                if now - track.last_seen <= self._hold:
                    xyxy = tuple(round(v) for v in track.box)
                    detections.setdefault(track.label, []).append({"confidence": round(track.score, 2), "bounding_box_xyxy": xyxy})
        return detections

    @classmethod
    def _follow(cls, old: float, new: float, size: float) -> float:
        """The smoothed edge coordinate of a box `size` wide or tall: unchanged within the jitter, then moved by
        a weight growing with the change, both measured against the box size, a pixel being the least that shows."""
        delta = new - old
        jitter, motion = max(1.0, cls.JITTER * size), max(2.0, cls.MOTION * size)
        distance = abs(delta) - jitter
        if distance <= 0:
            return old
        weight = cls.SMOOTHING + (cls.FAST_SMOOTHING - cls.SMOOTHING) * min(1.0, distance / (motion - jitter))
        return old + weight * delta

    @classmethod
    def _best_match(cls, track: _Track, boxes: list) -> object | None:
        best, best_iou = None, cls.MIN_IOU
        for box in boxes:
            if box.label != track.label:
                continue
            iou = _iou(track.box, (box.x, box.y, box.x + box.w, box.y + box.h))
            if iou >= best_iou:
                best, best_iou = box, iou
        return best


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw_detections(frame: np.ndarray, detections: dict[str, list[dict]], colors: LabelColors) -> np.ndarray:
    """A copy of the frame with a box around every detection and a filled label chip at its top-right corner, "label" and "(score)".

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
            chip_w = max(cv2.getTextSize(line, FONT, font_scale, font_thickness)[0][0] for line in lines) + 2 * pad
            left = max(0, x2 - chip_w)  # right-aligned with the box, kept inside the frame
            top = y1 - chip_h if y1 - chip_h >= 0 else y1  # inside the box when there is no room above
            cv2.rectangle(image, (left, top), (left + chip_w, top + chip_h), color, cv2.FILLED)
            for i, line in enumerate(lines):
                text_w = cv2.getTextSize(line, FONT, font_scale, font_thickness)[0][0]
                origin = (left + (chip_w - text_w) // 2, top + pad + (i + 1) * (line_h + pad) - pad // 2)
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
        with self._cond:
            self._clients += 1
        seq = 0
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
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
