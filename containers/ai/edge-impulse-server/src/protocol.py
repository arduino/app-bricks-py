# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Protocol between clients and the inference server (Unix domain socket SOCK_STREAM).

Every message:  [type: 4 ASCII bytes] [payload length: uint32 LE] [payload]

Many clients connect to the same socket. One connection = one model.

    client: OPEN {"model": name}        first message, mandatory
    server: OPND {...}                  once the model is ready; the client waits
            ERR  {...} + close          if the model cannot be opened
    client: FRAM ...                    repeated, up to "slots" frames in flight
    server: RSLT {...} | ERR {...}      the connection stays open after an ERR too
            SLOT {...}                  when the number of frames the client may keep in flight changes
    client: CONF {...}                  optional, sets threshold values of the model
    server: CONF {...} | ERR {...}      the blocks as they stand, or bad_request

The model stays available as long as the connection is open. On disconnect
the server releases it; if no other connection is using it and it is not
pinned, the server terminates it.

Slots: the server may run several instances of a model and tells each connection how many frames it
may keep in flight, 1 at open. The client sends a frame whenever fewer than that are in flight; the
server chooses when to raise the allowance so the results of a connection stay evenly spaced, and it
sends results in arrival order, dropping one that completes after a later frame's result went out.
A client that ignores the allowance and sends one frame at a time keeps working.

Thresholds: the .eim exposes its threshold blocks, each with an "id" and a "type" ("object_detection" with
"min_score", "object_tracking" with "max_age", "min_hits" and "iou_threshold" or "threshold"...), "thresholds"
in OPND. A CONF message sets values of one block; they belong to the model, so they hold for every connection
using it and for the instances added later. A model with the object tracking block reports "object_tracking"
true and its "tracks" in every result: the boxes with the "id" of the object, stable while it stays in view.

Messages
  OPEN  C->S  JSON     {"model": name}
  OPND  S->C  JSON     {"model", "project", "width", "height", "channels", "labels", "model_type",
                        "resize_mode", "object_tracking", "thresholds", "slots"}
  FRAM  C->S  binary   FRAME_HEADER + the pixels of the frame, any size, RGB or BGR; the server
                       resizes it to the model input as the Studio does
  RSLT  S->C  JSON     {"seq", "ts_ns", "boxes", "tracks", "classes", "anomaly", "timing_ms", "slots"}, the
                       boxes and the tracks in the coordinates of the submitted frame
  SLOT  S->C  JSON     {"slots"}, the frames the client may keep in flight from now on
  CONF  C->S  JSON     {"id": block id, key: value, ...}, threshold values of one block of the model
  CONF  S->C  JSON     {"thresholds"}, the blocks with their current values, once the values are set
  ERR   S->C  JSON     {"op": "open"|"frame"|"configure", "code", "error", "slots", ...}
"""

import json
import struct
import time

import numpy as np

SOCKET_NAME = "ei.sock"
PROTOCOL_VERSION = 0  # the protocol is not versioned yet, every change is breaking

OPEN = b"OPEN"
OPENED = b"OPND"
FRAME = b"FRAM"
RESULT = b"RSLT"
SLOTS = b"SLOT"
CONFIGURE = b"CONF"
ERROR = b"ERR "

# Open errors: the server closes the connection after sending them
E_BAD_REQUEST = "bad_request"  # invalid first message, unexpected message, unknown threshold block or key
E_TOO_MANY_CLIENTS = "too_many_clients"  # --max-clients reached
E_UNKNOWN_MODEL = "unknown_model"  # no <name>.eim file
E_TOO_MANY_MODELS = "too_many_models"  # --max-models reached
E_LOW_MEMORY = "low_memory"  # available memory below --memory-reserve-mb
E_LOAD_FAILED = "load_failed"  # the .eim file does not start
# Frame errors: the connection stays open
E_BAD_FRAME = "bad_frame"  # invalid size or format
E_INTERNAL = "internal"  # error during inference

HEADER = struct.Struct("<4sI")  # type, payload length
FRAME_HEADER = struct.Struct("<QqHHBB2x")  # seq, ts_ns, width, height, channels, color (RGB or BGR)
RGB, BGR = 0, 1  # the color codes of the frame header
MAX_FRAME_PIXELS = 1920 * 1080  # frames above this are discarded, the buffer of a connection never grows past them
MAX_PAYLOAD = 16 * 1024 * 1024  # above this the stream is considered corrupted
MAX_CONTROL_PAYLOAD = 64 * 1024  # enough for any JSON message


class ProtocolError(Exception):
    pass


class PayloadTooLarge(ProtocolError):
    """Payload above the Reader limit: it was read and discarded, only `head` was kept."""

    def __init__(self, kind: bytes, length: int, head: bytes):
        super().__init__(f"{kind!r} payload of {length} bytes exceeds the limit")
        self.kind, self.length, self.head = kind, length, head


def frame_size(width: int, height: int, channels: int = 3) -> int:
    """Payload length of a FRAM message for an image of the given size."""
    return FRAME_HEADER.size + width * height * channels


MAX_FRAME_PAYLOAD = frame_size(MAX_FRAME_PIXELS, 1)


# ------------------------------------------------------------------ sending
def send_json(sock, kind: bytes, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(HEADER.pack(kind, len(data)) + data)


def send_frame(sock, seq: int, ts_ns: int, image: np.ndarray, color: int = RGB) -> None:
    """Send an HxWx3 uint8 image, in RGB or BGR order, without copying its pixels into a new buffer."""
    h, w, c = image.shape
    pixels = memoryview(np.ascontiguousarray(image)).cast("B")
    head = FRAME_HEADER.pack(seq, ts_ns, w, h, c, color)
    sock.sendall(HEADER.pack(FRAME, len(head) + len(pixels)) + head)
    sock.sendall(pixels)


# ------------------------------------------------------------------ receiving
class Reader:
    """Reads complete messages, reusing the same buffer.
    The returned payload is only valid until the next read.
    Payloads above `limit` are discarded instead of buffered (PayloadTooLarge),
    so the buffer never grows beyond the largest accepted message.
    `transfer_ms` is the time spent receiving the last payload once its header
    had arrived: the socket transfer, not the time waiting for the sender."""

    def __init__(self, sock, limit: int = MAX_PAYLOAD):
        self.sock = sock
        self.limit = limit
        self.buffer = bytearray(min(limit, MAX_CONTROL_PAYLOAD))
        self.transfer_ms = 0.0

    def read(self):
        kind, length = HEADER.unpack(self._read_exactly(HEADER.size))
        if length > MAX_PAYLOAD:
            raise ProtocolError(f"message too large: {length} bytes")
        if length > self.limit:
            head = bytes(self._read_exactly(min(length, FRAME_HEADER.size)))
            self._discard(length - len(head))
            raise PayloadTooLarge(bytes(kind), length, head)
        start = time.perf_counter()
        payload = self._read_exactly(length)
        self.transfer_ms = (time.perf_counter() - start) * 1e3
        return bytes(kind), payload

    def _read_exactly(self, n: int) -> memoryview:
        if len(self.buffer) < n:
            self.buffer = bytearray(n)
        view = memoryview(self.buffer)[:n]
        received = 0
        while received < n:
            count = self.sock.recv_into(view[received:], n - received)
            if count == 0:
                raise ConnectionError("connection closed")
            received += count
        return view

    def _discard(self, n: int) -> None:
        view = memoryview(self.buffer)
        while n > 0:
            count = self.sock.recv_into(view, min(n, len(view)))
            if count == 0:
                raise ConnectionError("connection closed")
            n -= count


def parse_json(payload) -> dict | None:
    """JSON object in the payload, or None if it is not a valid JSON object."""
    try:
        obj = json.loads(bytes(payload))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def frame_seq(payload) -> int | None:
    """seq of a FRAM payload, or None if the header is incomplete."""
    return FRAME_HEADER.unpack_from(payload)[0] if len(payload) >= FRAME_HEADER.size else None


def parse_frame(payload):
    """Return (seq, ts_ns, image, color). The image points into the Reader buffer."""
    if len(payload) < FRAME_HEADER.size:
        raise ProtocolError("frame too short")
    seq, ts_ns, w, h, c, color = FRAME_HEADER.unpack_from(payload)
    if c != 3:
        raise ProtocolError(f"expected 3 channels, received {c}")
    if color not in (RGB, BGR):
        raise ProtocolError(f"unknown color code {color}")
    if w == 0 or h == 0 or w * h > MAX_FRAME_PIXELS:
        raise ProtocolError(f"frame of {w}x{h} pixels, the limit is {MAX_FRAME_PIXELS} pixels")
    pixels = payload[FRAME_HEADER.size :]
    if len(pixels) != w * h * c:
        raise ProtocolError(f"expected {w * h * c} pixel bytes, received {len(pixels)}")
    return seq, ts_ns, np.frombuffer(pixels, np.uint8).reshape(h, w, c), color
