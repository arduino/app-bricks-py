# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Protocol between the clients and the inference service (Unix domain socket SOCK_STREAM).

Every message: [type: 4 ASCII bytes] [payload length: uint32 LE] [payload]. One connection uses one model,
requested with the first message; the connection stays open after a frame error, the server closes it
after an open error and releases the model when it closes. Kept identical to the copy in the containers.

    client: OPEN {"model": name}        first message, mandatory
    server: OPND {...}                  once the model is ready
            ERR  {...} + close          if the model cannot be opened
    client: FRAM ...                    repeated, up to "slots" frames in flight, any size, RGB or BGR
    server: RSLT {...} | ERR {...}      the connection stays open after an ERR too
            SLOT {...}                  when the number of frames the client may keep in flight changes

The server tells each connection how many frames it may keep in flight, "slots" in OPND, RSLT, ERR and
SLOT, 1 at open: it raises the allowance when it runs several instances of the model, at the moment
that keeps the results evenly spaced, and sends the results in arrival order.
"""

import json
import socket
import struct
import time
from typing import Any, cast

import numpy as np

SOCKET_NAME = "ei.sock"
PROTOCOL_VERSION = 0  # the protocol is not versioned yet, every change is breaking

OPEN = b"OPEN"
OPENED = b"OPND"
FRAME = b"FRAM"
RESULT = b"RSLT"
SLOTS = b"SLOT"
ERROR = b"ERR "

# Open errors: the server closes the connection after sending them
E_BAD_REQUEST = "bad_request"
E_TOO_MANY_CLIENTS = "too_many_clients"
E_UNKNOWN_MODEL = "unknown_model"
E_TOO_MANY_MODELS = "too_many_models"
E_LOW_MEMORY = "low_memory"
E_LOAD_FAILED = "load_failed"
# Frame errors: the connection stays open
E_BAD_FRAME = "bad_frame"
E_INTERNAL = "internal"

HEADER = struct.Struct("<4sI")  # type, payload length
FRAME_HEADER = struct.Struct("<QqHHBB2x")  # seq, ts_ns, width, height, channels, color (RGB or BGR)
RGB, BGR = 0, 1  # the color codes of the frame header
MAX_FRAME_PIXELS = 1920 * 1080  # frames above this are rejected by the service
MAX_PAYLOAD = 16 * 1024 * 1024  # above this the stream is considered corrupted
MAX_CONTROL_PAYLOAD = 64 * 1024  # enough for any JSON message


class ProtocolError(Exception):
    """The peer sent something the protocol does not allow."""


class PayloadTooLarge(ProtocolError):
    """Payload above the Reader limit: it was read and discarded, only ``head`` was kept."""

    def __init__(self, kind: bytes, length: int, head: bytes) -> None:
        super().__init__(f"{kind!r} payload of {length} bytes exceeds the limit")
        self.kind, self.length, self.head = kind, length, head


def frame_size(width: int, height: int, channels: int = 3) -> int:
    """Payload length of a FRAM message for an image of the given size."""
    return FRAME_HEADER.size + width * height * channels


MAX_FRAME_PAYLOAD = frame_size(MAX_FRAME_PIXELS, 1)


def send_json(sock: socket.socket, kind: bytes, obj: dict[str, Any]) -> None:
    """Send a control message with a JSON payload."""
    data = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(HEADER.pack(kind, len(data)) + data)


def send_frame(sock: socket.socket, seq: int, ts_ns: int, image: np.ndarray, color: int = RGB) -> None:
    """Send an HxWx3 uint8 image, in RGB or BGR order, without copying its pixels into a new buffer."""
    h, w, c = image.shape
    pixels = memoryview(np.ascontiguousarray(image)).cast("B")
    head = FRAME_HEADER.pack(seq, ts_ns, w, h, c, color)
    sock.sendall(HEADER.pack(FRAME, len(head) + len(pixels)) + head)
    sock.sendall(pixels)


class Reader:
    """Reads complete messages from a socket, reusing the same buffer.

    The returned payload is only valid until the next read. Payloads above ``limit`` are discarded instead
    of buffered (PayloadTooLarge), so the buffer never grows beyond the largest accepted message.
    ``transfer_ms`` is the time spent receiving the last payload once its header had arrived.
    """

    def __init__(self, sock: socket.socket, limit: int = MAX_PAYLOAD) -> None:
        self.sock = sock
        self.limit = limit
        self.buffer = bytearray(min(limit, MAX_CONTROL_PAYLOAD))
        self.transfer_ms = 0.0

    def read(self) -> tuple[bytes, memoryview]:
        """Return the next (type, payload) message, blocking until it is complete."""
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


def parse_json(payload: bytes | memoryview) -> dict[str, Any] | None:
    """JSON object in the payload, or None if it is not a valid JSON object."""
    try:
        obj = json.loads(bytes(payload))
    except ValueError:
        return None
    return cast(dict[str, Any], obj) if isinstance(obj, dict) else None


def parse_frame(payload: bytes | memoryview) -> tuple[int, int, np.ndarray, int]:
    """Return (seq, ts_ns, image, color) of a FRAM payload, the image points into the payload buffer."""
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
