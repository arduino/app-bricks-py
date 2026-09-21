# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Protocol between clients and the inference server (v8, Unix domain socket SOCK_STREAM).

Every message:  [type: 4 ASCII bytes] [payload length: uint32 LE] [payload]

Many clients connect to the same socket. One connection = one model.

    client: OPEN {"model": name}        first message, mandatory
    server: OPND {...}                  once the model is ready; the client waits
            ERR  {...} + close          if the model cannot be opened
    client: FRAM ...                    repeated, one frame at a time
    server: RSLT {...} | ERR {...}      the connection stays open after an ERR too

The model stays available as long as the connection is open. On disconnect
the server releases it; if no other connection is using it and it is not
pinned, the server terminates it.

Messages
  OPEN  C->S  JSON     {"model": name}
  OPND  S->C  JSON     {"model", "project", "width", "height", "channels",
                        "labels", "model_type", "resize_mode"}
  FRAM  C->S  binary   FRAME_HEADER + RGB24 pixels at the model resolution
  RSLT  S->C  JSON     {"seq", "ts_ns", "boxes", "classes", "anomaly", "timing_ms"}
  ERR   S->C  JSON     {"op": "open"|"frame", "code", "error", ...}
"""

import json
import struct
import time

import numpy as np

SOCKET_NAME = "ei.sock"
PROTOCOL_VERSION = 8

OPEN = b"OPEN"
OPENED = b"OPND"
FRAME = b"FRAM"
RESULT = b"RSLT"
ERROR = b"ERR "

# Open errors: the server closes the connection after sending them
E_BAD_REQUEST = "bad_request"  # invalid first message
E_TOO_MANY_CLIENTS = "too_many_clients"  # --max-clients reached
E_UNKNOWN_MODEL = "unknown_model"  # no <name>.eim file
E_TOO_MANY_MODELS = "too_many_models"  # --max-models reached
E_LOW_MEMORY = "low_memory"  # available memory below --memory-reserve-mb
E_LOAD_FAILED = "load_failed"  # the .eim file does not start
# Frame errors: the connection stays open
E_BAD_FRAME = "bad_frame"  # invalid size or format
E_INTERNAL = "internal"  # error during inference

HEADER = struct.Struct("<4sI")  # type, payload length
FRAME_HEADER = struct.Struct("<QqHHB3x")  # seq, ts_ns, width, height, channels
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


# ------------------------------------------------------------------ sending
def send_json(sock, kind: bytes, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(HEADER.pack(kind, len(data)) + data)


def send_frame(sock, seq: int, ts_ns: int, rgb: np.ndarray) -> None:
    """Send an HxWx3 uint8 image without copying its pixels into a new buffer."""
    h, w, c = rgb.shape
    pixels = memoryview(np.ascontiguousarray(rgb)).cast("B")
    head = FRAME_HEADER.pack(seq, ts_ns, w, h, c)
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
    """Return (seq, ts_ns, image). The image points into the Reader buffer."""
    if len(payload) < FRAME_HEADER.size:
        raise ProtocolError("frame too short")
    seq, ts_ns, w, h, c = FRAME_HEADER.unpack_from(payload)
    pixels = payload[FRAME_HEADER.size :]
    if len(pixels) != w * h * c:
        raise ProtocolError(f"expected {w * h * c} pixel bytes, received {len(pixels)}")
    return seq, ts_ns, np.frombuffer(pixels, np.uint8).reshape(h, w, c)
