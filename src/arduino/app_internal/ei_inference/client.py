# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Client of the inference service: one connection, one model.

detector = InferenceClient("detector", confidence=0.5)   # only the boxes reaching 0.5 come back
result = detector.infer(frame)      # synchronous, the frame as captured, boxes in its coordinates
detector.submit(frame)              # streaming: non-blocking, skipped while the slots are taken
result = detector.latest
detector.close()                    # the service releases the model

The service says how many frames the connection may keep in flight, its slots: one, or more when it
runs several instances of the model. The client only follows: `wait_idle` returns when a slot is free.
A model with the object tracking block reports the objects it follows in `Result.tracks`, boxes with an
`id`. `set_confidence` changes what the connection receives, `configure` sets the other threshold values of
the model, the blocks `thresholds` lists.
"""

import collections
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import protocol as P

# The service creates its socket in the app .cache directory, mounted at /ipc in its container
DEFAULT_SOCKET_PATH = os.getenv("EI_INFERENCE_SOCKET_PATH", "/app/.cache/edge_impulse/ei.sock")
# The models directory of the service: the out-of-the-box models of the image under ootb/ei, the ones the app CLI
# installs mounted under edge-impulse and custom-ei. Model names are paths relative to it, without the extension.
IMAGE_MODELS_DIR = "/models"
DEFAULT_MODELS_DIR = os.getenv("MODELS_PATH", "/var/lib/arduino-app-cli/models")


def model_name_from_path(model_path: str, models_dirs: tuple[str, ...] = (DEFAULT_MODELS_DIR, IMAGE_MODELS_DIR)) -> str:
    """The service name of an ``.eim`` file, its path relative to the models directory holding it.

    ``/var/lib/arduino-app-cli/models/custom-ei/abc/model.eim`` and ``/models/ootb/ei/yolo-x-nano.eim`` become
    ``custom-ei/abc/model`` and ``ootb/ei/yolo-x-nano``.

    Args:
        model_path (str): Path of the ``.eim`` file as the app CLI configures it, on the host or in the image.
        models_dirs (tuple[str, ...]): The directories the service serves models from, the app CLI models directory
            (MODELS_PATH variable) and the models directory of the image.

    Returns:
        str: The model name to open on the service.

    Raises:
        ValueError: If the path is not an ``.eim`` file under one of the directories.
    """
    for models_dir in models_dirs:
        relative = os.path.relpath(os.path.normpath(model_path), os.path.normpath(models_dir))
        if not relative.startswith("..") and not os.path.isabs(relative) and relative.endswith(".eim"):
            return relative[: -len(".eim")].replace(os.sep, "/")
    raise ValueError(f"'{model_path}' is not a .eim file under {' or '.join(models_dirs)}: the inference service only loads models from there")


class ServerError(Exception):
    """Error returned by the service, ``code`` is one of the protocol error codes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Box:
    """A detected object, in the coordinates of the submitted frame; a tracked one carries the ``id`` the model
    keeps for it while it stays in view."""

    label: str
    score: float
    x: float
    y: float
    w: float
    h: float
    id: int | None = None


@dataclass
class Result:
    """The outcome of one inference, an error of the frame or the boxes, tracks, classes and timings."""

    model: str
    seq: int
    ts_ns: int  # timestamp of the submitted frame (CLOCK_MONOTONIC)
    source_size: tuple[int, int]  # (w, h) of the image the boxes refer to
    boxes: list[Box] = field(default_factory=list)
    tracks: list[Box] = field(default_factory=list)  # the boxes with their ids, only from a model with the object tracking block
    classes: dict[str, float] = field(default_factory=dict)
    anomaly: float = 0.0
    timing_ms: dict[str, float] = field(default_factory=dict)
    frame: np.ndarray | None = None  # copy of the submitted image, only with keep_frame=True
    error: str | None = None
    error_code: str | None = None

    @property
    def ok(self) -> bool:
        """True when the frame was processed."""
        return self.error is None

    def top_class(self) -> tuple[str, float] | None:
        """(label, score) of the most likely class, or None."""
        return max(self.classes.items(), key=lambda kv: kv[1]) if self.classes else None


@dataclass
class _InFlight:
    seq: int
    ts_ns: int = 0
    source_size: tuple[int, int] = (0, 0)
    frame: np.ndarray | None = None


class InferenceClient:
    """A connection to the inference service holding one model, thread-safe, as many frames in flight as the service allows."""

    def __init__(
        self, model: str, socket_path: str = DEFAULT_SOCKET_PATH, open_timeout: float | None = None, confidence: float | None = None
    ) -> None:
        """Connect and request the model, blocking until the service has loaded it.

        Args:
            model (str): Model name, the ``.eim`` path relative to the models directory without extension.
            socket_path (str): Path of the service socket.
            open_timeout (float | None): Seconds to wait for the model to load, None waits indefinitely.
            confidence (float | None): Score the boxes, tracks and classes must reach to be returned, between 0 and 1;
                None returns everything the model reports. The model's own score threshold follows the lowest
                confidence of the connections using it.

        Raises:
            ServerError: If the service refuses the model (unknown_model, too_many_models, low_memory, load_failed,
                too_many_clients, bad_request).
            OSError: If the socket cannot be reached.
            TimeoutError: If the model is not ready within open_timeout.
        """
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(socket_path)
            self._reader = P.Reader(self.sock)
            P.send_json(self.sock, P.OPEN, {"model": model, **({} if confidence is None else {"confidence": confidence})})
            self.sock.settimeout(open_timeout)
            kind, payload = self._reader.read()
            self.sock.settimeout(None)
            data = P.parse_json(payload)
            if data is None:
                raise P.ProtocolError(f"invalid reply to OPEN: {kind!r}")
            if kind == P.ERROR:
                raise ServerError(data["code"], data["error"])
            if kind != P.OPENED:
                raise P.ProtocolError(f"expected OPND, received {kind!r}")
        except BaseException:
            self.sock.close()
            raise

        self.info: dict[str, Any] = data  # model details (OPND)
        self.model: str = data["model"]
        self.labels: list[str] = data["labels"]
        self.input_size: tuple[int, int] = (data["width"], data["height"])
        self.resize_mode: str = data["resize_mode"]  # how the service fits the frames into the model input
        self.object_tracking: bool = bool(data.get("object_tracking", False))  # the results carry the tracked objects
        self.thresholds: list[dict[str, Any]] = list(data.get("thresholds", []))  # the threshold blocks, with their current values
        self.confidence: float | None = data.get("confidence")  # what the results must reach for this connection
        self.slots: int = int(data.get("slots", 1))  # frames the service lets this connection keep in flight
        self._send_lock = threading.Lock()
        self._configure_lock = threading.Lock()  # one configuration at a time, the reply carries no id
        self._configured: dict[str, Any] | ServerError | None = None  # the reply to the configuration in progress
        self._cond = threading.Condition()
        self._in_flight: dict[int, _InFlight] = {}  # by seq, in submission order
        self._latest: Result | None = None
        self._unread: collections.deque[Result] = collections.deque()
        self._seq = 0
        self.closed = False
        self.sent = self.received = self.skipped = 0
        self.round_trip_total_ms = 0.0  # from submit to the parsed reply, summed over the received results
        threading.Thread(target=self._receive_loop, daemon=True, name="ei-inference-receive").start()

    @property
    def busy(self) -> bool:
        """True while every slot holds a frame in flight."""
        return len(self._in_flight) >= self.slots

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until a slot is free, the moment to capture the next frame.

        Args:
            timeout (float | None): Seconds to wait, None waits indefinitely.

        Returns:
            bool: True when a frame can be submitted, False when every slot is still taken after the timeout.

        Raises:
            ConnectionError: If the connection is closed.
        """
        with self._cond:
            free = self._cond.wait_for(lambda: len(self._in_flight) < self.slots or self.closed, timeout)
            if self.closed:
                raise ConnectionError("connection closed")
            return free

    def submit(self, image: np.ndarray, ts_ns: int | None = None, color: str = "bgr", keep_frame: bool = False) -> int | None:
        """Send the image if a slot is free, otherwise skip it and return None.

        The frame goes out as it is, the service resizes it to the model input and returns the boxes in its coordinates.

        Args:
            image (np.ndarray): HxWx3 uint8 frame, up to 1920x1080.
            ts_ns (int | None): Timestamp of the frame, monotonic nanoseconds, now when None.
            color (str): Channel order of the frame, "bgr" (OpenCV) or "rgb".
            keep_frame (bool): Keep a copy of the frame in the Result.

        Returns:
            int | None: The sequence number of the frame, None when skipped.

        Raises:
            ConnectionError: If the connection is closed.
        """
        with self._cond:
            if self.closed:
                raise ConnectionError("connection closed")
            if len(self._in_flight) >= self.slots:
                self.skipped += 1
                return None
            self._seq += 1
            pending = _InFlight(self._seq)
            self._in_flight[pending.seq] = pending  # reserve the slot

        # The reply cannot arrive before the frame is sent: finish without the lock
        pending.ts_ns = ts_ns if ts_ns is not None else time.monotonic_ns()
        pending.source_size = (image.shape[1], image.shape[0])
        if keep_frame:
            pending.frame = image.copy()
        try:
            with self._send_lock:
                P.send_frame(self.sock, pending.seq, pending.ts_ns, image, P.BGR if color == "bgr" else P.RGB)
        except OSError as exc:
            self.close()
            raise ConnectionError("send failed") from exc
        self.sent += 1
        return pending.seq

    def infer(self, image: np.ndarray, color: str = "bgr", keep_frame: bool = False, timeout: float | None = 30.0) -> Result:
        """Wait for a free slot, send the image and wait for its result.

        Args:
            image (np.ndarray): HxWx3 uint8 frame.
            color (str): Channel order of the frame, "bgr" (OpenCV) or "rgb".
            keep_frame (bool): Keep a copy of the frame in the Result.
            timeout (float | None): Seconds to wait overall, None waits indefinitely.

        Returns:
            Result: The result of this frame, ``ok`` is False when the service reported a frame error.

        Raises:
            TimeoutError: If the model stays busy or the reply does not arrive in time.
            ConnectionError: If the connection is closed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if not self.wait_idle(_remaining(deadline)):
                raise TimeoutError(f"'{self.model}' is busy")
            seq = self.submit(image, color=color, keep_frame=keep_frame)
            if seq is not None:
                break  # otherwise another thread got there first
        with self._cond:
            done = self._cond.wait_for(lambda: seq not in self._in_flight or self.closed, _remaining(deadline))
            if not done:
                raise TimeoutError(f"no reply from '{self.model}'")
            result = next((r for r in self._unread if r.seq == seq), None)
            if result is None:
                if self._latest is not None and self._latest.seq == seq:
                    return self._latest
                raise ConnectionError("connection closed")
            self._unread.remove(result)
            return result

    @property
    def latest(self) -> Result | None:
        """The last received result, even if already read."""
        return self._latest

    def threshold_block(self, kind: str) -> dict[str, Any] | None:
        """The threshold block of the given ``type`` ("object_detection", "object_tracking"...), None if the model has none."""
        return next((block for block in self.thresholds if block.get("type") == kind), None)

    def set_confidence(self, confidence: float | None, timeout: float | None = 10.0) -> None:
        """Change the score the boxes, tracks and classes must reach for this connection, None for everything.

        Args:
            confidence (float | None): The new confidence, between 0 and 1.
            timeout (float | None): Seconds to wait for the service to apply it, None waits indefinitely.

        Raises:
            ServerError: If the service refuses it (bad_request).
            TimeoutError: If the service does not answer in time.
            ConnectionError: If the connection is closed.
        """
        self._send_configuration({"confidence": confidence}, timeout)

    def configure(self, block_id: int, timeout: float | None = 10.0, **values: float) -> list[dict[str, Any]]:
        """Set threshold values of one block of the model, for every connection using it.

        The score threshold is not one of them: it follows the confidence of the connections, see ``set_confidence``.

        Args:
            block_id (int): The ``id`` of the block in ``thresholds``.
            timeout (float | None): Seconds to wait for the service to apply them, None waits indefinitely.
            **values (float): The values to set, by the keys of the block, e.g. ``max_age=3``.

        Returns:
            list[dict]: The blocks with their current values, also kept in ``thresholds``.

        Raises:
            ServerError: If the service refuses them (bad_request: unknown block or key, internal).
            TimeoutError: If the service does not answer in time.
            ConnectionError: If the connection is closed.
        """
        return self._send_configuration({"id": block_id, **values}, timeout)["thresholds"]

    def _send_configuration(self, values: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        """Send a CONF message and wait for its reply, keeping the thresholds and the confidence it reports."""
        with self._configure_lock:
            with self._cond:
                if self.closed:
                    raise ConnectionError("connection closed")
                self._configured = None
            try:
                with self._send_lock:
                    P.send_json(self.sock, P.CONFIGURE, values)
            except OSError as exc:
                self.close()
                raise ConnectionError("send failed") from exc
            with self._cond:
                if not self._cond.wait_for(lambda: self._configured is not None or self.closed, timeout):
                    raise TimeoutError(f"no reply from '{self.model}' to the configuration")
                reply, self._configured = self._configured, None
                if reply is None:
                    raise ConnectionError("connection closed")
                if isinstance(reply, ServerError):
                    raise reply
                self.thresholds = list(reply.get("thresholds", self.thresholds))
                self.confidence = reply.get("confidence")
                return reply

    def get_result(self, timeout: float | None = 0) -> Result | None:
        """The oldest result not read yet, timeout=0 does not wait and None waits indefinitely.

        Args:
            timeout (float | None): Seconds to wait for a result.

        Returns:
            Result | None: The unread result, None when there is none within the timeout.

        Raises:
            ConnectionError: If the connection is closed and no result is pending.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._unread or self.closed, timeout)
            result = self._unread.popleft() if self._unread else None
            if result is None and self.closed:
                raise ConnectionError("connection closed")
            return result

    def _receive_loop(self) -> None:
        try:
            while True:
                kind, payload = self._reader.read()
                data = P.parse_json(payload)
                if data is None:
                    raise P.ProtocolError(f"unexpected reply: {kind!r}")
                with self._cond:
                    if "slots" in data:
                        self.slots = max(1, int(data["slots"]))
                    if kind == P.SLOTS:
                        self._cond.notify_all()
                        continue
                    if kind == P.CONFIGURE or (kind == P.ERROR and data.get("op") == "configure"):
                        self._configured = ServerError(data["code"], data["error"]) if kind == P.ERROR else data
                        self._cond.notify_all()
                        continue
                    # An ERR without seq refers to the oldest frame in flight (unparsable header)
                    seq = data.get("seq", next(iter(self._in_flight), None))
                    pending = self._in_flight.pop(seq, None)
                    if pending is None:
                        raise P.ProtocolError(f"unexpected reply: {data}")
                    self._latest = self._to_result(kind, data, pending)
                    self._unread.append(self._latest)
                    self.received += 1
                    self.round_trip_total_ms += (time.monotonic_ns() - pending.ts_ns) / 1e6
                    self._cond.notify_all()
        except (ConnectionError, OSError, P.ProtocolError, ValueError):
            pass
        finally:
            with self._cond:
                self.closed = True
                self._cond.notify_all()

    def _to_result(self, kind: bytes, data: dict[str, Any], pending: _InFlight) -> Result:
        result = Result(model=self.model, seq=pending.seq, ts_ns=pending.ts_ns, source_size=pending.source_size, frame=pending.frame)
        if kind != P.RESULT:
            result.error, result.error_code = data.get("error"), data.get("code")
            return result
        result.boxes = [Box(b["label"], b["score"], b["x"], b["y"], b["w"], b["h"]) for b in data["boxes"]]
        result.tracks = [Box(t["label"], t["score"], t["x"], t["y"], t["w"], t["h"], int(t["id"])) for t in data.get("tracks", [])]
        result.classes = data["classes"]
        result.anomaly = data["anomaly"]
        result.timing_ms = data["timing_ms"]
        return result

    def close(self) -> None:
        """Close the connection: the service releases the model, and terminates it if nobody else uses it."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)  # wakes up the receiving thread
        except OSError:
            pass
        self.sock.close()

    def __enter__(self) -> "InferenceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())
