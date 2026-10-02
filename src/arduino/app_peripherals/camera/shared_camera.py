# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from .base_camera import BaseCamera
from .errors import CameraConfigError, CameraReadError


class _Cursor(threading.local):
    """Per-thread position in the shared frame sequence."""

    seq: int = 0
    """Sequence number of the last frame handed to this thread, 0 before its first capture."""


class SharedCamera(BaseCamera):
    """
    Camera wrapper that shares a single camera between several consumers.

    Every consumer, i.e. every thread calling ``capture()`` (directly or through ``stream()``),
    receives every frame read from the source camera exactly once. This allows passing the same
    camera instance to multiple bricks, for example a code detector and an object detector.

    The source is read on demand by whichever consumer asks first for a new frame, so the frame
    rate is the one of the source camera and no extra thread is used. A slower consumer is never
    left behind: it receives the latest frame and skips the ones it was too slow to see. A frame
    older than one source frame interval is never handed out, a new one is read instead.

    All consumers receive the very same frame object, which is therefore read-only: a consumer
    that needs to modify it in place must copy it first.

    The source is started when the first user calls ``start()`` and stopped when the last one
    calls ``stop()``, so a brick stopping does not stop the camera for the others. The lifecycle
    of the source must be managed through the shared camera only.

    Note: consumers are told apart by thread. Two consumers reading from the same thread would
    split the frames between them.

    Example:
        ```python
        from arduino.app_peripherals.camera import Camera, SharedCamera
        from arduino.app_bricks.camera_code_detection import CameraCodeDetection
        from arduino.app_bricks.video_objectdetection import VideoObjectDetection

        camera = SharedCamera(Camera())
        code_detection = CameraCodeDetection(camera=camera)
        object_detection = VideoObjectDetection(camera=camera)
        ```
    """

    def __init__(self, source: BaseCamera) -> None:
        """
        Initialize the shared camera.

        Args:
            source (BaseCamera): The camera to share. Its resolution, FPS and adjustments apply
                to every consumer, and its adjustments are applied only once per frame.

        Raises:
            CameraConfigError: If source is not a BaseCamera or is itself a SharedCamera.
        """
        if not isinstance(source, BaseCamera):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise CameraConfigError(f"source must be a BaseCamera, got {type(source).__name__}")
        if isinstance(source, SharedCamera):
            raise CameraConfigError("source is already a SharedCamera and can be used directly")

        # Set ahead of the base initializer, which assigns resolution and fps through the
        # properties below
        self._source = source
        super().__init__(resolution=source.resolution, fps=source.fps, auto_reconnect=False)
        self.name = f"{self.__class__.__name__}({source.name})"
        self.logger = source.logger

        self._users = 0
        self._frame_cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._frame_time = 0.0
        self._producing = False
        self._cursor = _Cursor()

    @property
    def source(self) -> BaseCamera:
        """The shared camera."""
        return self._source

    @property
    def resolution(self) -> tuple[int, int]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Resolution of the source camera, as negotiated with the device once started."""
        return self._source.resolution

    @resolution.setter
    def resolution(self, value: tuple[int, int]) -> None:
        self._source.resolution = value

    @property
    def fps(self) -> int:
        """Frames per second of the source camera, as negotiated with the device once started."""
        return self._source.fps

    @fps.setter
    def fps(self, value: int) -> None:
        self._source.fps = value

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Status of the source camera."""
        return self._source.status

    def start(self) -> None:
        """
        Register a user of the shared camera, starting the source camera for the first one.

        Raises:
            CameraOpenError: If the source camera fails to start after the retries.
            Exception: If the source camera implementation fails to start.
        """
        with self._camera_lock:
            self._source.start()
            self._users += 1

    def stop(self) -> None:
        """Unregister a user of the shared camera, stopping the source camera after the last one."""
        with self._camera_lock:
            if self._users == 0:
                return
            self._users -= 1
            if self._users > 0:
                return
            self._source.stop()

        # Wake up the consumers waiting for a frame, so they notice the camera is stopped
        with self._frame_cond:
            self._frame = None
            self._frame_cond.notify_all()

    def is_started(self) -> bool:
        """Check if the source camera has been started."""
        return self._source.is_started()

    def capture(self) -> np.ndarray | None:
        """
        Capture the next frame for the calling thread.

        Returns the latest frame read from the source if this thread has not received it yet and
        it is not older than one source frame interval. Otherwise waits for the next frame,
        reading it from the source if no other consumer is already doing it.

        Returns:
            Read-only numpy array, or None if no frame is available.

        Raises:
            CameraReadError: If the camera is not started.
            Exception: If the source camera fails to read a frame.
        """
        with self._frame_cond:
            while True:
                if not self.is_started():
                    raise CameraReadError(f"Attempted to read from {self.name} before starting it.")

                if self._frame is not None and self._cursor.seq < self._frame_seq and self._is_fresh():
                    self._cursor.seq = self._frame_seq
                    return self._frame
                if not self._producing:
                    break
                self._frame_cond.wait()

            self._producing = True

        # The source is read without holding the condition: it blocks for up to a frame interval
        frame = None
        try:
            frame = self._source.capture()
        finally:
            with self._frame_cond:
                self._producing = False
                if frame is not None:
                    frame.flags.writeable = False
                    self._frame = frame
                    self._frame_seq += 1
                    self._frame_time = time.monotonic()
                    self._cursor.seq = self._frame_seq
                self._frame_cond.notify_all()

        return frame

    def _is_fresh(self) -> bool:
        """Whether the latest frame is not older than one source frame interval."""
        return time.monotonic() - self._frame_time < 1.0 / self._source.fps

    def on_status_changed(self, callback: Callable[[str, dict[str, Any]], None] | None) -> None:
        """Registers or removes a callback to be triggered on the source camera lifecycle events.

        This replaces any callback registered directly on the source camera.

        Args:
            callback (Callable[[str, dict], None]): A callback that will be called every time the
                source camera status changes, see ``BaseCamera.on_status_changed()``.
            callback (None): To unregister the current callback, if any.
        """
        self._source.on_status_changed(callback)

    def _open_camera(self) -> None:
        self._source.start()

    def _close_camera(self) -> None:
        self._source.stop()

    def _read_frame(self) -> np.ndarray | None:
        return self._source.capture()
