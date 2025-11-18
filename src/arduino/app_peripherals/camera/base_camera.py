# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable
import numpy as np

from arduino.app_utils import Logger

from .errors import CameraOpenError, CameraReadError, CameraTransformError

logger = Logger("Camera")


class BaseCamera(ABC):
    """
    Abstract base class for camera implementations.

    This class defines the common interface that all camera implementations must follow,
    providing a unified API regardless of the underlying camera protocol or type.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize the camera base.

        Args:
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int): Frames per second to capture from the camera.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        self.resolution = resolution
        self.fps = fps
        self.adjustments = adjustments
        self.logger = logger  # This will be overridden by subclasses if needed
        self.name = self.__class__.__name__  # This will be overridden by subclasses if needed

        self._camera_lock = threading.Lock()
        self._is_started = False
        self._last_capture_time = time.monotonic()
        self._desired_interval = 1.0 / fps if fps > 0 else 0

        # Auto-reconnection parameters
        self.auto_reconnect = auto_reconnect
        self.auto_reconnect_delay = 1.0
        self.first_connection_max_retries = 10

        # Stream interruption detection
        self._consecutive_none_frames = 0
        self._stream_paused = False

        # Event handling
        self._on_event_cb: Callable[[str, dict], None] | None = None
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CameraEvent")
    
    @property
    def _none_frame_threshold(self) -> int:
        """Heuristic: 750ms of empty frames based on current fps."""
        return int(0.75 * self.fps) if self.fps > 0 else 10

    def start(self) -> None:
        """
        Start the camera capture with retries, if enabled.

        Raises:
            CameraOpenError: If the camera fails to start after the retries.
            Exception: If the underlying implementation fails to start the camera.
        """
        with self._camera_lock:
            self.logger.info("Starting camera...")

            attempt = 0
            while not self.is_started():
                try:
                    self._open_camera()
                    self._is_started = True
                    self._last_capture_time = time.monotonic()
                    self.logger.info(f"Successfully started {self.name}")
                except Exception as e:
                    if not self.auto_reconnect:
                        raise
                    attempt += 1
                    if attempt >= self.first_connection_max_retries:
                        raise CameraOpenError(
                            f"Failed to start camera {self.name} after {self.first_connection_max_retries} attempts, last error is: {e}"
                        )

                    delay = min(self.auto_reconnect_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff
                    self.logger.warning(
                        f"Failed to start camera {self.name} (attempt {attempt}/{self.first_connection_max_retries}). Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

    def stop(self) -> None:
        """Stop the camera and release resources."""
        with self._camera_lock:
            if not self.is_started():
                return

            self.logger.info("Stopping camera...")

            try:
                self._close_camera()
                self._event_executor.shutdown()
                self._is_started = False
                self.logger.info(f"Successfully stopped {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to stop camera: {e}")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera, respecting the configured FPS.

        Returns:
            Numpy array or None if no frame is available.

        Raises:
            CameraReadError: If the camera is not started.
            Exception: If the underlying implementation fails to read a frame.
        """
        with self._camera_lock:
            if not self.is_started():
                raise CameraReadError(f"Attempted to read from {self.name} before starting it.")

            # Apply FPS throttling
            if self._desired_interval > 0:
                current_time = time.monotonic()
                elapsed = current_time - self._last_capture_time
                if elapsed < self._desired_interval:
                    time.sleep(self._desired_interval - elapsed)

            self._last_capture_time = time.monotonic()

            frame = self._read_frame()
            if frame is None:
                self._consecutive_none_frames += 1
                if self._consecutive_none_frames >= self._none_frame_threshold and not self._stream_paused:
                    self._stream_paused = True
                    self._emit_event("paused")
                return None

            if self._stream_paused:
                self._stream_paused = False
                self._emit_event("resumed")

            self._consecutive_none_frames = 0

            if self.adjustments is not None:
                try:
                    frame = self.adjustments(frame)
                except Exception as e:
                    raise CameraTransformError(f"Frame transformation failed ({self.adjustments}): {e}")

            return frame

    def stream(self):
        """
        Continuously capture frames from the camera.

        This is a generator that yields frames continuously while the camera is started.
        Built on top of capture() for convenience.

        Yields:
            np.ndarray: Video frames as numpy arrays.
        """
        if not self.is_started():
            raise CameraReadError(f"Attempted to acquire stream from {self.name} before starting it.")

        while self.is_started():
            frame = self.capture()
            if frame is not None:
                yield frame
            else:
                # Avoid busy-waiting if no frame available
                time.sleep(0.001)

    def is_started(self) -> bool:
        """Check if the camera has been started."""
        return self._is_started

    def on_event(self, callback: Callable[[str, dict | None], None] | None):
        """Registers or removes a callback to be triggered on camera lifecycle events.

        When a camera lifecycle event will happen, the provided callback function will be invoked.
        If None is provided, the callback will be removed.

        Args:
            callback (Callable[[str, dict | None], None]): A callback that will be called every time a camera
                lifecycle event will happen with the event name and any associated data. The event
                names depend on the actual camera implementation being used. Some common events are:
                - 'disconnected': The camera has been disconnected.
                - 'connected': The camera has been reconnected.
                - 'paused': The stream has been paused and is temporarily unavailable.
                - 'resumed': The stream has resumed after being paused.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_event(event: str, data: dict):
                print(f"Camera is now: {event}")
                print(f"Data: {data}")
                # Here you can add your code to react to the event

            camera.on_event(on_event)
        """
        if callback is None:
            self._on_event_cb = None
        else:

            def _callback_wrapper(event: str, data: dict):
                try:
                    callback(event, data)
                except Exception as e:
                    self.logger.error(f"Callback for event '{event}' failed with error: {e}")

            self._on_event_cb = _callback_wrapper

    @abstractmethod
    def _open_camera(self) -> None:
        """
        Open the camera connection.

        Must be implemented by subclasses and events should be emitted accordingly.
        """
        pass

    @abstractmethod
    def _close_camera(self) -> None:
        """
        Close the camera connection.

        Must be implemented by subclasses and events should be emitted accordingly.
        """
        pass

    @abstractmethod
    def _read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.

        Must be implemented by subclasses.
        """
        pass

    def _emit_event(self, event: str, data: dict | None = None) -> None:
        """
        Invoke the registered event callback in the background, if any.

        Args:
            event (str): The name of the event.
            data (dict): Additional data associated with the event.
        """
        if self._on_event_cb is not None:
            self._event_executor.submit(self._on_event_cb, event, data if data is not None else {})

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
