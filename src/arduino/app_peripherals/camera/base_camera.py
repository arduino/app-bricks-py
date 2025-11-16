# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from abc import ABC, abstractmethod
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
    ):
        """
        Initialize the camera base.

        Args:
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int): Frames per second to capture from the camera.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None
        """
        self.resolution = resolution
        self.fps = fps
        self.adjustments = adjustments
        self.logger = logger  # This will be overridden by subclasses if needed

        self._camera_lock = threading.Lock()
        self._is_started = False
        self._last_capture_time = time.monotonic()
        self._desired_interval = 1.0 / fps if fps > 0 else 0

    def start(self) -> None:
        """Start the camera capture."""
        with self._camera_lock:
            if self._is_started:
                return

            try:
                self._open_camera()
                self._is_started = True
                self._last_capture_time = time.monotonic()
                self.logger.info(f"Successfully opened {self.__class__.__name__}")
            except Exception as e:
                raise CameraOpenError(f"Failed to open camera: {e}")

    def stop(self) -> None:
        """Stop the camera and release resources."""
        with self._camera_lock:
            if not self._is_started:
                return

            try:
                self._close_camera()
                self._is_started = False
                self.logger.info(f"Successfully closed {self.__class__.__name__}")
            except Exception as e:
                self.logger.warning(f"Failed to close camera: {e}")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera, respecting the configured FPS.

        Returns:
            Numpy array or None if no frame is available.

        Raises:
            CameraReadError: If the camera is not started.
        """
        if not self.is_started():
            raise CameraReadError(f"Attempted to read from {self.__class__.__name__} before starting it.")

        with self._camera_lock:
            if not self.is_started():
                raise CameraReadError(f"Attempted to read from {self.__class__.__name__} before starting it.")

            # Apply FPS throttling
            if self._desired_interval > 0:
                current_time = time.monotonic()
                elapsed = current_time - self._last_capture_time
                if elapsed < self._desired_interval:
                    time.sleep(self._desired_interval - elapsed)

            self._last_capture_time = time.monotonic()

            frame = self._read_frame()
            if frame is None:
                return None

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
            raise CameraReadError(f"Attempted to acquire stream from {self.__class__.__name__} before starting it.")

        while self.is_started():
            frame = self.capture()
            if frame is not None:
                yield frame
            else:
                # Avoid busy-waiting if no frame available
                time.sleep(0.001)

    def is_started(self) -> bool:
        """Check if the camera is started."""
        return self._is_started

    @abstractmethod
    def _open_camera(self) -> None:
        """Open the camera connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _close_camera(self) -> None:
        """Close the camera connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera. Must be implemented by subclasses."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
