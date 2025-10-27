# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
import numpy as np

from arduino.app_utils import Logger

from .errors import CameraOpenError, CameraTransformError

logger = Logger("Camera")


class BaseCamera(ABC):
    """
    Abstract base class for camera implementations.
    
    This class defines the common interface that all camera implementations must follow,
    providing a unified API regardless of the underlying camera protocol or type.
    """

    def __init__(self, resolution: Optional[Tuple[int, int]] = (640, 480), fps: int = 10, 
                 adjuster: Optional[Callable[[np.ndarray], np.ndarray]] = None, **kwargs):
        """
        Initialize the camera base.

        Args:
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int): Frames per second for the camera.
            adjuster (callable, optional): Function pipeline to adjust frames that takes a numpy array and returns a numpy array. Default: None
            **kwargs: Additional camera-specific parameters.
        """
        self.resolution = resolution
        self.fps = fps
        self.adjuster = adjuster
        self._is_started = False
        self._cap_lock = threading.Lock()
        self._last_capture_time = time.monotonic()
        self.desired_interval = 1.0 / fps if fps > 0 else 0

    def start(self) -> None:
        """Start the camera capture."""
        with self._cap_lock:
            if self._is_started:
                return
            
            try:
                self._open_camera()
                self._is_started = True
                self._last_capture_time = time.monotonic()
                logger.info(f"Successfully started {self.__class__.__name__}")
            except Exception as e:
                raise CameraOpenError(f"Failed to start camera: {e}")

    def stop(self) -> None:
        """Stop the camera and release resources."""
        with self._cap_lock:
            if not self._is_started:
                return
            
            try:
                self._close_camera()
                self._is_started = False
                logger.info(f"Stopped {self.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera, respecting the configured FPS.

        Returns:
            Numpy array or None if no frame is available.
        """
        frame = self._extract_frame()
        if frame is None:
            return None
        return frame

    def _extract_frame(self) -> Optional[np.ndarray]:
        """Extract a frame with FPS throttling and post-processing."""
        # FPS throttling
        if self.desired_interval > 0:
            current_time = time.monotonic()
            elapsed = current_time - self._last_capture_time
            if elapsed < self.desired_interval:
                time.sleep(self.desired_interval - elapsed)

        with self._cap_lock:
            if not self._is_started:
                return None
            
            frame = self._read_frame()
            if frame is None:
                return None
            
            self._last_capture_time = time.monotonic()
            
            if self.adjuster is not None:
                try:
                    frame = self.adjuster(frame)
                except Exception as e:
                    raise CameraTransformError(f"Frame transformation failed ({self.adjuster}): {e}")
            
            return frame

    def is_started(self) -> bool:
        """Check if the camera is started."""
        return self._is_started

    def produce(self) -> Optional[np.ndarray]:
        """Alias for capture method for compatibility."""
        return self.capture()

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
