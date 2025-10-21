# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
import io
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from PIL import Image
import cv2
import numpy as np

from arduino.app_utils import Logger

from .errors import CameraOpenError

logger = Logger("Camera")


class BaseCamera(ABC):
    """
    Abstract base class for camera implementations.
    
    This class defines the common interface that all camera implementations must follow,
    providing a unified API regardless of the underlying camera protocol or type.
    """

    def __init__(self, resolution: Optional[Tuple[int, int]] = None, fps: int = 10, 
                 compression: bool = False, letterbox: bool = False, **kwargs):
        """
        Initialize the camera base.

        Args:
            resolution: Resolution as (width, height). None uses default resolution.
            fps: Frames per second for the camera.
            compression: Whether to compress captured images to PNG format.
            letterbox: Whether to apply letterboxing to make images square.
            **kwargs: Additional camera-specific parameters.
        """
        self.resolution = resolution
        self.fps = fps
        self.compression = compression
        self.letterbox = letterbox
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

    def capture(self) -> Optional[Image.Image]:
        """
        Capture a frame from the camera, respecting the configured FPS.

        Returns:
            PIL Image or None if no frame is available.
        """
        frame = self._extract_frame()
        if frame is None:
            return None
        
        try:
            if self.compression:
                # Convert to PNG bytes first, then to PIL Image
                success, encoded = cv2.imencode('.png', frame)
                if success:
                    return Image.open(io.BytesIO(encoded.tobytes()))
                else:
                    return None
            else:
                # Convert BGR to RGB for PIL
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame
                return Image.fromarray(rgb_frame)
        except Exception as e:
            logger.exception(f"Error converting frame to PIL Image: {e}")
            return None

    def capture_bytes(self) -> Optional[bytes]:
        """
        Capture a frame and return as bytes.

        Returns:
            Frame as bytes or None if no frame is available.
        """
        frame = self._extract_frame()
        if frame is None:
            return None
        
        if self.compression:
            success, encoded = cv2.imencode('.png', frame)
            return encoded.tobytes() if success else None
        else:
            return frame.tobytes()

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
            
            # Apply post-processing
            if self.letterbox:
                frame = self._letterbox(frame)
            
            return frame

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Apply letterboxing to make the frame square."""
        h, w = frame.shape[:2]
        if w != h:
            size = max(h, w)
            return cv2.copyMakeBorder(
                frame,
                top=(size - h) // 2,
                bottom=(size - h + 1) // 2,
                left=(size - w) // 2,
                right=(size - w + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            )
        return frame

    def is_started(self) -> bool:
        """Check if the camera is started."""
        return self._is_started

    def produce(self) -> Optional[Image.Image]:
        """Alias for capture method for compatibility."""
        return self.capture()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

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
