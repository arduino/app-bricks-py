# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
import time
from typing import Optional
import cv2
import numpy as np
from collections.abc import Callable

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraOpenError, CameraReadError

logger = Logger("V4LCamera")


class V4LCamera(BaseCamera):
    """
    V4L (Video4Linux) camera implementation for USB and local cameras.

    This class handles USB cameras and other V4L-compatible devices on Linux systems.
    It supports both device indices and device paths.
    """

    def __init__(
        self,
        device: str | int = 0,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize V4L camera.

        Args:
            device: Camera identifier - can be:
                   - int: Camera index (e.g., 0, 1)
                   - str: Camera index as string or device path
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int, optional): Frames per second to capture from the camera. Default: 10.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        super().__init__(resolution, fps, adjustments)
        self.device = self._resolve_camera_id(device)
        self.logger = logger

        self._cap = None

        # Auto-reconnection parameters
        self.reconnect_delay = 1.0
        self.reconnect_max_retries = 5
        self._auto_reconnect = auto_reconnect
        self._last_reconnect_attempt = 0.0

    def _resolve_camera_id(self, device: str | int) -> int:
        """
        Resolve camera identifier to a numeric device ID.

        Args:
            device: Camera identifier

        Returns:
            Numeric camera device ID

        Raises:
            CameraOpenError: If camera cannot be resolved
        """
        if isinstance(device, int):
            return device

        if isinstance(device, str):
            # If it's a numeric string, convert directly
            if device.isdigit():
                device_idx = int(device)
                # Validate using device index mapping
                video_devices = self._get_video_devices_by_index()
                if device_idx in video_devices:
                    return int(video_devices[device_idx])
                else:
                    # Fallback to direct device ID if mapping not available
                    return device_idx

            # If it's a device path like "/dev/video0"
            if device.startswith("/dev/video"):
                return int(device.replace("/dev/video", ""))

        raise CameraOpenError(f"Cannot resolve camera identifier: {device}")

    def _get_video_devices_by_index(self) -> dict[int, str]:
        """
        Map camera indices to device numbers by reading /dev/v4l/by-id/.

        Returns:
            Dict mapping index to device number
        """
        devices_by_index = {}
        directory_path = "/dev/v4l/by-id/"

        # Check if the directory exists
        if not os.path.exists(directory_path):
            logger.warning(f"Directory '{directory_path}' not found.")
            return devices_by_index

        try:
            entries = os.listdir(directory_path)
            for entry in entries:
                full_path = os.path.join(directory_path, entry)

                if os.path.islink(full_path):
                    # Find numeric index at end of filename
                    match = re.search(r"index(\d+)$", entry)
                    if match:
                        try:
                            index = int(match.group(1))
                            resolved_path = os.path.realpath(full_path)
                            device_name = os.path.basename(resolved_path)
                            device_number = device_name.replace("video", "")
                            devices_by_index[index] = device_number
                        except ValueError:
                            logger.warning(f"Could not parse index from '{entry}'")
                            continue
        except OSError as e:
            logger.error(f"Error accessing directory '{directory_path}': {e}")

        return devices_by_index

    def _open_camera(self) -> None:
        """
        Open the V4L camera connection with retry logic.

        Retries with exponential backoff until successful or self.max_retries is reached.
        """
        attempt = 0

        while not self._connect():
            if not self._auto_reconnect:
                raise CameraOpenError(f"VideoCapture returned unopened state for device {self.device}")
            if attempt >= self.reconnect_max_retries:
                raise CameraOpenError(f"Unable to open camera {self.device} after {self.reconnect_max_retries} attempts")

            delay = self.reconnect_delay * (2 ** min(attempt, 5))  # Cap exponential backoff at 32s
            logger.warning(f"Failed to open camera {self.device} (attempt {attempt + 1}/{self.reconnect_max_retries}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
            attempt += 1

    def _close_camera(self) -> None:
        """Close the V4L camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _connect(self) -> bool:
        """
        Attempt to connect to the camera.

        Returns:
            bool: True if reconnection successful, False otherwise
        """
        current_time = time.time()

        # Prevent too frequent connection attempts
        if current_time - self._last_reconnect_attempt < self.reconnect_delay:
            return False

        self._last_reconnect_attempt = current_time

        self._close_camera()

        try:
            self._cap = cv2.VideoCapture(self.device)
            if not self._cap.isOpened():
                raise CameraOpenError(f"Failed to open camera {self.device}")

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

            if self.resolution and self.resolution[0] and self.resolution[1]:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

                # Verify resolution setting
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                    logger.warning(
                        f"Camera {self.device} resolution set to {actual_width}x{actual_height} "
                        f"instead of requested {self.resolution[0]}x{self.resolution[1]}"
                    )
                    self.resolution = (actual_width, actual_height)

            if self.fps:
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)

                actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
                if actual_fps != self.fps:
                    logger.warning(f"Camera {self.device} FPS set to {actual_fps} instead of requested {self.fps}")
                    self.fps = actual_fps

            # Verify connection with a test read
            ret, _ = self._cap.read()
            if not ret:
                raise CameraReadError(f"Read test failed for camera {self.device}")

            return True

        except (CameraOpenError, CameraReadError):
            self._close_camera()
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to camera {self.device}: {e}")
            self._close_camera()
            return False

    def _read_frame(self) -> np.ndarray | None:
        """
        Read a frame from the V4L camera with auto-reconnection on failure.

        Returns:
            np.ndarray | None: Frame data or None if read fails
        """
        if self._cap is None:
            return None

        try:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"Failed to read from V4L camera {self.device}")

                # Attempt auto-reconnection if enabled
                if self._auto_reconnect:
                    if self._connect():
                        # Try reading again after successful reconnect
                        ret, frame = self._cap.read()
                        if ret:
                            logger.info(f"Successfully reconnected to camera {self.device}")
                            return frame

                return None

            return frame

        except Exception as e:
            logger.error(f"Unexpected error reading from camera {self.device}: {e}")

            # Attempt reconnection on unexpected errors
            if self._auto_reconnect:
                if self._connect():
                    # Try reading again after successful reconnect
                    ret, frame = self._cap.read()
                    if ret:
                        logger.info(f"Successfully reconnected to camera {self.device}")
                        return frame

            return None
