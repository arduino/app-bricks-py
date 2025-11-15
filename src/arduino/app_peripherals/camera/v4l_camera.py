# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
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

        self.device_path = self._resolve_stable_path(device)
        self.device_name = self._resolve_name(self.device_path)
        self.logger = logger

        self._cap = None

        # Auto-reconnection parameters
        self.reconnect_delay = 1.0
        self.reconnect_max_retries = 5
        self._auto_reconnect = auto_reconnect
        self._last_reconnect_attempt = 0.0

    def _resolve_stable_path(self, device: str | int) -> str:
        """
        Resolve a camera identifier to a link stable across reconnections.

        Args:
            device: Camera identifier

        Returns:
            str: stable path to the camera device

        Raises:
            CameraOpenError: If camera cannot be resolved
        """
        if isinstance(device, str) and device.startswith("/dev/v4l/by-id"):
            # Already a stable link
            return device
        elif isinstance(device, str) and device.startswith("/dev/v4l/by-path"):
            # A stable link, but not the one we want, resolve to by-id
            if not os.path.exists(device):
                raise CameraOpenError(f"Device path {device} does not exist")
            resolved_path = os.path.realpath(device)
            video_path = resolved_path
        elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # Treat as /dev/video<device>
            dev_num = int(device)
            video_path = f"/dev/video{dev_num}"
        elif isinstance(device, str) and device.startswith("/dev/video"):
            # A device node path
            video_path = device
        else:
            raise CameraOpenError(f"Unrecognized device identifier: {device}")

        # Now map /dev/videoX to a stable link in /dev/v4l/by-id
        by_id_dir = "/dev/v4l/by-id/"
        if not os.path.exists(by_id_dir):
            raise CameraOpenError(f"Directory '{by_id_dir}' not found.")

        try:
            for entry in os.listdir(by_id_dir):
                full_path = os.path.join(by_id_dir, entry)
                if os.path.islink(full_path):
                    target = os.path.realpath(full_path)
                    if target == video_path:
                        return full_path
        except Exception as e:
            raise CameraOpenError(f"Error resolving stable link: {e}")

        raise CameraOpenError(f"No stable link found for device {device} (resolved as {video_path})")

    def _resolve_name(self, stable_path: str) -> str:
        """
        Resolve a human-readable name for the camera whose stable path is provided
        by looking at /sys/class/video4linux/<video>/name. Falls back to the device
        path (/dev/videoX) if no by-id entry exists.

        Args:
            stable_path: camera's stable path

        Returns:
            str: human readable name

        Raises:
            CameraOpenError: If device cannot be resolved at all
        """
        if not isinstance(stable_path, str) or not stable_path.startswith("/dev/v4l/by-id"):
            raise CameraOpenError(f"Invalid stable path provided: {stable_path}")

        if not os.path.exists(stable_path):
            raise CameraOpenError(f"The provided stable path does not exist: {stable_path}")

        target = os.path.realpath(stable_path)
        video_basename = os.path.basename(target)

        # Try sysfs name first (/sys/class/video4linux/<video>/name)
        try:
            sysfs_path = f"/sys/class/video4linux/{video_basename}/name"
            if os.path.exists(sysfs_path):
                with open(sysfs_path, "r", encoding="utf-8", errors="ignore") as f:
                    name = f.read().strip()
                    if name:
                        return name
        except Exception:
            # Ignore and fall through to fallback
            pass

        # As fallback just return /dev/videoX
        return target or stable_path

    def _open_camera(self) -> None:
        """
        Open the V4L camera connection with retry logic.

        Retries with exponential backoff until successful or self.max_retries is reached.
        """
        delay = 0
        attempt = 0

        while not self._safe_connect(delay):
            if not self._auto_reconnect:
                raise CameraOpenError(f"VideoCapture returned unopened state for device {self.device_name}")
            if attempt >= self.reconnect_max_retries:
                raise CameraOpenError(f"Unable to open camera {self.device_name} after {self.reconnect_max_retries} attempts")

            delay = min(self.reconnect_delay * (2**attempt), 30)  # Cap exponential backoff at 30s
            logger.warning(
                f"Failed to open camera {self.device_name} (attempt {attempt + 1}/{self.reconnect_max_retries}). Retrying in {delay:.1f}s..."
            )
            attempt += 1

    def _safe_connect(self, delay: float | None = None) -> bool:
        """
        Attempt to reopen to the camera with delay between attempts.

        Args:
            delay (float | None): Delay in seconds before attempting reconnection.
                If None, uses self.reconnect_delay.
        Returns:
            bool: True if reconnection successful, False otherwise.
        """
        current_time = time.time()

        # Prevent too frequent connection attempts
        if delay is None:
            # If no delay specified, use the default reconnect_delay
            if current_time - self._last_reconnect_attempt < self.reconnect_delay:
                time.sleep(self.reconnect_delay - (current_time - self._last_reconnect_attempt))
        else:
            # If a specific delay is forced, use it
            time.sleep(delay)

        self._last_reconnect_attempt = current_time

        if not os.path.exists(self.device_path):
            self.logger.warning(f"Camera device {self.device_name} not found  at {self.device_path}.")
            return False

        return self._connect()

    def _connect(self) -> bool:
        """
        Attempt to connect to the camera.

        Returns:
            bool: True if reconnection successful, False otherwise.
        """
        self._close_camera()

        try:
            self._cap = cv2.VideoCapture(self.device_path)
            if not self._cap.isOpened():
                raise CameraOpenError(f"Failed to open camera {self.device_name}")

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

            if self.resolution and self.resolution[0] and self.resolution[1]:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

                # Verify resolution setting
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                    logger.warning(
                        f"Camera {self.device_name} resolution set to {actual_width}x{actual_height} "
                        f"instead of requested {self.resolution[0]}x{self.resolution[1]}"
                    )
                    self.resolution = (actual_width, actual_height)

            if self.fps:
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)

                actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
                if actual_fps != self.fps:
                    logger.warning(f"Camera {self.device_name} FPS set to {actual_fps} instead of requested {self.fps}")
                    self.fps = actual_fps

            # Verify camera with a test read
            ret, _ = self._cap.read()
            if not ret:
                raise CameraReadError(f"Read test failed for camera {self.device_name}")

            return True

        except (CameraOpenError, CameraReadError):
            self._close_camera()
            return False
        except Exception as e:
            logger.error(f"Unexpected error opening camera {self.device_name}: {e}")
            self._close_camera()
            return False

    def _close_camera(self) -> None:
        """Close the V4L camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_frame(self) -> np.ndarray | None:
        """
        Read a frame from the V4L camera with auto-reconnection on failure.

        Returns:
            np.ndarray | None: Frame data or None if read fails
        """
        if self._cap is None:
            if not self._auto_reconnect:
                return None

            if self._safe_connect():
                self.logger.info(f"Successfully opened camera {self.device_name} at {self.device_path}")
            else:
                return None

        try:
            ret, frame = self._cap.read()
            if not ret:
                self.logger.error(
                    f"Unexpected error reading from camera {self.device_name}."
                    f"{' Retrying...' if self._auto_reconnect else ' Auto-reconnect is disabled, please restart the app.'}"
                )
                self._close_camera()
                return None

            return frame

        except Exception as e:
            self.logger.error(
                f"Unexpected error reading from camera {self.device_name}: {type(e)}."
                f"{' Retrying...' if self._auto_reconnect else ' Auto-reconnect is disabled, please restart the app.'}"
            )
            self._close_camera()
            return None
