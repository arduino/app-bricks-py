# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
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
    V4L (Video4Linux) camera implementation for physically connected cameras.

    This class handles USB cameras and other V4L-compatible devices on Linux systems.
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
        super().__init__(resolution, fps, adjustments, auto_reconnect)

        self.v4l_path = self._resolve_stable_path(device)
        self.name = self._resolve_name(self.v4l_path)  # Override parent name with a human-readable name
        self.logger = logger

        self._cap = None

        self._last_reconnection_attempt = 0.0  # Used for auto-reconnection when _read_frame is called

    @staticmethod
    def list_devices() -> list[int]:
        """
        Return a list of available USB cameras.

        Returns:
            list[int]: List of USB camera indices.
        """
        indices: list[int] = []
        try:
            devices = [dev for dev in os.listdir("/dev/v4l/by-id/")]
            for dev in devices:
                dev_path = os.path.join("/dev/v4l/by-id", dev)
                target = os.path.realpath(dev_path)
                video_basename = os.path.basename(target)
                if video_basename.startswith("video"):
                    index = int(video_basename.removeprefix("video"))
                    indices.append(index)

        except Exception as e:
            logger.error(f"Error listing available cameras: {e}")

        indices.sort()
        return indices

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
            # Already a stable link, resolve video device
            device_path = os.path.realpath(device)
        elif isinstance(device, str) and device.startswith("/dev/v4l/by-path"):
            # A stable link, but not the one we want, resolve video device
            if not os.path.exists(device):
                raise CameraOpenError(f"Device path {device} does not exist")
            device_path = os.path.realpath(device)
        elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # Resolve video device as /dev/video<device>
            device_index = int(device)
            device_indices = V4LCamera.list_devices()
            if device_index < 0 or device_index >= len(device_indices):
                raise CameraOpenError(f"Camera index {device_index} out of range. Available: 0-{len(device_indices)}")
            device_path = f"/dev/video{device_indices[device_index]}"
        elif isinstance(device, str) and device.startswith("/dev/video"):
            # Already a video device
            device_path = device
        else:
            raise CameraOpenError(f"Unrecognized device identifier: {device}")

        # Now map /dev/videoX to a stable link under /dev/v4l/by-id
        by_id_dir = "/dev/v4l/by-id/"
        if not os.path.exists(by_id_dir):
            raise CameraOpenError(f"Directory '{by_id_dir}' not found.")

        try:
            for entry in os.listdir(by_id_dir):
                full_path = os.path.join(by_id_dir, entry)
                if os.path.islink(full_path):
                    target = os.path.realpath(full_path)
                    if target == device_path:
                        return full_path
        except Exception as e:
            raise CameraOpenError(f"Error resolving stable link: {e}")

        raise CameraOpenError(f"No stable link found for device {device} (resolved as {device_path})")

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
        self._close_camera()

        if not os.path.exists(self.v4l_path):
            raise RuntimeError(f"No device found at {self.v4l_path}")

        try:
            self._cap = cv2.VideoCapture(self.v4l_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.name}")

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

            if self.resolution and self.resolution[0] and self.resolution[1]:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

                # Verify resolution setting
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                    logger.warning(
                        f"Camera {self.name} resolution set to {actual_width}x{actual_height} "
                        f"instead of requested {self.resolution[0]}x{self.resolution[1]}"
                    )
                    self.resolution = (actual_width, actual_height)

            if self.fps:
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)

                actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
                if actual_fps != self.fps:
                    logger.warning(f"Camera {self.name} FPS set to {actual_fps} instead of requested {self.fps}")
                    self.fps = actual_fps

            # Verify camera with a test read
            ret, frame = self._cap.read()
            if not ret and frame is None:
                raise RuntimeError(f"Read test failed for camera {self.name}")

            self._set_status("connected", {"camera_name": self.name, "camera_path": self.v4l_path})

        except Exception as e:
            logger.error(f"Unexpected error opening camera {self.name}: {e}")
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise

    def _close_camera(self) -> None:
        """Close the V4L camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._set_status("disconnected", {"camera_name": self.name, "camera_path": self.v4l_path})

    def _read_frame(self) -> np.ndarray | None:
        """
        Read a frame from the V4L camera with auto-reconnection on failure, if enabled.

        Returns:
            np.ndarray | None: Frame data or None if the read fails
        """
        try:
            if self._cap is None:
                if not self.auto_reconnect:
                    return None

                # Prevent spamming connection attempts
                current_time = time.monotonic()
                elapsed = current_time - self._last_reconnection_attempt
                if elapsed < self.auto_reconnect_delay:
                    time.sleep(self.auto_reconnect_delay - elapsed)
                self._last_reconnection_attempt = current_time

                self._open_camera()
                self.logger.info(f"Successfully reopened camera {self.name} at {self.v4l_path}")

            ret, frame = self._cap.read()
            if (not ret and frame is None) or not self._cap.isOpened():
                raise CameraReadError(f"Invalid frame returned")

            return frame

        except (CameraOpenError, CameraReadError, Exception) as e:
            self.logger.error(
                f"Failed to read from camera {self.name}: {e}."
                f"{' Retrying...' if self.auto_reconnect else ' Auto-reconnect is disabled, please restart the app.'}"
            )
            self._close_camera()  # Will reconnect on next call
            return None
