# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import re
import time
from typing import Optional
import cv2
import numpy as np
from collections.abc import Callable

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraOpenError, CameraReadError
from .media_graph import scan_sensor_i2c_addresses, find_sensor_i2c_addr, resolve_camera_name


logger = Logger("CSICamera")


class CSICamera(BaseCamera):
    """
    CSI (Camera Serial Interface) camera implementation for physically connected cameras.

    This class handles CSI cameras on Linux systems.
    """

    def __init__(
        self,
        device: str | int = 0,
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 30,
        adjustments: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize CSI camera.

        Args:
            device: Camera identifier in the form of either:
                - int: Camera ordinal index (e.g., 0, 1)
                - str: Camera name (e.g., "CAMERA0", "CAMERA1")
                Default: 0 (first available CSI camera).
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int, optional): Frames per second to capture from the camera. Default: 10.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        super().__init__(resolution, fps, adjustments, auto_reconnect)

        self.media_dev = "/dev/media0"

        self.csi_path = self._get_camera(device)

        self.name = f"CSI:{self.csi_path}"

        self.logger = logger

        self._cap = None

        self._last_reconnection_attempt = 0.0  # Used for auto-reconnection when _read_frame is called

    @staticmethod
    def list_devices() -> list[int]:
        """
        Return a sorted list of available CSI cameras.

        Returns:
            list[int]: List of CSI camera indices.
        """
        entries = scan_sensor_i2c_addresses("/dev/media0")
        indices = []
        for csiphy_name, _ in entries:
            m = re.search(r"msm_csiphy(\d+)", csiphy_name)
            if m:
                indices.append(int(m.group(1)))
        indices.sort()

        return indices

    @staticmethod
    def list_device_names() -> list[str]:
        """
        Return a list of available CSI cameras.

        Returns:
            list[str]: List of CSI camera device paths.
        """
        paths: list[str] = []
        try:
            entities = scan_sensor_i2c_addresses("/dev/media0")
            for entity in entities:
                camera_name = resolve_camera_name(entity[1])
                paths.append(camera_name)

        except Exception as e:
            logger.error(f"Error listing available cameras: {e}")

        return paths

    def _get_camera_name(self, csiphy_index) -> str:
        """
        Get the camera name wired at the given CSIPHY index, managed by the specified media device.
        """
        i2c = find_sensor_i2c_addr(self.media_dev, csiphy_index)

        return resolve_camera_name(i2c)

    def _get_camera(self, device: str | int) -> str:
        """
        Get the camera path for a given device identifier.

        Args:
            device: Camera identifier in the form of either:
                - int: Camera ordinal index into available devices (e.g., 0, 1)
                - str: Camera name (e.g., "CAMERA0", "CAMERA1")

        Returns:
            str: Camera device path

        Raises:
            CameraOpenError: If camera index is out of range or device cannot be found
        """
        device_indices = self.list_devices()

        if isinstance(device, int):
            if device < 0 or device >= len(device_indices):
                raise CameraOpenError(f"Camera index {device} out of range. Available: 0-{len(device_indices) - 1}")

            csiphy_index = device_indices[device]

        elif isinstance(device, str):
            if not "CAMERA" in device.upper():
                raise CameraOpenError(f"Invalid camera name: {device}. Expected format like 'CAMERA0'")

            m = re.search(r"(\d+)", device)
            if not m:
                raise CameraOpenError(f"Invalid camera device string: {device}")
            requested = int(m.group(1))
            if requested not in device_indices:
                raise CameraOpenError(f"Camera csiphy index {requested} not available. Available: {device_indices}")
            csiphy_index = requested

        else:
            raise CameraOpenError(f"Invalid device identifier: {device}")

        return self._get_camera_name(csiphy_index)

    def _open_camera(self) -> None:
        """
        Open the CSI camera connection with retry logic.

        Retries with exponential backoff until successful or self.max_retries is reached.
        """
        self._close_camera()
        camera_name = self.csi_path.replace(" ", r"\ ")  # Escape spaces for GStreamer pipeline
        width, height = 1280, 720  # Default resolution if not specified
        if self.resolution and self.resolution[0] and self.resolution[1]:
            width, height = self.resolution

        gstreamer_pipeline = (
            f"libcamerasrc camera-name={camera_name} ! "
            f"video/x-raw,width={width},height={height},framerate={self.fps}/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1"
        )

        try:
            self._cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.name}")

            # Verify camera with a test read
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Read test failed for camera {self.name}")

            if self.resolution and self.resolution[0] and self.resolution[1]:
                # Verify resolution setting
                actual_height, actual_width = frame.shape[:2]
                if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
                    logger.warning(
                        f"Camera {self.name} resolution set to {actual_width}x{actual_height} "
                        f"instead of requested {self.resolution[0]}x{self.resolution[1]}"
                    )
                    self.resolution = (actual_width, actual_height)

            self._set_status("connected", {"camera_name": self.name, "camera_path": self.csi_path})

        except Exception as e:
            logger.error(f"Unexpected error opening camera {self.name}: {e}")
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise

    def _close_camera(self) -> None:
        """Close the CSI camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._set_status("disconnected", {"camera_name": self.name, "camera_path": self.csi_path})

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
                self.logger.info(f"Successfully reopened camera {self.name} at {self.csi_path}")

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
