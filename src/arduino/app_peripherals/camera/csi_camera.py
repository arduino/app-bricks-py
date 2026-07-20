# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import re
import time
from collections.abc import Callable
from typing import Optional

import cv2
import numpy as np

from arduino.app_utils import Logger

from . import csi_camss_discovery, csi_camx_discovery
from .camera import BaseCamera
from .errors import CameraOpenError, CameraReadError


logger = Logger("CSICamera")

_BACKENDS = {
    "camss": csi_camss_discovery,
    "camx": csi_camx_discovery,
}


def detect_camera_stack() -> str:
    """Detect the host CSI camera stack."""
    if csi_camss_discovery.camss_driver_present():
        return "camss"
    if csi_camx_discovery.camx_driver_present() and csi_camx_discovery.camx_socket_available():
        return "camx"
    raise RuntimeError("No supported camera stack detected. Please ensure either CAMSS or CAMX is available.")


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
                - str: Camera ordinal index as string (e.g., "0", "1")
                - str: Camera name (e.g., "CAMERA0", "CAMERA1")
                Default: 0 (first available CSI camera).
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int, optional): Frames per second to capture from the camera. Default: 10.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        super().__init__(resolution, fps, adjustments, auto_reconnect)

        self._backend = _BACKENDS[detect_camera_stack()]
        self._camera_id = self._resolve_camera_id(device)
        self.csi_path = self._backend.get_camera_identifier(self._camera_id)
        self.name = f"csi:{self.csi_path}"  # Override parent name with a human-readable name
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
        try:
            backend = _BACKENDS[detect_camera_stack()]
            return backend.list_camera_ids()
        except Exception as e:
            logger.error(f"Error listing available cameras: {e}")
            return []

    @staticmethod
    def list_device_names() -> list[str]:
        """
        Return a list of available CSI cameras.

        Returns:
            list[str]: List of CSI camera device paths.
        """
        try:
            backend = _BACKENDS[detect_camera_stack()]
            return [backend.get_camera_identifier(camera_id) for camera_id in backend.list_camera_ids()]
        except Exception as e:
            logger.error(f"Error listing available cameras: {e}")
            return []

    def _resolve_camera_id(self, device: str | int) -> int:
        """
        Resolve a device identifier to a backend-specific camera id.

        Args:
            device: Camera identifier in the form of either:
                - int: Camera ordinal index (e.g., 0, 1)
                - str: Camera ordinal index as string (e.g., "0", "1")
                - str: Camera name (e.g., "CAMERA0", "CAMERA1")

        Returns:
            int: Backend-specific camera id

        Raises:
            CameraOpenError: If camera index is out of range or device cannot be found
        """
        camera_ids = self._backend.list_camera_ids()

        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            ordinal = int(device)
            if ordinal < 0 or ordinal >= len(camera_ids):
                raise CameraOpenError(f"Camera index {ordinal} out of range. Available: 0-{len(camera_ids) - 1}")

            return camera_ids[ordinal]

        elif isinstance(device, str):
            if "CAMERA" not in device.upper():
                raise CameraOpenError(f"Invalid camera name: {device}. Expected format like 'CAMERA0'")

            m = re.search(r"(\d+)", device)
            if not m:
                raise CameraOpenError(f"Invalid camera device string: {device}")
            requested = int(m.group(1))
            if requested not in camera_ids:
                raise CameraOpenError(f"Camera id {requested} not available. Available: {camera_ids}")
            return requested

        else:
            raise CameraOpenError(f"Invalid device identifier: {device}")

    def _open_camera(self) -> None:
        """
        Open the CSI camera connection with retry logic.

        Retries with exponential backoff until successful or self.max_retries is reached.
        """
        self._close_camera()
        width, height = 1280, 720  # Default resolution if not specified
        if self.resolution and self.resolution[0] and self.resolution[1]:
            width, height = self.resolution

        gstreamer_pipeline = (
            f"{self._backend.gstreamer_source(self._camera_id)} ! "
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
