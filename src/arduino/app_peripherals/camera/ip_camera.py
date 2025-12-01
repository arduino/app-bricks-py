# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
import requests
from urllib.parse import urlparse
from collections.abc import Callable

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraConfigError, CameraOpenError, CameraReadError

logger = Logger("IPCamera")


class IPCamera(BaseCamera):
    """
    IP Camera implementation for network-based cameras.

    Supports RTSP, HTTP, and HTTPS camera streams.
    Can handle authentication and various streaming protocols.
    """

    def __init__(
        self,
        url: str,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 10,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize IP camera.

        Args:
            url: Camera stream URL (i.e. rtsp://..., http://..., https://...)
            username: Optional authentication username
            password: Optional authentication password
            timeout: Connection timeout in seconds
            resolution (tuple, optional): Resolution as (width, height). None uses default resolution.
            fps (int): Frames per second to capture from the camera.
            adjustments (callable, optional): Function or function pipeline to adjust frames that takes
                a numpy array and returns a numpy array. Default: None
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        super().__init__(resolution, fps, adjustments, auto_reconnect)
        self.url = url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.logger = logger

        self._cap = None

        self._last_reconnection_attempt = 0.0  # Used for auto-reconnection when _read_frame is called

        self._validate_url()

    def _validate_url(self) -> None:
        """Validate the camera URL format."""
        try:
            parsed = urlparse(self.url)
            if parsed.scheme not in ["http", "https", "rtsp"]:
                raise CameraConfigError(f"Unsupported URL scheme: {parsed.scheme}")
        except Exception as e:
            raise CameraConfigError(f"Invalid URL format: {e}")

    def _open_camera(self) -> None:
        """Open the IP camera connection."""
        url = self._build_url()

        # Test connectivity first for HTTP streams
        if self.url.startswith(("http://", "https://")):
            self._test_http_connectivity()

        try:
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open IP camera at {self.url}")

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

            # Test by reading one frame
            ret, frame = self._cap.read()
            if not ret and frame is None:
                raise RuntimeError(f"Read test failed for IP camera at {self.url}")

            self._set_status("connected", {"camera_url": self.url})

        except Exception as e:
            logger.error(f"Unexpected error opening IP camera at {self.url}: {e}")
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise

    def _build_url(self) -> str:
        """Build URL with authentication if credentials provided."""
        # If no username or password provided as parameters, return original URL
        if not self.username or not self.password:
            return self.url

        parsed = urlparse(self.url)

        # Override any URL credentials if credentials are provided
        auth_netloc = f"{self.username}:{self.password}@{parsed.hostname}"
        if parsed.port:
            auth_netloc += f":{parsed.port}"

        return f"{parsed.scheme}://{auth_netloc}{parsed.path}"

    def _test_http_connectivity(self) -> None:
        """Test HTTP/HTTPS camera connectivity."""
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = requests.head(self.url, auth=auth, timeout=self.timeout, allow_redirects=True)

            if response.status_code not in [200, 206]:  # 206 for partial content
                raise RuntimeError(f"HTTP camera returned status {response.status_code}: {self.url}")

        except requests.RequestException as e:
            raise RuntimeError(f"Cannot connect to HTTP camera {self.url}: {e}")

    def _close_camera(self) -> None:
        """Close the IP camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._set_status("disconnected", {"camera_url": self.url})

    def _read_frame(self) -> np.ndarray | None:
        """Read a frame from the IP camera with automatic reconnection."""
        try:
            if self._cap is None:
                if not self.auto_reconnect:
                    return None

                # Prevent spamming connection attempts
                import time

                current_time = time.monotonic()
                elapsed = current_time - self._last_reconnection_attempt
                if elapsed < self.auto_reconnect_delay:
                    time.sleep(self.auto_reconnect_delay - elapsed)
                self._last_reconnection_attempt = current_time

                self._open_camera()
                self.logger.info(f"Successfully reconnected to IP camera at {self.url}")

            ret, frame = self._cap.read()
            if (not ret and frame is None) or not self._cap.isOpened():
                raise CameraReadError(f"Invalid frame returned")

            return frame

        except (CameraOpenError, CameraReadError, Exception) as e:
            self.logger.error(
                f"Failed to read from IP camera at {self.url}: {e}."
                f"{' Retrying...' if self.auto_reconnect else ' Auto-reconnect is disabled, please restart the app.'}"
            )
            self._close_camera()  # Will reconnect on next call
            return None
