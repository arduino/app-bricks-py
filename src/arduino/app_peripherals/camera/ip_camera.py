# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np
import requests
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse

from arduino.app_utils import Logger

from .camera import BaseCamera
from .errors import CameraConfigError, CameraOpenError

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
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 10,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        fps: int = 10,
        adjustments: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
        """
        super().__init__(resolution, fps, adjustments)
        self.url = url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.logger = logger

        self._cap = None

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

        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise CameraOpenError(f"Failed to open IP camera: {self.url}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

        # Test by reading one frame
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._cap.release()
            self._cap = None
            raise CameraOpenError(f"Cannot read from IP camera: {self.url}")

        logger.info(f"Opened IP camera: {self.url}")

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
                raise CameraOpenError(f"HTTP camera returned status {response.status_code}: {self.url}")

        except requests.RequestException as e:
            raise CameraOpenError(f"Cannot connect to HTTP camera {self.url}: {e}")

    def _close_camera(self) -> None:
        """Close the IP camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the IP camera with automatic reconnection."""
        if self._cap is None:
            logger.info(f"No connection to IP camera {self.url}, attempting to reconnect")
            try:
                self._open_camera()
            except Exception as e:
                logger.error(f"Failed to reconnect to IP camera {self.url}: {e}")
                return None

        ret, frame = self._cap.read()
        if ret and frame is not None:
            return frame

        if not self._cap.isOpened():
            logger.warning(f"IP camera connection dropped: {self.url}")
            self._close_camera()  # Will reconnect on next call

        return None
