# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from urllib.parse import urlparse

import numpy as np

from .base_camera import BaseCamera
from .errors import CameraConfigError
from .utils import nth_plugged_camera


class Camera:
    """
    Unified Camera class that can be configured for different camera types.

    This class serves as both a factory and a wrapper, automatically creating
    the appropriate camera implementation based on the provided configuration.

    Supports:
        - USB Cameras (local cameras connected using USB interface)
        - CSI Cameras (local cameras connected using MIPI CSI-2 interface)
        - IP Cameras (network-based cameras via RTSP, HLS)
        - WebSocket Cameras (input video streams via WebSocket client)

    Note: constructor arguments (except those in signature) must be provided in
    keyword format to forward them correctly to the specific camera implementations.
    """

    def __new__(
        cls,
        source: str | int = 0,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        **kwargs,
    ) -> BaseCamera:
        """
        Create a camera instance based on the source type.

        Args:
            source (Union[str, int]): Camera source identifier. Supports:
                - int: Auto-select the n-th available physically connected camera
                    giving priority to USB cameras, then CSI cameras if supported
                    by the platform
                - str: V4L camera ordinal index (e.g., "usb:0", "usb:1")
                - str: V4L camera device path (e.g., "usb:/dev/video0",
                    "usb:/dev/v4l/by-id/...", "usb:/dev/v4l/by-path/...")
                - str: CSI camera ordinal index (e.g., "csi:0", "csi:1")
                - str: CSI camera name (e.g., "csi:CAMERA0", "csi:CAMERA1")
                - str: URL for IP cameras (e.g., "rtsp://...", "http://...")
                - str: WebSocket URL for input streams (e.g., "ws://0.0.0.0:8080")
                Default: 0.
            resolution (tuple[int, int]): Frame resolution as (width, height).
                Default: (640, 480).
            fps (int): Target frames per second. Default: 10.
            adjustments (callable, optional): Function pipeline to adjust frames
                that takes a numpy array and returns a numpy array. Default: None.
            **kwargs: Camera-specific configuration parameters grouped by type:
                V4L Camera Parameters:
                    device (int | str): V4L device. Default: 0.
                    codec (str, optional): Video codec to use (FourCC). Options: "YUVY",
                            "MJPG", "H264". Default: "" (auto).
                CSI Camera Parameters:
                    device (int | str): CSI device. Default: 0.
                IP Camera Parameters:
                    url (str): Camera stream URL
                    username (str, optional): Authentication username.
                    password (str, optional): Authentication password.
                    timeout (float): Connection timeout in seconds. Default: 10.0.
                WebSocket Camera Parameters:
                    host (str): WebSocket server host. Default: "0.0.0.0".
                    port (int): WebSocket server port. Default: 8080.
                    timeout (float): Connection timeout in seconds. Default: 10.0.

        Returns:
            BaseCamera: Appropriate camera implementation instance

        Raises:
            CameraConfigError: If source type is not supported or parameters are invalid
            CameraOpenError: If the camera cannot be opened

        Examples:
            V4L Camera:

            ```python
            camera = Camera("usb:0", resolution=(640, 480), fps=30)
            camera = Camera("usb:/dev/video1", fps=15)
            ```

            CSI Camera:

            ```python
            camera = Camera("csi:0", resolution=(640, 480), fps=30)
            camera = Camera("csi:CAMERA1", fps=15)
            ```

            IP Camera:

            ```python
            camera = Camera("rtsp://192.168.1.100:554/stream", username="admin", password="secret", timeout=15.0)
            camera = Camera("http://192.168.1.100:8080/video.mp4")
            ```

            WebSocket Camera:

            ```python
            camera = Camera("ws://0.0.0.0:8080")
            camera = Camera("ws://192.168.1.100:8080", timeout=5)
            ```
        """
        if not isinstance(source, (str, int)):
            raise CameraConfigError(f"Invalid source type: {type(source)}. Must be str or int.")

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Select the n-th available camera
            idx = int(source) if isinstance(source, str) else source
            source = nth_plugged_camera(idx)

        if source.startswith("csi:"):
            from .csi_camera import CSICamera

            csi_source = source[4:]  # Remove "csi:" prefix
            return CSICamera(csi_source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)

        elif source.startswith("usb:"):
            from .v4l_camera import V4LCamera

            v4l_source = source[4:]  # Remove "usb:" prefix
            return V4LCamera(v4l_source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)

        # All other cases are handled by URL parsing
        else:
            parsed = urlparse(source)
            if parsed.scheme in ["http", "https", "rtsp"]:
                # IP Camera
                from .ip_camera import IPCamera

                return IPCamera(source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)
            elif parsed.scheme in ["ws", "wss"]:
                # WebSocket Camera - extract host and port from URL
                from .websocket_camera import WebSocketCamera

                port = parsed.port or 8080
                return WebSocketCamera(port=port, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)
            elif source.startswith("/dev/video") or source.startswith("/dev/v4l/by-id/") or source.startswith("/dev/v4l/by-path/"):
                # V4L device path, by-id, or by-path
                from .v4l_camera import V4LCamera

                return V4LCamera(source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)
            else:
                raise CameraConfigError(f"Unsupported camera source: {source}")
