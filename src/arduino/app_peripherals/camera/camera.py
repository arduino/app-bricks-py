# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from urllib.parse import urlparse

import numpy as np

from .base_camera import BaseCamera
from .errors import CameraConfigError
from .utils import camera_registry, claim_first_available_camera, nth_plugged_camera


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
        source: str | int | None = None,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 10,
        adjustments: Callable[[np.ndarray], np.ndarray] | None = None,
        **kwargs,
    ) -> BaseCamera:
        """
        Create a camera instance based on the source type.

        Args:
            source (str | int | None): Camera source identifier. Supports:
                - None: Auto-select the first available plugged camera, i.e. not
                    already in use by another instance, giving priority to USB
                    cameras, then CSI cameras if supported by the platform.
                    Raises if every plugged camera is already in use
                - int | str: Select the n-th plugged camera (e.g., 0, 1, "0", "1",
                    ...) counting USB cameras first, then CSI ones, regardless of
                    whether it is already in use: contention on a reused camera
                    is only discovered when starting it
                - str: V4L camera ordinal index (e.g., "usb:0", "usb:1")
                - str: V4L camera device path (e.g., "usb:/dev/video0",
                    "usb:/dev/v4l/by-id/...", "usb:/dev/v4l/by-path/...
                    the "usb:" prefix is optional)
                - str: CSI camera ordinal index (e.g., "csi:0", "csi:1")
                - str: CSI camera name (e.g., "csi:CAMERA0", "csi:CAMERA1")
                - str: URL for IP cameras (e.g., "rtsp://...", "http://...")
                - str: WebSocket URL for input streams (e.g., "ws://0.0.0.0:8080")
                Default: None.
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
                    url (str): Camera stream URL.
                    username (str, optional): Authentication username.
                    password (str, optional): Authentication password.
                    timeout (float): Connection timeout in seconds. Default: 10.0.
                WebSocket Camera Parameters:
                    port (int): Port to bind the server to. Default: 8080.
                    timeout (int): Connection timeout in seconds. Default: 3.
                    certs_dir_path (str): Path to directory containing TLS certificates.
                        Default: "/app/certs".
                    use_tls (bool): Enable TLS for secure connections. If True, 'encrypt'
                        will be ignored. Default: False.
                    secret (str): Secret key for authentication/encryption. Empty string
                        disables security. Default: "".
                    encrypt (bool): Enable encryption (only effective if secret is provided).
                        Default: False.
                    auto_reconnect (bool): Whether to automatically attempt to reconnect
                        if the camera connection is lost. Default: True.

        Returns:
            BaseCamera: Appropriate camera implementation instance

        Raises:
            CameraConfigError: If camera type is not supported or parameters are invalid
            CameraOpenError: If no camera is available at the requested index

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
            camera = Camera("rtsp://192.168.1.100:554/stream")
            camera = Camera("http://192.168.1.100:8080/video.mp4", username="admin", password="secret")
            ```

            WebSocket Camera:

            ```python
            camera = Camera("ws://0.0.0.0:8080")
            camera = Camera("ws://0.0.0.0:8080", secret="my_secret", encrypt=True)
            camera = Camera("ws://0.0.0.0:8080", use_tls=True, certs_dir_path="/path/to/certs")
            ```
        """
        if source is None:
            # Auto-selection: claim the first available camera so other instances don't select it
            source, key = claim_first_available_camera()
            try:
                camera = _create_camera(source, resolution, fps, adjustments, **kwargs)
            except BaseException:
                camera_registry.release(key)
                raise
            camera_registry.bind(key, camera)
            return camera

        if not isinstance(source, (str, int)):
            raise CameraConfigError(f"Invalid source type: {type(source)}. Must be str, int or None.")

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Positional selection of the n-th plugged camera
            source = nth_plugged_camera(int(source))

        camera = _create_camera(source, resolution, fps, adjustments, **kwargs)

        # Claim local devices so auto-selection doesn't pick them
        key = _claim_key(camera)
        if key is not None:
            camera_registry.claim(key)
            camera_registry.bind(key, camera)
        return camera


def _claim_key(camera: BaseCamera) -> str | None:
    """Return the stable identity of the local device held by the camera, if any."""
    from .csi_camera import CSICamera
    from .v4l_camera import V4LCamera

    if isinstance(camera, V4LCamera):
        return camera.v4l_path
    if isinstance(camera, CSICamera):
        return camera.csi_path
    return None


def _create_camera(
    source: str,
    resolution: tuple[int, int],
    fps: int,
    adjustments: Callable[[np.ndarray], np.ndarray] | None,
    **kwargs,
) -> BaseCamera:
    """Create the camera implementation matching the given source identifier."""
    if source.startswith("usb:"):
        from .v4l_camera import V4LCamera

        v4l_source = source[4:]  # Remove "usb:" prefix
        return V4LCamera(v4l_source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)

    elif source.startswith("csi:"):
        from .csi_camera import CSICamera

        csi_source = source[4:]  # Remove "csi:" prefix
        return CSICamera(csi_source, resolution=resolution, fps=fps, adjustments=adjustments, **kwargs)

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
