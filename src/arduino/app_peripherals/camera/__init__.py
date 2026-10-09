# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from arduino.app_utils._lazy_exports import lazy_exports as _lazy_exports

from .camera import Camera
from .base_camera import BaseCamera, CameraInfo
from .errors import *

if TYPE_CHECKING:
    from .csi_camera import CSICamera
    from .ip_camera import IPCamera
    from .v4l_camera import V4LCamera
    from .websocket_camera import WebSocketCamera

# Backends are loaded on first access (Camera() imports the one it needs): each
# pulls in its own dependencies (cv2, requests, websockets, cryptography), which
# would otherwise slow down the start of every app using a camera.
_LAZY_EXPORTS = {
    "V4LCamera": "v4l_camera",
    "IPCamera": "ip_camera",
    "WebSocketCamera": "websocket_camera",
    "CSICamera": "csi_camera",
}


if not TYPE_CHECKING:  # Type checkers resolve the names through the imports above
    __getattr__, __dir__ = _lazy_exports(__name__, globals(), _LAZY_EXPORTS)


__all__ = [
    "Camera",
    "BaseCamera",
    "CameraInfo",
    "V4LCamera",
    "IPCamera",
    "WebSocketCamera",
    "CSICamera",
    "CameraError",
    "CameraConfigError",
    "CameraOpenError",
    "CameraReadError",
    "CameraTransformError",
]
