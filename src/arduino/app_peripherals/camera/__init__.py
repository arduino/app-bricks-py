# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from .camera import Camera
from .v4l_camera import V4LCamera
from .ip_camera import IPCamera
from .websocket_camera import WebSocketCamera
from .errors import *
    
__all__ = [
    "Camera",
    "V4LCamera",
    "IPCamera",
    "WebSocketCamera",
    "CameraError",
    "CameraReadError",
    "CameraOpenError",
    "CameraConfigError",
    "CameraTransformError",
]