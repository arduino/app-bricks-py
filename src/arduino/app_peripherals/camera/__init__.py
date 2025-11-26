# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .camera import Camera
from .base_camera import BaseCamera
from .v4l_camera import V4LCamera
from .ip_camera import IPCamera
from .websocket_camera import WebSocketCamera
from .errors import *

__all__ = [
    "Camera",
    "BaseCamera",
    "V4LCamera",
    "IPCamera",
    "WebSocketCamera",
    "CameraError",
    "CameraConfigError",
    "CameraOpenError",
    "CameraReadError",
    "CameraTransformError",
]
