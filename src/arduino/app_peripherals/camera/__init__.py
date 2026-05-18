# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .camera import Camera
from .base_camera import BaseCamera
from .v4l_camera import V4LCamera
from .ip_camera import IPCamera
from .websocket_camera import WebSocketCamera
from .csi_camera import CSICamera
from .errors import *

__all__ = [
    "Camera",
    "BaseCamera",
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
