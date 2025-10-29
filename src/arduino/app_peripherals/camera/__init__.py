# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from .camera import Camera
from .errors import *
    
__all__ = [
    "Camera",
    "CameraError",
    "CameraReadError",
    "CameraOpenError",
    "CameraConfigError",
    "CameraTransformError",
]