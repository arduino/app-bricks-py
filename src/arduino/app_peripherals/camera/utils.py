# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .errors import CameraOpenError


def first_plugged_camera() -> str | int:
    """
    Find the first available physically connected camera.

    Returns:
        str | int: Identifier of the first available camera

    Raises:
        CameraOpenError: If no cameras are found
    """
    from .v4l_camera import V4LCamera

    usb_devices = V4LCamera.list_devices()
    if len(usb_devices) > 0:
        return usb_devices[0]

    raise CameraOpenError("No available cameras found")
