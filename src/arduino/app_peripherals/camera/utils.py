# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .errors import CameraOpenError


def nth_plugged_camera(idx: int) -> str:
    """
    Find the n-th available physically connected camera.
    The precedence is CSI cameras first, then USB cameras.

    Args:
        idx (int): Index of the camera to select (0-based).

    Returns:
        str | int: Identifier of the n-th available camera

    Raises:
        CameraOpenError: If no cameras are found or index is out of range
    """
    from .csi_camera import CSICamera

    csi_cameras = CSICamera.list_devices()
    if len(csi_cameras) > 0:
        if idx < len(csi_cameras):
            return "csi:" + str(csi_cameras[idx])

    from .v4l_camera import V4LCamera

    usb_cameras = V4LCamera.list_devices()
    if len(usb_cameras) > 0:
        if idx < len(usb_cameras):
            return "usb:" + str(usb_cameras[idx])

    raise CameraOpenError("No available cameras found")
