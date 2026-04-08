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


def resolve_camera_name(i2c_addr: str) -> str:
    """
    Find the camera name corresponding to the given I2C address.

    Args:
        i2c_addr (str): I2C address of the camera.

    Returns:
        str: Camera name corresponding to the I2C address.

    Raises:
        CameraOpenError: If no camera matches the given I2C address.
    """
    import re
    import subprocess

    output = subprocess.run(
        ["gst-device-monitor-1.0", "Video/Source"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout

    for line in output.splitlines():
        m = re.match(r"^\s+name\s+:\s+(.+)$", line)
        if m and i2c_addr in m.group(1):
            return m.group(1).strip()

    raise CameraOpenError(f"No camera matches I2C address '{i2c_addr}'")
