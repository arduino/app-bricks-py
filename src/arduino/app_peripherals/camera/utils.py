# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from ..device_registry import DeviceRegistry
from .errors import CameraOpenError

camera_registry = DeviceRegistry()
"""Tracks the cameras assigned to auto-selected Camera instances."""


def claim_first_available_camera() -> tuple[str, str]:
    """
    Find and claim the first plugged camera not assigned to another instance.

    USB cameras take precedence over CSI ones, if supported by the current
    platform. The claim is keyed on the camera's stable identity so it survives
    device reordering, and must be released back to camera_registry, either
    explicitly or by binding it to its owner.

    Returns:
        tuple[str, str]: The claimed camera as (source, claim key), where
            source is a Camera factory source string and claim key is the
            camera's stable identity.

    Raises:
        CameraOpenError: If no camera is plugged or all are already in use.
    """
    from .v4l_camera import V4LCamera

    path = camera_registry.select(V4LCamera._list_stable_paths)
    if path is not None:
        return f"usb:{path}", path

    from .csi_camera import CSICamera

    names = CSICamera.list_device_names()
    name = camera_registry.select(lambda: names)
    if name is not None:
        return f"csi:{names.index(name)}", name

    raise CameraOpenError("No available cameras found: either none is plugged or all are already in use")


def nth_plugged_camera(idx: int) -> str:
    """
    Find the n-th plugged camera, regardless of whether it is already in use.

    The index spans USB cameras first, then CSI cameras, if supported by the
    current platform.

    Args:
        idx (int): Index of the camera to select (0-based).

    Returns:
        str: Identifier of the n-th plugged camera ("usb:X" or "csi:X").

    Raises:
        CameraOpenError: If no camera is plugged at the given index.
    """
    from .v4l_camera import V4LCamera

    usb_count = len(V4LCamera.list_devices())
    if idx < usb_count:
        return f"usb:{idx}"

    from .csi_camera import CSICamera

    csi_count = len(CSICamera.list_devices())
    if idx - usb_count < csi_count:
        return f"csi:{idx - usb_count}"

    raise CameraOpenError(f"No camera found at index {idx}: only {usb_count + csi_count} camera(s) plugged")


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
