# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .errors import CameraOpenError


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
