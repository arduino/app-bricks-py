# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from .remote_sensor import RemoteSensor
from .errors import *

__all__ = [
    "RemoteSensor",
    "RemoteSensorError",
    "RemoteSensorOpenError",
    "RemoteSensorReadError",
    "RemoteSensorConfigError",
    "RemoteSensorDisconnectedError",
]
