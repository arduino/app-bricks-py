# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .remote_sensor import RemoteSensor
from .errors import *

__all__ = [
    "RemoteSensor",
    "RemoteSensorOpenError",
    "RemoteSensorConfigError",
]
