# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .remote_sensor import RemoteSensor
from .errors import *

__all__ = [
    "RemoteSensor",
    "RemoteSensorOpenError",
    "RemoteSensorConfigError",
]
