# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from typing import TypeVar


# Sentinel value that triggers shutdown logic
_SHUTDOWN = object()

T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")
