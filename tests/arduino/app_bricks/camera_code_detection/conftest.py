# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Stands in for pyzbar, whose zbar shared library the test hosts lack."""

import sys
from enum import IntEnum
from types import ModuleType
from unittest.mock import MagicMock


class ZBarSymbol(IntEnum):
    EAN13 = 13
    QRCODE = 64
    SQCODE = 80


class PyZbarError(Exception):
    pass


mock_pyzbar = ModuleType("pyzbar.pyzbar")
setattr(mock_pyzbar, "ZBarSymbol", ZBarSymbol)
setattr(mock_pyzbar, "PyZbarError", PyZbarError)
setattr(mock_pyzbar, "decode", MagicMock(return_value=[]))

mock_pyzbar_package = ModuleType("pyzbar")
setattr(mock_pyzbar_package, "pyzbar", mock_pyzbar)

sys.modules["pyzbar"] = mock_pyzbar_package
sys.modules["pyzbar.pyzbar"] = mock_pyzbar
