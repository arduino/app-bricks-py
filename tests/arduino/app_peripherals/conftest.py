# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_utils.peripheral_registry import Peripherals


@pytest.fixture(autouse=True)
def clean_peripheral_registry():
    """Keep the peripherals built by one test out of the registry seen by the others.

    Peripherals register themselves for automatic release, and the registry is process-wide. Left
    alone, every peripheral built anywhere in this suite would pile up in it and get stopped by the
    interpreter-exit hook, which logs after pytest has torn its logging down.
    """
    Peripherals.clear()
    yield
    Peripherals.clear()
