# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import Final, TypeVar


class _Shutdown:
    """Type of the sentinel that closes a pipeline queue, so a queue carries its items or the sentinel."""


_SHUTDOWN: Final = _Shutdown()

T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")

# Shared by the modules of this package only
__all__ = ["_SHUTDOWN", "_Shutdown", "T_IN", "T_OUT"]
