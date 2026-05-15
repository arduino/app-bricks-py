# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import TypeVar


# Sentinel value that triggers shutdown logic
_SHUTDOWN = object()

T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")
