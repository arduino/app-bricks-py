# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


class RemoteSensorError(Exception):
    """Base exception for remote sensor-related errors."""

    pass


class RemoteSensorOpenError(RemoteSensorError):
    """Exception raised when the remote sensor server cannot be opened."""

    pass


class RemoteSensorConfigError(RemoteSensorError):
    """Exception raised when remote sensor configuration is invalid."""

    pass
