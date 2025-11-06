# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0


class RemoteSensorError(Exception):
    """Base exception for remote sensor-related errors."""

    pass


class RemoteSensorOpenError(RemoteSensorError):
    """Exception raised when the remote sensor server cannot be opened."""

    pass


class RemoteSensorReadError(RemoteSensorError):
    """Exception raised when reading from remote sensor fails."""

    pass


class RemoteSensorConfigError(RemoteSensorError):
    """Exception raised when remote sensor configuration is invalid."""

    pass


class RemoteSensorDisconnectedError(RemoteSensorError):
    """Exception raised when the remote sensor client is disconnected."""

    pass
