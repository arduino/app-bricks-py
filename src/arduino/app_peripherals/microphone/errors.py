# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0


class MicrophoneError(Exception):
    """Base exception for microphone-related errors."""

    pass


class MicrophoneOpenError(MicrophoneError):
    """Exception raised when the microphone cannot be opened."""

    pass


class MicrophoneReadError(MicrophoneError):
    """Exception raised when reading from microphone fails."""

    pass


class MicrophoneConfigError(MicrophoneError):
    """Exception raised when microphone configuration is invalid."""

    pass


class MicrophoneDisconnectedError(MicrophoneError):
    """Exception raised when the microphone device is disconnected and max retries are exceeded."""

    pass
