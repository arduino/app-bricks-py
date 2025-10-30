# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0


class CameraError(Exception):
    """Base exception for camera-related errors."""

    pass


class CameraOpenError(CameraError):
    """Exception raised when the camera cannot be opened."""

    pass


class CameraReadError(CameraError):
    """Exception raised when reading from camera fails."""

    pass


class CameraConfigError(CameraError):
    """Exception raised when camera configuration is invalid."""

    pass


class CameraTransformError(CameraError):
    """Exception raised when frame transformation fails."""

    pass
