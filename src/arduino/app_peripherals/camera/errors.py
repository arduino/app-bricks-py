# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


from arduino.app_utils.errors import AppError


class CameraError(AppError):
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
