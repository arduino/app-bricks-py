# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


class SpeakerError(Exception):
    """Base exception for Speaker-related errors."""

    pass


class SpeakerOpenError(SpeakerError):
    """Exception raised when the speaker cannot be opened."""

    pass


class SpeakerWriteError(SpeakerError):
    """Exception raised when writing to speaker fails."""

    pass


class SpeakerConfigError(SpeakerError):
    """Exception raised when speaker configuration is invalid."""

    pass
