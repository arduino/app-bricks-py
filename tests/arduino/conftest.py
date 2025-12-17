# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
Pytest configuration for tests relying on microphone and speaker.

This file mocks alsaaudio so tests can run on systems without the library installed
(e.g., macOS or Windows which doesn't have ALSA) or without any specific hardware.
"""

import sys
from unittest.mock import MagicMock


# Define a proper exception class for ALSAAudioError that we can use in our mocks
class ALSAAudioError(Exception):
    """Mock ALSA audio error exception."""

    pass


# Mock alsaaudio for systems where it's not installed (e.g. dev machines) and for
# CI environments
mock_alsaaudio = MagicMock()
mock_alsaaudio.ALSAAudioError = ALSAAudioError
sys.modules["alsaaudio"] = mock_alsaaudio
