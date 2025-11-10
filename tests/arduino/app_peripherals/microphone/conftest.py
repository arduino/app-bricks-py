# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Pytest configuration for microphone tests.

This file mocks alsaaudio globally so tests can run without the library installed.
"""

import sys
from unittest.mock import MagicMock


# Define a proper exception class for ALSAAudioError
class ALSAAudioError(Exception):
    """Mock ALSA audio error exception."""

    pass


# Mock alsaaudio before any test imports it
mock_alsaaudio = MagicMock()
mock_alsaaudio.ALSAAudioError = ALSAAudioError
sys.modules["alsaaudio"] = mock_alsaaudio
