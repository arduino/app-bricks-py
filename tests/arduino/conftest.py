# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
Pytest configuration for tests relying on microphone and speaker.

This file mocks alsaaudio so tests can run on systems without the library installed
(e.g., macOS or Windows which doesn't have ALSA) or without any specific hardware.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


# Define a proper exception class for ALSAAudioError that we can use in our mocks
class ALSAAudioError(Exception):
    """Mock ALSA audio error exception."""

    pass


MOCK_CARDS = ["SomeCard", "AnotherCard"]
MOCK_CARD_INDEXES = [i for i in range(len(MOCK_CARDS))]
MOCK_PCMS = ["plughw:CARD=SomeCard,DEV=0", "hw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0", "hw:CARD=AnotherCard,DEV=0"]


class MockPCMRegistry:
    """Registry for tracking MockPCM instances created during tests."""

    def __init__(self):
        self._instances = []

    def register(self, instance):
        """Register a MockPCM instance."""
        self._instances.append(instance)

    def get_last_instance(self):
        """Get the most recently created MockPCM instance."""
        return self._instances[-1] if self._instances else None

    def get_all_instances(self):
        """Get all MockPCM instances created."""
        return self._instances.copy()

    def reset(self):
        """Clear the instance registry."""
        self._instances.clear()


# Global registry for MockPCM instances
_pcm_registry = MockPCMRegistry()


class MockPCM:
    """Mock PCM class that maintains state based on construction parameters and uses MagicMock for call tracking."""

    def __init__(self, type=None, mode=None, device=None, rate=16000, channels=1, format=2, periodsize=1024):
        """
        Initialize mock PCM with state tracking.

        Args:
            type: PCM type (e.g., PCM_CAPTURE, PCM_PLAYBACK)
            mode: PCM mode (e.g., PCM_NORMAL)
            device: Device name
            rate: Sample rate in Hz
            channels: Number of channels
            format: Format index (e.g., 2 for S16_LE)
            periodsize: Buffer size / period size
        """
        self.type = type
        self.mode = mode
        self.device = device
        self.rate = rate
        self.channels = channels
        self.format = format
        self.periodsize = periodsize
        self._is_open = True
        self._read_count = 0

        # Wrap methods with MagicMock for automatic call tracking
        self._real_info = self._info_impl
        self._real_read = self._read_impl
        self._real_write = self._write_impl
        self._real_close = self._close_impl

        self.info = MagicMock(side_effect=self._real_info)
        self.read = MagicMock(side_effect=self._real_read)
        self.write = MagicMock(side_effect=self._real_write)
        self.close = MagicMock(side_effect=self._real_close)

        # Register this instance
        _pcm_registry.register(self)

        # Map format indices to format names
        self._format_idx_to_format_name = {
            0: "S8",
            1: "U8",
            2: "S16_LE",
            3: "S16_BE",
            4: "U16_LE",
            5: "U16_BE",
            6: "S24_LE",
            7: "S24_BE",
            8: "U24_LE",
            9: "U24_BE",
            10: "S32_LE",
            11: "S32_BE",
            12: "U32_LE",
            13: "U32_BE",
            14: "FLOAT_LE",
            15: "FLOAT_BE",
            16: "FLOAT64_LE",
            17: "FLOAT64_BE",
        }

        # Map format indices to numpy dtypes
        self._format_idx_to_dtype = {
            0: np.int8,
            1: np.uint8,
            2: np.int16,
            3: np.dtype(">i2"),
            4: np.dtype("<u2"),
            5: np.dtype(">u2"),
            6: np.dtype("<i4"),  # 24-bit stored as 32-bit signed integer
            7: np.dtype(">i4"),  # 24-bit stored as 32-bit signed integer
            8: np.dtype("<u4"),  # 24-bit stored as 32-bit unsigned integer
            9: np.dtype(">u4"),  # 24-bit stored as 32-bit unsigned integer
            10: np.int32,
            11: np.dtype(">i4"),
            12: np.dtype("<u4"),
            13: np.dtype(">u4"),
            14: np.float32,
            15: np.dtype(">f4"),
            16: np.float64,
            17: np.dtype(">f8"),
        }

    def _info_impl(self):
        """Returns PCM device info based on initialization parameters."""
        return {
            "format": self.format,
            "format_name": self._format_idx_to_format_name.get(self.format, "UNKNOWN"),
            "rate": self.rate,
            "channels": self.channels,
            "period_size": self.periodsize,
        }

    def _read_impl(self):
        """
        Returns audio data from the mock PCM device.

        Returns:
            tuple: (length, data_bytes) where length is the number of frames read
        """
        if not self._is_open:
            raise ALSAAudioError("PCM device is not open")

        self._read_count += 1

        dtype = self._format_idx_to_dtype.get(self.format, np.int16)
        audio_data = np.arange(self.periodsize, dtype=dtype)

        return (self.periodsize, audio_data.tobytes())

    def _write_impl(self, data: bytes):
        """
        Writes audio data to the mock PCM device.

        Args:
            data: Audio data bytes to write

        Returns:
            int: Number of frames written
        """
        if not self._is_open:
            raise ALSAAudioError("PCM device is not open")

        dtype = self._format_idx_to_dtype.get(self.format, np.int16)
        frames = len(data) // (dtype.itemsize * self.channels)
        return frames

    def _close_impl(self):
        """Closes the mock PCM device."""
        self._is_open = False


# Mock alsaaudio for systems where it's not available (e.g. dev machines, CI)
mock_alsaaudio = MagicMock()

mock_alsaaudio.ALSAAudioError = ALSAAudioError

# Define PCM type constants
mock_alsaaudio.PCM_CAPTURE = 0
mock_alsaaudio.PCM_PLAYBACK = 1

# Define PCM mode constants
mock_alsaaudio.PCM_NORMAL = 0
mock_alsaaudio.PCM_NONBLOCK = 1

# Define PCM format constants
mock_alsaaudio.PCM_FORMAT_S8 = 0
mock_alsaaudio.PCM_FORMAT_U8 = 1
mock_alsaaudio.PCM_FORMAT_S16_LE = 2
mock_alsaaudio.PCM_FORMAT_S16_BE = 3
mock_alsaaudio.PCM_FORMAT_U16_LE = 4
mock_alsaaudio.PCM_FORMAT_U16_BE = 5
mock_alsaaudio.PCM_FORMAT_S24_LE = 6
mock_alsaaudio.PCM_FORMAT_S24_BE = 7
mock_alsaaudio.PCM_FORMAT_U24_LE = 8  # Not supported
mock_alsaaudio.PCM_FORMAT_U24_BE = 9  # Not supported
mock_alsaaudio.PCM_FORMAT_S32_LE = 10
mock_alsaaudio.PCM_FORMAT_S32_BE = 11
mock_alsaaudio.PCM_FORMAT_U32_LE = 12
mock_alsaaudio.PCM_FORMAT_U32_BE = 13
mock_alsaaudio.PCM_FORMAT_FLOAT_LE = 14
mock_alsaaudio.PCM_FORMAT_FLOAT_BE = 15
mock_alsaaudio.PCM_FORMAT_FLOAT64_LE = 16
mock_alsaaudio.PCM_FORMAT_FLOAT64_BE = 17
# Other formats are not supported

# Mock free functions - these return realistic values for the mock environment
mock_alsaaudio.cards = MagicMock(return_value=MOCK_CARDS)
mock_alsaaudio.card_indexes = MagicMock(return_value=MOCK_CARD_INDEXES)


def mock_card_name(idx):
    """Mock card_name function that returns card name and description."""
    if idx < len(MOCK_CARDS):
        return [MOCK_CARDS[idx], f"USB Audio Device {idx}"]
    raise ALSAAudioError(f"Card index {idx} out of range")


mock_alsaaudio.card_name = MagicMock(side_effect=mock_card_name)


def mock_pcms(pcm_type=None):
    """Mock pcms function that returns available PCM devices."""
    # Filter based on type if specified
    if pcm_type == mock_alsaaudio.PCM_CAPTURE:
        return MOCK_PCMS
    elif pcm_type == mock_alsaaudio.PCM_PLAYBACK:
        return MOCK_PCMS  # For simplicity, return same devices
    return MOCK_PCMS


mock_alsaaudio.pcms = MagicMock(side_effect=mock_pcms)

# Mock mixers function - return empty list by default
mock_alsaaudio.mixers = MagicMock(return_value=[])

# Mock PCM constructor - return MockPCM instances
mock_alsaaudio.PCM = MockPCM

sys.modules["alsaaudio"] = mock_alsaaudio


@pytest.fixture
def pcm_registry():
    """
    Fixture that provides a registry for tracking MockPCM instances during tests.

    The registry provides:
    - .get_last_instance() - Get the most recently created PCM instance
    - .get_all_instances() - Get all PCM instances created in the test

    Use standard MagicMock attributes for verification on the returned instances.

    Example:
        def test_pcm_operations(pcm_registry):
            mic = ALSAMicrophone()
            mic.start()

            pcm = pcm_registry.get_last_instance()
            assert pcm.info.call_count >= 1
    """
    _pcm_registry.reset()
    yield _pcm_registry


@pytest.fixture
def mock_alsa_usb_mics():
    """
    Fixture that mocks ALSA USB device detection for USB microphone tests.

    This fixture patches Path.exists and Path.resolve to simulate a USB audio device
    being present on the system. Use this fixture in tests that need to work with
    USB microphones.

    Example:
        def test_usb_microphone(mock_alsa_usb_mics):
            mic = Microphone()
            mic.start()
            # ... test operations
    """
    from unittest.mock import patch

    # Mock USB device path resolution
    usb_device_path = "/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c"

    with (
        patch("arduino.app_peripherals.microphone.alsa_microphone.Path.exists", return_value=True) as mock_exists,
        patch("arduino.app_peripherals.microphone.alsa_microphone.Path.resolve", return_value=Path(usb_device_path)) as mock_resolve,
    ):
        yield {"mock_exists": mock_exists, "mock_resolve": mock_resolve, "usb_device_path": usb_device_path}


@pytest.fixture
def mock_alsa_usb_speakers():
    """
    Fixture that mocks ALSA USB device detection for USB speaker tests.

    This fixture patches Path.exists and Path.resolve to simulate a USB audio device
    being present on the system. Use this fixture in tests that need to work with
    USB speakers.

    Example:
        def test_usb_speaker(mock_alsa_usb_speakers):
            spkr = Speaker()
            spkr.start()
            # ... test operations
    """
    from unittest.mock import patch

    # Mock USB device path resolution
    usb_device_path = "/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0p"

    with (
        patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists", return_value=True) as mock_exists,
        patch("arduino.app_peripherals.speaker.alsa_speaker.Path.resolve", return_value=Path(usb_device_path)) as mock_resolve,
    ):
        yield {"mock_exists": mock_exists, "mock_resolve": mock_resolve, "usb_device_path": usb_device_path}
