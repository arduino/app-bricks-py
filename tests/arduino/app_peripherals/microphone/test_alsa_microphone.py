# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch

import alsaaudio
import numpy as np

from arduino.app_peripherals.microphone.microphone import Microphone
from arduino.app_peripherals.microphone.alsa_microphone import ALSAMicrophone, _alsa_format_name_to_dtype, _dtype_to_alsa_format_name
from arduino.app_peripherals.microphone.errors import MicrophoneConfigError, MicrophoneOpenError


class TestAlSAMicrophoneInitialization:
    """Test ALSA microphone initialization."""

    def test_alsa_start_opens_device(self, pcm_registry):
        """Test that start() opens ALSA device."""
        mic = Microphone(device=0)

        assert not mic.is_started()

        mic.start()

        assert mic.is_started()
        pcm_instance = pcm_registry.get_last_instance()
        assert pcm_instance is not None

    def test_alsa_stop_closes_device(self, pcm_registry):
        """Test that stop() closes ALSA device."""
        mic = Microphone(device=0)
        mic.start()
        mic.stop()

        assert not mic.is_started()
        pcm_instance = pcm_registry.get_last_instance()
        assert pcm_instance.close.called


class TestALSAMicrophoneDeviceResolution:
    """Test ALSA device resolution."""

    def test_resolve_by_shorthand(self, mock_alsa_usb_mics):
        """Test resolving device by integer index."""
        mic = ALSAMicrophone()
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device=Microphone.USB_MIC_1)
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device=Microphone.USB_MIC_2)
        assert mic.device_stable_ref == "CARD=AnotherCard,DEV=0"

    def test_resolve_by_integer_index(self):
        """Test resolving device by integer index."""
        mic = ALSAMicrophone(device=0)
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device=1)
        assert mic.device_stable_ref == "CARD=AnotherCard,DEV=0"

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=[])
    def test_resolve_no_usb_devices_raises_error(self, mock_pcms):
        """Test that error is raised when no devices found."""
        with pytest.raises(MicrophoneConfigError) as exc_info:
            ALSAMicrophone(device=0)

        assert "No ALSA microphones found" in str(exc_info.value)

    def test_resolve_out_of_range_raises_error(self, mock_alsa_usb_mics):
        """Test that out of range index raises error."""
        with pytest.raises(MicrophoneConfigError) as exc_info:
            ALSAMicrophone(device=5)

        assert "out of range" in str(exc_info.value)

    def test_resolve_explicit_device_name(self, mock_alsa_usb_mics):
        """Test that explicit device names are passed through."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device="plughw:CARD=SomeCard,DEV=0")
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device="hw:1,0")
        assert mic.device_stable_ref == "CARD=AnotherCard,DEV=0"


class TestALSAErrorManagement:
    """Test handling ALSA errors."""

    def test_device_busy_error(self):
        """Test that device busy error is properly reported."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError("Device or resource busy"),
            return_value=[],
        ):
            with pytest.raises(MicrophoneOpenError) as exc_info:
                mic.start()

        assert "busy" in str(exc_info.value).lower()

    def test_generic_alsa_error(self):
        """Test generic ALSA error handling."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError("Some generic ALSA error"),
            return_value=[],
        ):
            with pytest.raises(MicrophoneOpenError):
                mic.start()

    def test_read_with_no_data_returns_none(self, pcm_registry):
        """Test that read with no data returns None."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.start()

        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.read.side_effect = None
        pcm_instance.read.return_value = (0, b"")  # Return 0 length

        audio = mic.capture()

        assert audio is None

    def test_read_error_doesnt_raise(self, pcm_registry):
        """Test that ALSA errors when reading don't raise exceptions."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.start()

        # Return ALSA error that's not disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.read.side_effect = alsaaudio.ALSAAudioError("Buffer overrun")

        mic.capture()

    def test_stop_with_close_error(self, pcm_registry):
        """Test that stop handles close errors gracefully."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.start()

        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.close.side_effect = alsaaudio.ALSAAudioError("Close failed")

        # Should not raise
        mic.stop()

        assert not mic.is_started()


class TestALSADeviceDisconnection:
    """Test ALSA device disconnection handling."""

    def test_detect_device_disconnection(self, mock_alsa_usb_mics, pcm_registry):
        """Test device disconnection detection during capture."""
        mic = ALSAMicrophone()
        mic.start()

        # Simulate device disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.read.side_effect = alsaaudio.ALSAAudioError("No such device")

        with patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", side_effect=None, return_value=[]):
            # Attempt to read should detect disconnection
            audio = mic.capture()

            assert audio is None
            assert mic._pcm is None  # PCM should be cleared

    def test_list_devices_check(self, mock_alsa_usb_mics):
        """Test device disconnection detection by enumerating devices."""
        mic = ALSAMicrophone()
        mic.start()

        devices = ALSAMicrophone.list_devices()
        assert len(devices) > 0

        # Simulate device removal
        with patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", side_effect=None, return_value=[]):
            devices = ALSAMicrophone.list_devices()
            assert len(devices) == 0


class TestALSADeviceReconnection:
    """Test ALSA device reconnection logic."""

    def test_reconnection_after_device_available(self, mock_alsa_usb_mics):
        """Test reconnection when device becomes available."""
        # Initially no devices - creation should fail
        with patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", side_effect=None, return_value=[]):
            with pytest.raises(MicrophoneConfigError):
                mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")

        # Now creation and start should work
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.start()

        assert mic.is_started()

        mic.stop()

    def test_read_reconnects(self, mock_alsa_usb_mics, pcm_registry):
        """Test read attempts reconnection after disconnection."""
        mic = ALSAMicrophone()
        mic.start()

        # Simulate a disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.read.side_effect = alsaaudio.ALSAAudioError("No such device")

        chunk = mic.capture()
        assert chunk is None

        # Mock successful reconnection
        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.side_effect = None
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        # Capturing a second time should trigger reconnection attempt
        # Note: in real situations, this would block until reconnected
        # For this test, we just verify the behavior
        chunk = mic.capture()
        assert chunk is not None


class TestALSACaptureStream:
    """Test ALSA microphone capture and stream methods."""

    def test_alsa_microphone_capture(self, mock_alsa_usb_mics):
        """Test capture with ALSA microphone."""
        mic = ALSAMicrophone()
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) == 1024

    def test_alsa_microphone_stream(self, mock_alsa_usb_mics):
        """Test streaming with ALSA microphone."""
        mic = ALSAMicrophone()
        mic.start()

        stream = mic.stream()
        chunks = []

        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            if i >= 2:
                break

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)

    @pytest.mark.parametrize(
        "format",
        [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.float32, np.float64],
    )
    def test_alsa_has_correct_format(self, mock_alsa_usb_mics, pcm_registry, format):
        """Test that ALSA is configured with correct format and that format's dtype is returned."""
        format_dtype = np.dtype(format)

        mic = ALSAMicrophone(format=format, buffer_size=128)
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert chunk.dtype == format_dtype

        pcm_instance = pcm_registry.get_last_instance()
        assert format_dtype == _alsa_format_name_to_dtype(mic.alsa_format_name)
        assert mic.alsa_format_name == _dtype_to_alsa_format_name(format_dtype)
        assert mic.alsa_format_idx == pcm_instance.info()["format"]
        assert mic.alsa_format_name == pcm_instance.info()["format_name"]

    def test_unsupported_format_with_none_dtype(self):
        """Test that unsupported formats trigger an error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(format=None)

        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(format="unsupported_format")
