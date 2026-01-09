# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import patch

import alsaaudio
import numpy as np

from arduino.app_peripherals.speaker.speaker import Speaker
from arduino.app_peripherals.speaker.alsa_speaker import ALSASpeaker, _alsa_format_name_to_dtype, _dtype_to_alsa_format_name
from arduino.app_peripherals.speaker.errors import SpeakerConfigError, SpeakerOpenError


class TestALSASpeakerInitialization:
    """Test ALSA speaker initialization."""

    def test_alsa_start_opens_device(self, pcm_registry):
        """Test that start() opens ALSA device."""
        spkr = Speaker(device=0)

        assert not spkr.is_started()

        spkr.start()

        assert spkr.is_started()
        pcm_instance = pcm_registry.get_last_instance()
        assert pcm_instance is not None

    def test_alsa_stop_closes_device(self, pcm_registry):
        """Test that stop() closes ALSA device."""
        spkr = Speaker(device=0)
        spkr.start()
        spkr.stop()

        assert not spkr.is_started()
        pcm_instance = pcm_registry.get_last_instance()
        assert pcm_instance.close.called


class TestALSASpeakerDeviceResolution:
    """Test ALSA device resolution."""

    def test_resolve_by_shorthand(self, mock_alsa_usb_speakers):
        """Test resolving device by integer index."""
        spkr = ALSASpeaker()
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

        spkr = ALSASpeaker(device=Speaker.USB_SPEAKER_1)
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

        spkr = ALSASpeaker(device=Speaker.USB_SPEAKER_2)
        assert spkr.device_stable_ref == "CARD=AnotherCard,DEV=0"

    def test_resolve_by_integer_index(self):
        """Test resolving device by integer index."""
        spkr = ALSASpeaker(device=0)
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

        spkr = ALSASpeaker(device=1)
        assert spkr.device_stable_ref == "CARD=AnotherCard,DEV=0"

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=[])
    def test_resolve_no_usb_devices_raises_error(self, mock_pcms):
        """Test that error is raised when no devices found."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            ALSASpeaker(device=0)

        assert "No ALSA speakers found" in str(exc_info.value)

    def test_resolve_out_of_range_raises_error(self, mock_alsa_usb_speakers):
        """Test that out of range index raises error."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            ALSASpeaker(device=5)

        assert "out of range" in str(exc_info.value)

    def test_resolve_explicit_device_name(self, mock_alsa_usb_speakers):
        """Test that explicit device names are passed through."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

        spkr = ALSASpeaker(device="plughw:CARD=SomeCard,DEV=0")
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

        spkr = ALSASpeaker(device="plughw:CARD=AnotherCard,DEV=0")
        assert spkr.device_stable_ref == "CARD=AnotherCard,DEV=0"


class TestALSAErrorManagement:
    """Test handling ALSA errors."""

    def test_device_busy_error(self):
        """Test that device busy error is properly reported."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError("Device or resource busy"),
            return_value=[],
        ):
            with pytest.raises(SpeakerOpenError) as exc_info:
                spkr.start()

        assert "busy" in str(exc_info.value).lower()

    def test_generic_alsa_error(self):
        """Test generic ALSA error handling."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError("Some generic ALSA error"),
            return_value=[],
        ):
            with pytest.raises(SpeakerOpenError):
                spkr.start()

    def test_write_error_doesnt_raise(self, pcm_registry):
        """Test that ALSA errors when writing don't raise exceptions."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.start()

        # Return ALSA error that's not disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.write = lambda data: -32  # EPIPE error

        audio_data = np.zeros(1024, dtype=np.int16)
        spkr.play(audio_data)  # Should not raise

    def test_stop_with_close_error(self, pcm_registry):
        """Test that stop handles close errors gracefully."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.start()

        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.close.side_effect = alsaaudio.ALSAAudioError("Close failed")

        # Should not raise
        spkr.stop()

        assert not spkr.is_started()


class TestALSADeviceDisconnection:
    """Test ALSA device disconnection handling."""

    def test_detect_device_disconnection(self, mock_alsa_usb_speakers, pcm_registry):
        """Test device disconnection detection during playback."""
        spkr = ALSASpeaker()
        spkr.start()

        # Simulate device disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.write = lambda data: None  # Simulate write failure

        with patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", side_effect=None, return_value=[]):
            # Attempt to write should detect disconnection
            audio_data = np.zeros(1024, dtype=np.int16)
            spkr.play(audio_data)  # Should handle disconnection gracefully

            assert spkr._pcm is None  # PCM should be cleared

    def test_list_devices_check(self, mock_alsa_usb_speakers):
        """Test device disconnection detection by enumerating devices."""
        spkr = ALSASpeaker()
        spkr.start()

        devices = ALSASpeaker.list_devices()
        assert len(devices) > 0

        # Simulate device removal
        with patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", side_effect=None, return_value=[]):
            devices = ALSASpeaker.list_devices()
            assert len(devices) == 0


class TestALSADeviceReconnection:
    """Test ALSA device reconnection logic."""

    def test_reconnection_after_device_available(self, mock_alsa_usb_speakers):
        """Test reconnection when device becomes available."""
        # Initially no devices - creation should fail
        with patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", side_effect=None, return_value=[]):
            with pytest.raises(SpeakerConfigError):
                spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")

        # Now creation and start should work
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.start()

        assert spkr.is_started()

        spkr.stop()

    def test_write_reconnects(self, mock_alsa_usb_speakers, pcm_registry):
        """Test write attempts reconnection after disconnection."""
        spkr = ALSASpeaker()
        spkr.start()

        # Simulate a disconnection
        pcm_instance = pcm_registry.get_last_instance()
        pcm_instance.write = lambda data: None

        audio_data = np.zeros(1024, dtype=np.int16)
        spkr.play(audio_data)

        # Mock successful reconnection
        pcm_instance.write = lambda data: len(data)

        # Playing a second time should trigger reconnection attempt
        spkr.play(audio_data)  # Should handle gracefully


class TestALSAPlayback:
    """Test ALSA speaker playback methods."""

    def test_alsa_speaker_play(self, mock_alsa_usb_speakers):
        """Test play with ALSA speaker."""
        spkr = ALSASpeaker()
        spkr.start()

        audio_data = np.zeros(1024, dtype=np.int16)
        spkr.play(audio_data)  # Should not raise

    @pytest.mark.parametrize(
        "format",
        [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.float32, np.float64],
    )
    def test_alsa_has_correct_format(self, mock_alsa_usb_speakers, pcm_registry, format):
        """Test that ALSA is configured with correct format."""
        format_dtype = np.dtype(format)

        spkr = ALSASpeaker(format=format, buffer_size=128)
        spkr.start()

        audio_data = np.zeros(128, dtype=format_dtype)
        spkr.play(audio_data)

        pcm_instance = pcm_registry.get_last_instance()
        assert format_dtype == _alsa_format_name_to_dtype(spkr.alsa_format_name)
        assert spkr.alsa_format_name == _dtype_to_alsa_format_name(format_dtype)
        assert spkr.alsa_format_idx == pcm_instance.info()["format"]
        assert spkr.alsa_format_name == pcm_instance.info()["format_name"]

    def test_unsupported_format_with_none_dtype(self):
        """Test that unsupported formats trigger an error."""
        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(format=None)

        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(format="unsupported_format")


class TestALSAVolumeControl:
    """Test ALSA speaker volume control."""

    def test_volume_default(self, mock_alsa_usb_speakers):
        """Test that default volume is 100."""
        spkr = ALSASpeaker()
        assert spkr.volume == 100

    def test_volume_setter(self, mock_alsa_usb_speakers):
        """Test setting volume."""
        spkr = ALSASpeaker()
        spkr.volume = 50
        assert spkr.volume == 50

        spkr.volume = 0
        assert spkr.volume == 0

        spkr.volume = 100
        assert spkr.volume == 100

    def test_volume_out_of_range(self, mock_alsa_usb_speakers):
        """Test that volume out of range raises error."""
        spkr = ALSASpeaker()

        with pytest.raises(ValueError):
            spkr.volume = -1

        with pytest.raises(ValueError):
            spkr.volume = 101

    def test_volume_affects_output(self, mock_alsa_usb_speakers, pcm_registry):
        """Test that volume changes affect audio output."""
        spkr = ALSASpeaker()
        spkr.start()
        spkr.volume = 50

        audio_data = np.full(1024, 1000, dtype=np.int16)
        spkr.play(audio_data)

        # Volume should scale the audio
        # (we can't directly test the output, but we verify no errors)


class TestALSASharedMode:
    """Test ALSA speaker shared mode."""

    def test_shared_mode_default(self, mock_alsa_usb_speakers):
        """Test that default shared mode is True."""
        spkr = ALSASpeaker()
        assert spkr.shared is True

    def test_exclusive_mode(self, mock_alsa_usb_speakers):
        """Test exclusive mode."""
        spkr = ALSASpeaker(shared=False)
        assert spkr.shared is False
        spkr.start()
        assert spkr.is_started()
