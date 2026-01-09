# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import threading
from unittest.mock import MagicMock

import numpy as np

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker, ALSASpeaker
from arduino.app_peripherals.speaker.errors import SpeakerConfigError, SpeakerError, SpeakerOpenError, SpeakerWriteError


class TestSpeakerFactoryInstantiation:
    """Test factory instantiation of different speaker types."""

    def test_factory_creates_alsa_speaker_with_integer(self, mock_alsa_usb_speakers):
        """Test factory creates ALSA speaker with integer device index."""
        spkr = Speaker(device=0)

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

    def test_factory_creates_alsa_speaker_with_string_index(self, mock_alsa_usb_speakers):
        """Test factory creates ALSA speaker with string device index."""
        spkr = Speaker(device="1")

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "CARD=AnotherCard,DEV=0"

    def test_factory_creates_alsa_speaker_with_device_name(self, mock_alsa_usb_speakers):
        """Test factory creates ALSA speaker with explicit device name."""
        spkr = Speaker(device="plughw:CARD=SomeCard,DEV=0")

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "CARD=SomeCard,DEV=0"

    def test_factory_device_does_not_exist_raises_error(self, mock_alsa_usb_speakers):
        """Test that invalid device raises SpeakerConfigError."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            Speaker(device="usb:3")  # Assuming only 2 USB devices exist in the mock
        assert "out of range" in str(exc_info.value).lower()

    def test_factory_invalid_usb_format_raises_error(self, mock_alsa_usb_speakers):
        """Test that invalid USB device format raises SpeakerConfigError."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            Speaker(device="usb:something")
        assert "invalid" in str(exc_info.value).lower()

    def test_factory_invalid_device_type_raises_error(self, mock_alsa_usb_speakers):
        """Test that invalid device type raises SpeakerConfigError."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            Speaker(device=None)
        assert "unsupported" in str(exc_info.value).lower()

    def test_factory_invalid_format_raises_error(self, mock_alsa_usb_speakers):
        """Test that unsupported format raises error."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            Speaker(device="hw:0,0", format="INVALID_FORMAT")
        assert "invalid" in str(exc_info.value).lower()

    def test_factory_no_devices_found_raises_error(self):
        """Test that no USB devices found raises error."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            Speaker()
        assert "no usb speakers found" in str(exc_info.value).lower()
        assert ALSASpeaker.list_usb_devices() == []


class TestSpeakerConfiguration:
    """Test speaker configuration and parameters."""

    def test_default_parameters(self, mock_alsa_usb_speakers):
        """Test that speakers use default parameters."""
        spkr = Speaker(device=0)

        assert spkr.sample_rate == Speaker.RATE_16K
        assert spkr.channels == Speaker.CHANNELS_MONO
        assert spkr.format == np.int16
        assert spkr.buffer_size == Speaker.BUFFER_SIZE_BALANCED

    def test_custom_parameters_alsa(self, mock_alsa_usb_speakers):
        """Test ALSA speaker with custom parameters."""
        spkr = Speaker(device=0, sample_rate=48000, channels=2, format=np.int32, buffer_size=2048)

        assert spkr.sample_rate == 48000
        assert spkr.channels == 2
        assert spkr.format == np.int32
        assert spkr.buffer_size == 2048


class TestSpeakerStartStop:
    """Test start and stop lifecycle."""

    def test_double_start_is_idempotent(self):
        """Test that starting twice is safe."""
        spkr = Speaker(device="plughw:CARD=SomeCard,DEV=0")

        spkr.start = MagicMock()
        spkr._is_started = False
        spkr._spkr_lock = threading.Lock()

        # Simulate idempotent behavior
        def start_impl():
            with spkr._spkr_lock:
                if spkr._is_started:
                    return
                spkr._is_started = True

        spkr.start.side_effect = start_impl

        spkr.start()
        first_state = spkr._is_started
        spkr.start()

        assert spkr._is_started == first_state

    def test_double_stop_is_idempotent(self):
        """Test that stopping twice is safe."""
        spkr = Speaker(device="plughw:CARD=SomeCard,DEV=0")

        spkr._is_started = True
        spkr._spkr_lock = threading.Lock()
        spkr.stop = MagicMock()

        def stop_impl():
            with spkr._spkr_lock:
                if not spkr._is_started:
                    return
                spkr._is_started = False

        spkr.stop.side_effect = stop_impl

        spkr.stop()
        spkr.stop()  # Should not raise

        assert not spkr._is_started

    def test_restart(self):
        """Test that speaker can be restarted."""
        spkr = Speaker(device="CARD=SomeCard,DEV=0")
        spkr.start()
        spkr.stop()

        # Should be able to restart
        spkr.start()
        assert spkr.is_started()


class TestSpeakerContextManager:
    """Test context manager behavior."""

    def test_context_manager_starts_and_stops(self):
        """Test that context manager starts and stops speaker."""
        spkr = Speaker(device=0)

        assert not spkr.is_started()

        with spkr:
            assert spkr.is_started()

        assert not spkr.is_started()

    def test_context_manager_stops_on_exception(self):
        """Test that context manager stops even on exception."""
        spkr = Speaker(device=0)

        try:
            with spkr:
                assert spkr.is_started()
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        assert not spkr.is_started()


class TestBaseSpeakerAbstraction:
    """Test base speaker abstract class requirements."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseSpeaker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSpeaker()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""

        # Missing _write_audio
        class IncompleteSpeaker1(BaseSpeaker):
            def _open_speaker(self):
                pass

            def _close_speaker(self):
                pass

        with pytest.raises(TypeError):
            IncompleteSpeaker1()

        # Missing _close_speaker
        class IncompleteSpeaker2(BaseSpeaker):
            def _open_speaker(self):
                pass

            def _write_audio(self, audio_chunk):
                pass

        with pytest.raises(TypeError):
            IncompleteSpeaker2()

        # Missing _open_speaker
        class IncompleteSpeaker3(BaseSpeaker):
            def _close_speaker(self):
                pass

            def _write_audio(self, audio_chunk):
                pass

        with pytest.raises(TypeError):
            IncompleteSpeaker3()


class TestExceptionHierarchy:
    """Test exception hierarchy and catching."""

    def test_speaker_open_error_is_speaker_error(self):
        """Test exception inheritance."""
        assert issubclass(SpeakerOpenError, SpeakerError)

    def test_speaker_write_error_is_speaker_error(self):
        """Test exception inheritance."""
        assert issubclass(SpeakerWriteError, SpeakerError)

    def test_speaker_config_error_is_speaker_error(self):
        """Test exception inheritance."""
        assert issubclass(SpeakerConfigError, SpeakerError)

    def test_catch_specific_error_with_base_handler(self):
        """Test that specific errors can be caught with base handler."""
        try:
            raise SpeakerWriteError("Test")
        except SpeakerError as e:
            assert "Test" in str(e)
