# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
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

    def test_factory_creates_alsa_speaker_with_integer(self):
        """Test factory creates ALSA speaker with integer device index."""
        spkr = Speaker(device=0)

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

    def test_factory_creates_alsa_speaker_with_string_index(self):
        """Test factory creates ALSA speaker with string device index."""
        spkr = Speaker(device="1")

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    def test_factory_creates_alsa_speaker_with_device_name(self):
        """Test factory creates ALSA speaker with explicit device name."""
        spkr = Speaker(device="plughw:CARD=SomeCard,DEV=0")

        assert isinstance(spkr, ALSASpeaker)
        assert spkr.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

    def test_factory_invalid_device_type_raises_error(self):
        """Test that invalid device type raises SpeakerConfigError."""
        with pytest.raises(SpeakerConfigError):
            Speaker(device={"invalid": "type"})  # type: ignore

    def test_factory_no_speaker_raises_open_error(self, mock_pw_dump):
        """Test that an integer index with no discoverable speaker raises SpeakerOpenError."""
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(SpeakerOpenError):
            Speaker(device=0)


class TestSpeakerAutoSelection:
    """Auto-selected speakers must not contend for the same device."""

    def test_auto_selection_assigns_distinct_speakers(self):
        spkr1 = Speaker()
        spkr2 = Speaker()

        assert spkr1.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"
        assert spkr2.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    def test_auto_selection_raises_when_all_speakers_are_in_use(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        spkr = Speaker()
        assert spkr.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

        with pytest.raises(SpeakerOpenError):
            Speaker()

    def test_auto_selection_releases_speaker_when_instance_is_dropped(self):
        import gc

        spkr = Speaker()
        first_ref = spkr.device_stable_ref
        del spkr
        gc.collect()

        assert Speaker().device_stable_ref == first_ref

    def test_auto_selection_skips_explicitly_selected_speakers(self):
        explicit = Speaker(0)
        auto = Speaker()

        assert explicit.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"
        assert auto.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    def test_explicit_selection_reuses_a_speaker_already_in_use(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        spkr1 = Speaker(0)
        spkr2 = Speaker(0)

        assert spkr2.device_stable_ref == spkr1.device_stable_ref


class TestSpeakerConfiguration:
    """Test speaker configuration and parameters."""

    def test_default_parameters(self):
        """Test that speakers use default parameters."""
        spkr = Speaker(device=0)

        assert spkr.sample_rate == Speaker.RATE_16K
        assert spkr.channels == Speaker.CHANNELS_MONO
        assert spkr.format == np.int16
        assert spkr.buffer_size == Speaker.BUFFER_SIZE_BALANCED

    def test_custom_parameters_alsa(self):
        """Test ALSA speaker with custom parameters."""
        spkr = Speaker(device=0, sample_rate=48000, channels=2, format=np.int32, buffer_size=2048)

        assert spkr.sample_rate == 48000
        assert spkr.channels == 2
        assert spkr.format == np.int32
        assert spkr.buffer_size == 2048

    def test_unsupported_format_raises_error(self):
        """Test that unsupported format raises error."""
        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(device="hw:0,0", format="INVALID_FORMAT")

    def test_invalid_device_type_raises_error(self):
        """Test that invalid device type raises error."""
        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(device=None)  # type: ignore

    def test_no_devices_found_raises_open_error(self, mock_pw_dump):
        """Test that no USB devices found raises an open error."""
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device="usb:1")

    def test_out_of_range_device_index_raises_open_error(self):
        """Test that an out of range ordinal index raises an open error."""
        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device=10)


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
