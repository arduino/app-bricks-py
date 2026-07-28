# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
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
    """Test ALSA device resolution to a full, stable ALSA path."""

    @pytest.mark.parametrize(
        "device, expected",
        [
            (0, "plughw:CARD=SomeCard,DEV=0"),  # Ordinal -> n-th plugged speaker
            (1, "plughw:CARD=AnotherCard,DEV=0"),
            ("CARD=SomeCard,DEV=0", "plughw:CARD=SomeCard,DEV=0"),
            ("plughw:CARD=SomeCard,DEV=0", "plughw:CARD=SomeCard,DEV=0"),
            ("hw:1,0", "plughw:CARD=AnotherCard,DEV=0"),
            ("hw:0,0,0", "hw:0,0,0"),  # Fully-specified raw paths pass through as-is
            ("plughw:SomeCard,0,0", "plughw:SomeCard,0,0"),
        ],
    )
    def test_resolves_to_stable_ref(self, device, expected):
        """Test that supported identifiers resolve to a full ALSA path."""
        assert ALSASpeaker(device=device).device_stable_ref == expected

    def test_default_device_resolves_to_first_plugged(self):
        """Test that the default device selects the first plugged speaker."""
        assert ALSASpeaker().device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

    def test_resolve_by_id_symlink(self):
        """Test resolving a /dev/snd/by-id symlink to a full ALSA path."""
        with (
            patch("arduino.app_peripherals.speaker.alsa_speaker.os.path.exists", return_value=True),
            patch("arduino.app_peripherals.speaker.alsa_speaker.os.path.realpath", return_value="/dev/snd/controlC1"),
        ):
            spkr = ALSASpeaker(device="/dev/snd/by-id/usb-Some-Spk-00")

        assert spkr.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    @pytest.mark.parametrize(
        "device, message",
        [
            (None, "Invalid device type"),  # Wrong type
            ("not-a-real-device", "Unsupported device identifier"),  # Unrecognized format
        ],
    )
    def test_bad_parameter_raises_config_error(self, device, message):
        """Test that a wrong or unsupported device parameter raises a config error."""
        with pytest.raises(SpeakerConfigError) as exc_info:
            ALSASpeaker(device=device)

        assert message in str(exc_info.value)

    @pytest.mark.parametrize(
        "device, message",
        [
            (5, "No speaker found at index 5"),  # Ordinal beyond the plugged speakers
            ("CARD=Ghost,DEV=0", "not found among available"),  # Well-formed but disconnected
        ],
    )
    def test_unavailable_device_raises_open_error(self, device, message):
        """Test that a missing or unavailable device raises an open error."""
        with pytest.raises(SpeakerOpenError) as exc_info:
            ALSASpeaker(device=device)

        assert message in str(exc_info.value)

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=[])
    def test_no_alsa_devices_raises_open_error(self, mock_pcms):
        """Test that an explicit device with no ALSA playback devices present raises an open error."""
        with pytest.raises(SpeakerOpenError) as exc_info:
            ALSASpeaker(device="CARD=SomeCard,DEV=0")

        assert "No ALSA speakers found" in str(exc_info.value)

    def test_ordinal_with_no_plugged_speaker_raises_open_error(self, mock_pw_dump):
        """Test that an ordinal index with no plugged speaker raises an open error."""
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device=0)


class TestALSAErrorManagement:
    """Test handling ALSA errors."""

    @pytest.mark.parametrize(
        "alsa_message, expected",
        [
            ("Device or resource busy", "busy"),  # Surfaced with a dedicated hint
            ("Some generic ALSA error", None),  # Any other ALSA error still raises
        ],
    )
    def test_open_alsa_error_raises(self, alsa_message, expected):
        """Test that ALSA errors while opening surface as SpeakerOpenError."""
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError(alsa_message),
        ):
            with pytest.raises(SpeakerOpenError) as exc_info:
                spkr.start()

        if expected:
            assert expected in str(exc_info.value).lower()

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

    def test_detect_device_disconnection(self, pcm_registry):
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

    def test_list_devices_enumerates_via_pw_dump(self, mock_pw_dump, monkeypatch):
        """Test that the public list_devices() reports speakers discovered via pw-dump."""
        monkeypatch.delenv("CONFIGURED_CARRIERS", raising=False)

        # Default fixture: two USB sinks.
        devices = ALSASpeaker.list_devices()
        assert devices == ["plughw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0"]

        # No discoverable sinks -> empty list.
        mock_pw_dump(usb_ids=(), builtin_ids=())
        assert ALSASpeaker.list_devices() == []


class TestALSADeviceReconnection:
    """Test ALSA device reconnection logic."""

    def test_reconnection_after_device_available(self):
        """Test reconnection when device becomes available."""
        # Initially no devices - creation should fail
        with patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", side_effect=None, return_value=[]):
            with pytest.raises(SpeakerOpenError):
                spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")

        # Now creation and start should work
        spkr = ALSASpeaker(device="CARD=SomeCard,DEV=0")
        spkr.start()

        assert spkr.is_started()

        spkr.stop()

    def test_write_reconnects(self, pcm_registry):
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

    def test_alsa_speaker_play(self):
        """Test play with ALSA speaker."""
        spkr = ALSASpeaker()
        spkr.start()

        audio_data = np.zeros(1024, dtype=np.int16)
        spkr.play(audio_data)  # Should not raise

    @pytest.mark.parametrize(
        "format",
        [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.float32, np.float64],
    )
    def test_alsa_has_correct_format(self, pcm_registry, format):
        """Test that ALSA is configured with correct format."""
        format_dtype = np.dtype(format)

        spkr = ALSASpeaker(format=format, buffer_size=128)
        spkr.start()

        audio_data = np.zeros(128, dtype=format_dtype)
        spkr.play(audio_data)

        pcm_instance = pcm_registry.get_last_instance()
        assert format_dtype == _alsa_format_name_to_dtype(spkr._alsa_format_name)
        assert spkr._alsa_format_name == _dtype_to_alsa_format_name(format_dtype)
        assert spkr._alsa_format_idx == pcm_instance.info()["format"]
        assert spkr._alsa_format_name == pcm_instance.info()["format_name"]

    def test_unsupported_format_with_none_dtype(self):
        """Test that unsupported formats trigger an error."""
        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(format=None)

        with pytest.raises(SpeakerConfigError):
            ALSASpeaker(format="unsupported_format")


class TestALSAVolumeControl:
    """Test ALSA speaker volume control."""

    def test_volume_default(self):
        """Test that default volume is 100."""
        spkr = ALSASpeaker()
        assert spkr.volume == 100

    def test_volume_setter(self):
        """Test setting volume."""
        spkr = ALSASpeaker()
        spkr.volume = 50
        assert spkr.volume == 50

        spkr.volume = 0
        assert spkr.volume == 0

        spkr.volume = 100
        assert spkr.volume == 100

    def test_volume_out_of_range(self):
        """Test that volume out of range raises error."""
        spkr = ALSASpeaker()

        with pytest.raises(ValueError):
            spkr.volume = -1

        with pytest.raises(ValueError):
            spkr.volume = 101

    def test_volume_affects_output(self, pcm_registry):
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

    def test_shared_mode_default(self):
        """Test that default shared mode is True."""
        spkr = ALSASpeaker()
        assert spkr.shared is True

    def test_exclusive_mode(self):
        """Test exclusive mode."""
        spkr = ALSASpeaker(shared=False)
        assert spkr.shared is False
        spkr.start()
        assert spkr.is_started()


class TestALSASpeakerUsbDiscovery:
    """USB resolution backed by pw-dump (ordered by PipeWire node id)."""

    def test_list_usb_devices_returns_full_alsa_paths(self):
        # Default fixture: two USB sinks (node ids 50, 60 -> cards 0, 1).
        assert ALSASpeaker.list_usb_devices() == ["plughw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0"]

    @pytest.mark.parametrize(
        "device, expected",
        [
            (Speaker.USB_SPEAKER_1, "plughw:CARD=SomeCard,DEV=0"),  # "usb:1"
            (Speaker.USB_SPEAKER_2, "plughw:CARD=AnotherCard,DEV=0"),  # "usb:2"
        ],
    )
    def test_usb_shorthand_resolves_to_full_path(self, device, expected):
        assert ALSASpeaker(device=device).device_stable_ref == expected

    def test_usb_out_of_range_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device="usb:2")

    def test_listed_device_is_accepted_as_input(self):
        # A path returned by list_devices() round-trips back in unchanged.
        for listed in ALSASpeaker.list_devices():
            assert ALSASpeaker(device=listed).device_stable_ref == listed


class TestALSASpeakerJackResolution:
    """Built-in (jack) resolution to a pipewire:NODE stable ref, gated on media-carrier."""

    @pytest.fixture
    def media_carrier(self, monkeypatch):
        monkeypatch.setenv("CONFIGURED_CARRIERS", "media-carrier")

    @pytest.fixture(autouse=True)
    def _clear_carrier(self, monkeypatch):
        monkeypatch.delenv("CONFIGURED_CARRIERS", raising=False)

    def test_list_jack_devices_requires_media_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        # Without media carrier the built-in speaker is not exposed.
        assert ALSASpeaker.list_jack_devices() == []

    def test_list_jack_devices_under_media_carrier(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert ALSASpeaker.list_jack_devices() == ["pipewire:NODE=alsa_output.platform-sound.Sink-52"]

    def test_list_devices_combines_usb_and_jack(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert ALSASpeaker.list_devices() == ["plughw:CARD=SomeCard,DEV=0", "pipewire:NODE=alsa_output.platform-sound.Sink-52"]

    def test_jack_resolves_to_pipewire_node(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        spkr = ALSASpeaker(device="jack:1")
        assert spkr.device_stable_ref == "pipewire:NODE=alsa_output.platform-sound.Sink-52"

    def test_jack_name_uses_pipewire_description(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        spkr = ALSASpeaker(device="jack:1")
        assert spkr.name == "Built-in Audio 52"

    def test_jack_name_falls_back_to_node_ref_without_description(self):
        # An explicit pipewire node absent from pw-dump keeps the technical ref as name.
        spkr = ALSASpeaker(device="pipewire:NODE=unknown.node")
        assert spkr.name == "pipewire:NODE=unknown.node"

    def test_jack_opens_pipewire_device(self, mock_pw_dump, media_carrier, pcm_registry):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        spkr = ALSASpeaker(device="jack:1")
        spkr.start()

        assert pcm_registry.get_last_instance().device == "pipewire:NODE=alsa_output.platform-sound.Sink-52"
        assert spkr._is_device_disconnected() is False

    def test_second_jack_unsupported(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device="jack:2")

    def test_jack_off_media_carrier_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device="jack:1")

    def test_ordinal_falls_back_to_jack_under_media_carrier(self, mock_pw_dump, media_carrier):
        # No USB, one built-in: the first plugged speaker is the jack device.
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert ALSASpeaker(device=0).device_stable_ref == "pipewire:NODE=alsa_output.platform-sound.Sink-52"

    def test_ordinal_no_jack_fallback_off_media_carrier(self, mock_pw_dump):
        # Without media carrier there is no jack fallback, so an ordinal with no USB raises.
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            ALSASpeaker(device=0)

    def test_explicit_pipewire_node_passthrough(self, pcm_registry):
        spkr = ALSASpeaker(device="pipewire:NODE=unknown.node")
        assert spkr.device_stable_ref == "pipewire:NODE=unknown.node"

        spkr.start()
        assert pcm_registry.get_last_instance().device == "pipewire:NODE=unknown.node"

    @pytest.mark.parametrize(
        "device",
        [
            "pipewire",  # Bare default PipeWire device
            "pipewire:NODE=unknown.node",  # Explicit PipeWire node
        ],
    )
    def test_pipewire_opens_as_direct_device(self, pcm_registry, device):
        """Both the bare 'pipewire' string and 'pipewire:NODE=...' open as-is without runtime resolution."""
        spkr = ALSASpeaker(device=device)
        assert spkr.device_stable_ref == device

        spkr.start()

        # The stable ref is handed to ALSA verbatim, no plug_card_* remapping.
        assert pcm_registry.get_last_instance().device == device
        # PipeWire devices are always considered present.
        assert spkr._is_device_disconnected() is False
