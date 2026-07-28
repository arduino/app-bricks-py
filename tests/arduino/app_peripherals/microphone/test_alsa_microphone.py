# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
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
    """Test ALSA device resolution to a full, stable ALSA path."""

    @pytest.mark.parametrize(
        "device, expected",
        [
            (0, "plughw:CARD=SomeCard,DEV=0"),  # Ordinal -> n-th plugged microphone
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
        assert ALSAMicrophone(device=device).device_stable_ref == expected

    def test_default_device_resolves_to_first_plugged(self):
        """Test that the default device selects the first plugged microphone."""
        assert ALSAMicrophone().device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

    def test_resolve_by_id_symlink(self):
        """Test resolving a /dev/snd/by-id symlink to a full ALSA path."""
        with (
            patch("arduino.app_peripherals.microphone.alsa_microphone.os.path.exists", return_value=True),
            patch("arduino.app_peripherals.microphone.alsa_microphone.os.path.realpath", return_value="/dev/snd/controlC1"),
        ):
            mic = ALSAMicrophone(device="/dev/snd/by-id/usb-Some-Mic-00")

        assert mic.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    @pytest.mark.parametrize(
        "device, message",
        [
            (None, "Invalid device type"),  # Wrong type
            ("not-a-real-device", "Unsupported device identifier"),  # Unrecognized format
        ],
    )
    def test_bad_parameter_raises_config_error(self, device, message):
        """Test that a wrong or unsupported device parameter raises a config error."""
        with pytest.raises(MicrophoneConfigError) as exc_info:
            ALSAMicrophone(device=device)

        assert message in str(exc_info.value)

    @pytest.mark.parametrize(
        "device, message",
        [
            (5, "No available microphones found"),  # Ordinal beyond the plugged microphones
            ("CARD=Ghost,DEV=0", "not found among available"),  # Well-formed but disconnected
        ],
    )
    def test_unavailable_device_raises_open_error(self, device, message):
        """Test that a missing or unavailable device raises an open error."""
        with pytest.raises(MicrophoneOpenError) as exc_info:
            ALSAMicrophone(device=device)

        assert message in str(exc_info.value)

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=[])
    def test_no_alsa_devices_raises_open_error(self, mock_pcms):
        """Test that an explicit device with no ALSA capture devices present raises an open error."""
        with pytest.raises(MicrophoneOpenError) as exc_info:
            ALSAMicrophone(device="CARD=SomeCard,DEV=0")

        assert "No ALSA microphones found" in str(exc_info.value)

    def test_ordinal_with_no_plugged_mic_raises_open_error(self, mock_pw_dump):
        """Test that an ordinal index with no plugged microphone raises an open error."""
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(MicrophoneOpenError):
            ALSAMicrophone(device=0)


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
        """Test that ALSA errors while opening surface as MicrophoneOpenError."""
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.auto_reconnect_delay = 0

        with patch(
            "arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM",
            side_effect=alsaaudio.ALSAAudioError(alsa_message),
        ):
            with pytest.raises(MicrophoneOpenError) as exc_info:
                mic.start()

        if expected:
            assert expected in str(exc_info.value).lower()

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

    def test_detect_device_disconnection(self, pcm_registry):
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

    def test_list_devices_enumerates_via_pw_dump(self, mock_pw_dump, monkeypatch):
        """Test that the public list_devices() reports microphones discovered via pw-dump."""
        monkeypatch.delenv("CONFIGURED_CARRIERS", raising=False)

        # Default fixture: two USB sources.
        devices = ALSAMicrophone.list_devices()
        assert devices == ["plughw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0"]

        # No discoverable sources -> empty list.
        mock_pw_dump(usb_ids=(), builtin_ids=())
        assert ALSAMicrophone.list_devices() == []


class TestALSADeviceReconnection:
    """Test ALSA device reconnection logic."""

    def test_reconnection_after_device_available(self):
        """Test reconnection when device becomes available."""
        # Initially no devices - creation should fail
        with patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", side_effect=None, return_value=[]):
            with pytest.raises(MicrophoneOpenError):
                mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")

        # Now creation and start should work
        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        mic.start()

        assert mic.is_started()

        mic.stop()

    def test_read_reconnects(self, pcm_registry):
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

    def test_alsa_microphone_capture(self):
        """Test capture with ALSA microphone."""
        mic = ALSAMicrophone()
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) == 1024

    def test_alsa_microphone_stream(self):
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
    def test_alsa_has_correct_format(self, pcm_registry, format):
        """Test that ALSA is configured with correct format and that format's dtype is returned."""
        format_dtype = np.dtype(format)

        mic = ALSAMicrophone(format=format, buffer_size=128)
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert chunk.dtype == format_dtype

        pcm_instance = pcm_registry.get_last_instance()
        assert format_dtype == _alsa_format_name_to_dtype(mic._alsa_format_name)
        assert mic._alsa_format_name == _dtype_to_alsa_format_name(format_dtype)
        assert mic._alsa_format_idx == pcm_instance.info()["format"]
        assert mic._alsa_format_name == pcm_instance.info()["format_name"]

    def test_unsupported_format_with_none_dtype(self):
        """Test that unsupported formats trigger an error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(format=None)  # type: ignore

        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(format="unsupported_format")


class TestALSAMicrophoneUsbDiscovery:
    """USB resolution backed by pw-dump (ordered by PipeWire node id)."""

    def test_list_usb_devices_returns_full_alsa_paths(self):
        # Default fixture: two USB sources (node ids 50, 60 -> cards 0, 1).
        assert ALSAMicrophone.list_usb_devices() == ["plughw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0"]

    @pytest.mark.parametrize(
        "device, expected",
        [
            (Microphone.USB_MIC_1, "plughw:CARD=SomeCard,DEV=0"),  # "usb:1"
            (Microphone.USB_MIC_2, "plughw:CARD=AnotherCard,DEV=0"),  # "usb:2"
        ],
    )
    def test_usb_shorthand_resolves_to_full_path(self, device, expected):
        assert ALSAMicrophone(device=device).device_stable_ref == expected

    def test_usb_out_of_range_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        with pytest.raises(MicrophoneOpenError):
            ALSAMicrophone(device="usb:2")

    def test_listed_device_is_accepted_as_input(self):
        # A path returned by list_devices() round-trips back in unchanged.
        for listed in ALSAMicrophone.list_devices():
            assert ALSAMicrophone(device=listed).device_stable_ref == listed


class TestALSAMicrophoneJackResolution:
    """Built-in (jack) resolution to a pipewire:NODE stable ref, gated on media-carrier."""

    @pytest.fixture
    def media_carrier(self, monkeypatch):
        monkeypatch.setenv("CONFIGURED_CARRIERS", "media-carrier")

    @pytest.fixture(autouse=True)
    def _clear_carrier(self, monkeypatch):
        monkeypatch.delenv("CONFIGURED_CARRIERS", raising=False)

    def test_list_jack_devices_requires_media_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        # Without media carrier the built-in mic is not exposed.
        assert ALSAMicrophone.list_jack_devices() == []

    def test_list_jack_devices_under_media_carrier(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert ALSAMicrophone.list_jack_devices() == ["pipewire:NODE=alsa_input.platform-sound.Source-52"]

    def test_list_devices_combines_usb_and_jack(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert ALSAMicrophone.list_devices() == ["plughw:CARD=SomeCard,DEV=0", "pipewire:NODE=alsa_input.platform-sound.Source-52"]

    def test_jack_resolves_to_pipewire_node(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        mic = ALSAMicrophone(device="jack:1")
        assert mic.device_stable_ref == "pipewire:NODE=alsa_input.platform-sound.Source-52"

    def test_jack_name_uses_pipewire_description(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        mic = ALSAMicrophone(device="jack:1")
        assert mic.name == "Built-in Audio 52"

    def test_jack_name_falls_back_to_node_ref_without_description(self):
        # An explicit pipewire node absent from pw-dump keeps the technical ref as name.
        mic = ALSAMicrophone(device="pipewire:NODE=unknown.node")
        assert mic.name == "pipewire:NODE=unknown.node"

    def test_jack_opens_pipewire_device(self, mock_pw_dump, media_carrier, pcm_registry):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        mic = ALSAMicrophone(device="jack:1")
        mic.start()

        assert pcm_registry.get_last_instance().device == "pipewire:NODE=alsa_input.platform-sound.Source-52"
        assert mic._is_device_disconnected() is False

    def test_second_jack_unsupported(self, mock_pw_dump, media_carrier):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            ALSAMicrophone(device="jack:2")

    def test_jack_off_media_carrier_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            ALSAMicrophone(device="jack:1")

    def test_ordinal_falls_back_to_jack_under_media_carrier(self, mock_pw_dump, media_carrier):
        # No USB, one built-in: the first plugged mic is the jack device.
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert ALSAMicrophone(device=0).device_stable_ref == "pipewire:NODE=alsa_input.platform-sound.Source-52"

    def test_ordinal_no_jack_fallback_off_media_carrier(self, mock_pw_dump):
        # Without media carrier there is no jack fallback, so an ordinal with no USB raises.
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            ALSAMicrophone(device=0)

    def test_explicit_pipewire_node_passthrough(self, pcm_registry):
        mic = ALSAMicrophone(device="pipewire:NODE=unknown.node")
        assert mic.device_stable_ref == "pipewire:NODE=unknown.node"

        mic.start()
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
        mic = ALSAMicrophone(device=device)
        assert mic.device_stable_ref == device

        mic.start()

        # The stable ref is handed to ALSA verbatim, no plug_card_* remapping.
        assert pcm_registry.get_last_instance().device == device
        # PipeWire devices are always considered present.
        assert mic._is_device_disconnected() is False
