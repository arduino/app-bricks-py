# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.microphone.errors import MicrophoneOpenError
from arduino.app_peripherals.microphone.utils import claim_first_available_microphone, list_audio_sources, nth_plugged_microphone

_CARRIER_ENV = "CONFIGURED_CARRIERS"


@pytest.fixture(autouse=True)
def _no_carrier(monkeypatch):
    """Default to a non-media-carrier environment unless a test opts in."""
    monkeypatch.delenv(_CARRIER_ENV, raising=False)


class TestNthPluggedMicrophone:
    """External behavior of nth_plugged_microphone device resolution."""

    def test_usb_first_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert nth_plugged_microphone(0) == "usb:1"

    def test_out_of_range_without_carrier_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        # Second position has no USB device and there is no built-in fallback without media carrier.
        with pytest.raises(MicrophoneOpenError):
            nth_plugged_microphone(1)

    def test_usb_precedence_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert nth_plugged_microphone(0) == "usb:1"

    def test_jack_fallback_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert nth_plugged_microphone(0) == "jack:1"

    def test_second_builtin_unsupported_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        # Only one built-in microphone is supported, so jack:2 is out of range.
        with pytest.raises(MicrophoneOpenError):
            nth_plugged_microphone(1)

    def test_no_jack_fallback_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            nth_plugged_microphone(0)

    def test_no_jack_fallback_for_other_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "some-other-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            nth_plugged_microphone(0)

    def test_no_devices_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(MicrophoneOpenError):
            nth_plugged_microphone(0)


class TestClaimFirstAvailableMicrophone:
    """Claim-aware device resolution used by Microphone auto-selection."""

    def test_skips_already_claimed_microphones(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50, 60))

        assert claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"
        assert claim_first_available_microphone() == "plughw:CARD=AnotherCard,DEV=0"

    def test_falls_back_to_jack_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"
        assert claim_first_available_microphone() == "pipewire:NODE=alsa_input.platform-sound.Source-52"

    def test_raises_when_all_microphones_are_claimed(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        claim_first_available_microphone()
        with pytest.raises(MicrophoneOpenError):
            claim_first_available_microphone()

    def test_raises_when_no_microphone_is_plugged(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(MicrophoneOpenError):
            claim_first_available_microphone()


class TestListAudioSources:
    """Discovery contract: USB/built-in partitioning ordered by PipeWire node id."""

    def test_partitions_usb_and_builtin(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        usb, builtin = list_audio_sources()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [52]

    def test_orders_by_ascending_node_id(self, mock_pw_dump):
        # Declared out of order; discovery must sort by node id (lowest first).
        mock_pw_dump(usb_ids=(60, 50), builtin_ids=())

        usb, _ = list_audio_sources()

        assert [s["id"] for s in usb] == [50, 60]

    def test_classifies_non_usb_bus_as_builtin(self):
        # A source is USB only when its parent device reports device.bus == "usb".
        objects = [
            {"id": 100, "info": {"props": {"media.class": "Audio/Device", "device.bus-path": "platform-sound"}}},
            {
                "id": 50,
                "info": {
                    "props": {
                        "media.class": "Audio/Source",
                        "node.name": "alsa_input.platform-sound.Source-50",
                        "device.id": 100,
                    }
                },
            },
        ]
        with patch("arduino.app_peripherals.microphone.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout=json.dumps(objects))
            usb, builtin = list_audio_sources()

        assert usb == []
        assert [s["id"] for s in builtin] == [50]


class TestPwDumpFailures:
    """pw-dump errors surface as MicrophoneOpenError."""

    def test_missing_binary_raises(self):
        with patch("arduino.app_peripherals.microphone.utils.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(MicrophoneOpenError):
                nth_plugged_microphone(0)

    def test_invalid_json_raises(self):
        with patch("arduino.app_peripherals.microphone.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout="not-json")
            with pytest.raises(MicrophoneOpenError):
                nth_plugged_microphone(0)
