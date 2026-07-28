# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.speaker.errors import SpeakerOpenError
from arduino.app_peripherals.speaker.utils import _claim_first_available_speaker, list_audio_sinks, _nth_plugged_speaker

_CARRIER_ENV = "CONFIGURED_CARRIERS"


@pytest.fixture(autouse=True)
def _no_carrier(monkeypatch):
    """Default to a non-media-carrier environment unless a test opts in."""
    monkeypatch.delenv(_CARRIER_ENV, raising=False)


class TestNthPluggedSpeaker:
    """External behavior of _nth_plugged_speaker device resolution."""

    def test_usb_first_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_speaker(0) == "usb:1"

    def test_out_of_range_without_carrier_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        # Second position has no USB device and there is no built-in fallback without media carrier.
        with pytest.raises(SpeakerOpenError):
            _nth_plugged_speaker(1)

    def test_usb_precedence_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_speaker(0) == "usb:1"

    def test_jack_fallback_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert _nth_plugged_speaker(0) == "jack:1"

    def test_index_spans_usb_then_jack_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_speaker(1) == "jack:1"

    def test_second_jack_is_addressable_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52, 54))

        assert _nth_plugged_speaker(1) == "jack:2"

    def test_out_of_range_jack_index_raises_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            _nth_plugged_speaker(1)

    def test_no_jack_fallback_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            _nth_plugged_speaker(0)

    def test_no_jack_fallback_for_other_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "some-other-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(SpeakerOpenError):
            _nth_plugged_speaker(0)

    def test_no_devices_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(SpeakerOpenError):
            _nth_plugged_speaker(0)


class TestClaimFirstAvailableSpeaker:
    """Claim-aware device resolution used by Speaker auto-selection."""

    def test_skips_already_claimed_speakers(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50, 60))

        assert _claim_first_available_speaker() == "plughw:CARD=SomeCard,DEV=0"
        assert _claim_first_available_speaker() == "plughw:CARD=AnotherCard,DEV=0"

    def test_falls_back_to_jack_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _claim_first_available_speaker() == "plughw:CARD=SomeCard,DEV=0"
        assert _claim_first_available_speaker() == "pipewire:NODE=alsa_output.platform-sound.Sink-52"

    def test_raises_when_all_speakers_are_claimed(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        _claim_first_available_speaker()
        with pytest.raises(SpeakerOpenError):
            _claim_first_available_speaker()

    def test_raises_when_no_speaker_is_plugged(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(SpeakerOpenError):
            _claim_first_available_speaker()

    def test_never_selects_bluetooth_or_hdmi_speakers(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(builtin_ids=(54,), bluetooth_ids=(50,), hdmi_ids=(52,))

        assert _claim_first_available_speaker() == "pipewire:NODE=alsa_output.platform-sound.Sink-54"


class TestListAudioSinks:
    """Discovery contract: USB/built-in partitioning ordered by PipeWire node id."""

    def test_partitions_usb_and_builtin(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        usb, builtin = list_audio_sinks()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [52]

    def test_excludes_bluetooth_sinks(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(54,), bluetooth_ids=(52,))

        usb, builtin = list_audio_sinks()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [54]

    def test_excludes_hdmi_sinks(self, mock_pw_dump):
        # The HDMI node has the lowest id: exclusion, not ordering, must keep it out.
        mock_pw_dump(builtin_ids=(54,), hdmi_ids=(50,))

        usb, builtin = list_audio_sinks()

        assert usb == []
        assert [s["id"] for s in builtin] == [54]

    def test_orders_by_ascending_node_id(self, mock_pw_dump):
        # Declared out of order; discovery must sort by node id (lowest first).
        mock_pw_dump(usb_ids=(60, 50), builtin_ids=())

        usb, _ = list_audio_sinks()

        assert [s["id"] for s in usb] == [50, 60]

    def test_classifies_non_usb_bus_as_builtin(self):
        # A sink is USB only when its parent device reports device.bus == "usb".
        objects = [
            {"id": 100, "info": {"props": {"media.class": "Audio/Device", "device.bus-path": "platform-sound"}}},
            {
                "id": 50,
                "info": {
                    "props": {
                        "media.class": "Audio/Sink",
                        "node.name": "alsa_output.platform-sound.Sink-50",
                        "device.id": 100,
                    }
                },
            },
        ]
        with patch("arduino.app_peripherals.speaker.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout=json.dumps(objects))
            usb, builtin = list_audio_sinks()

        assert usb == []
        assert [s["id"] for s in builtin] == [50]


class TestPwDumpFailures:
    """pw-dump errors surface as SpeakerOpenError."""

    def test_missing_binary_raises(self):
        with patch("arduino.app_peripherals.speaker.utils.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(SpeakerOpenError):
                _nth_plugged_speaker(0)

    def test_invalid_json_raises(self):
        with patch("arduino.app_peripherals.speaker.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout="not-json")
            with pytest.raises(SpeakerOpenError):
                _nth_plugged_speaker(0)
