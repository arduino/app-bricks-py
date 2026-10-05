# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.microphone.errors import MicrophoneOpenError
from arduino.app_peripherals.microphone.utils import list_audio_sources


class TestListAudioSources:
    """Discovery contract: USB/built-in partitioning ordered by PipeWire node id."""

    def test_partitions_usb_and_builtin(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        usb, builtin = list_audio_sources()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [52]

    def test_excludes_bluetooth_sources(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(54,), bluetooth_ids=(52,))

        usb, builtin = list_audio_sources()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [54]

    def test_excludes_hdmi_sources(self, mock_pw_dump):
        # The HDMI node has the lowest id: exclusion, not ordering, must keep it out.
        mock_pw_dump(builtin_ids=(54,), hdmi_ids=(50,))

        usb, builtin = list_audio_sources()

        assert usb == []
        assert [s["id"] for s in builtin] == [54]

    def test_orders_usb_by_ascending_node_id(self, mock_pw_dump):
        # Declared out of order; discovery must sort by node id (lowest first).
        mock_pw_dump(usb_ids=(60, 50), builtin_ids=())

        usb, _ = list_audio_sources()

        assert [s["id"] for s in usb] == [50, 60]

    def test_orders_builtin_by_alsa_path(self, mock_pw_dump):
        # Node ids are descending, but the ALSA path order (declaration order) must win.
        mock_pw_dump(builtin_ids=(54, 52))

        _, builtin = list_audio_sources()

        assert [s["id"] for s in builtin] == [54, 52]

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
                list_audio_sources()

    def test_invalid_json_raises(self):
        with patch("arduino.app_peripherals.microphone.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout="not-json")
            with pytest.raises(MicrophoneOpenError):
                list_audio_sources()
