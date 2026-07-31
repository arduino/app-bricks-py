# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""
Shared fixtures for speaker tests.

`mock_pw_dump` (autouse) patches the `pw-dump` invocation used by the device
discovery. It defaults to two USB Audio/Sink nodes and yields a setter
so a test can declare its own USB/built-in topology.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.speaker.utils import _speaker_registry


@pytest.fixture(autouse=True)
def clean_speaker_registry():
    """Give each test a clean slate of auto-selected speaker claims."""
    _speaker_registry.clear()
    yield


def build_pw_dump(usb_ids=(), builtin_ids=(), bluetooth_ids=(), hdmi_ids=()):
    """
    Build a minimal pw-dump JSON payload.

    Args:
        usb_ids: PipeWire node ids for USB Audio/Sink nodes. Each is assigned an
            ALSA card index by enumeration order, so the node-id ordering maps to the
            mocked alsaaudio cards (SomeCard=0, AnotherCard=1).
        builtin_ids: PipeWire node ids for built-in (jack) Audio/Sink nodes.
        bluetooth_ids: PipeWire node ids for Bluetooth Audio/Sink nodes.
        hdmi_ids: PipeWire node ids for built-in Audio/Sink nodes routed through HDMI.
    """
    objects = []
    dev_seq = iter(range(1000, 2000))

    def add(node_id, dev_props, sink_props, dev_params=None):
        dev_id = next(dev_seq)
        device = {"id": dev_id, "info": {"props": {"media.class": "Audio/Device", **dev_props}}}
        if dev_params:
            device["info"]["params"] = dev_params
        objects.append(device)
        objects.append({"id": node_id, "info": {"props": {"media.class": "Audio/Sink", "device.id": dev_id, **sink_props}}})

    for alsa_card, node_id in enumerate(usb_ids):
        add(
            node_id,
            {"device.bus": "usb", "device.name": f"alsa_card.usb-Device-{node_id}"},
            {
                "node.name": f"alsa_output.usb-Device-{node_id}.analog-stereo",
                "api.alsa.pcm.card": alsa_card,
                "api.alsa.path": f"hw:{alsa_card}",
            },
        )
    for alsa_device, node_id in enumerate(builtin_ids):
        add(
            node_id,
            {"device.bus-path": "platform-sound", "device.form-factor": "internal"},
            {
                "node.name": f"alsa_output.platform-sound.Sink-{node_id}",
                "node.description": f"Built-in Audio {node_id}",
                "api.alsa.path": f"hw:platformsound,{alsa_device}",
            },
        )
    for node_id in bluetooth_ids:
        add(
            node_id,
            {"device.bus": "bluetooth", "device.api": "bluez5"},
            {"node.name": f"bluez_output.device-{node_id}.1"},
        )
    for node_id in hdmi_ids:
        add(
            node_id,
            {"device.bus-path": "platform-sound", "device.form-factor": "internal"},
            {
                "node.name": f"alsa_output.platform-sound.HDMI-{node_id}",
                "card.profile.device": 0,
            },
            dev_params={"EnumRoute": [{"index": 0, "direction": "Output", "info": [1, "port.type", "hdmi"], "devices": [0]}]},
        )
    return objects


@pytest.fixture(autouse=True)
def mock_pw_dump():
    """Patch pw-dump to a configurable payload (defaults to two USB speakers)."""
    state = {"objects": build_pw_dump(usb_ids=(50, 60))}

    def fake_run(cmd, *args, **kwargs):
        result = MagicMock()
        result.stdout = json.dumps(state["objects"])
        return result

    def configure(**kwargs):
        state["objects"] = build_pw_dump(**kwargs)

    with patch("arduino.app_peripherals.speaker.utils.subprocess.run", side_effect=fake_run):
        yield configure
