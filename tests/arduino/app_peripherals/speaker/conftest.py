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


def build_pw_dump(usb_ids=(), builtin_ids=()):
    """
    Build a minimal pw-dump JSON payload.

    Args:
        usb_ids: PipeWire node ids for USB Audio/Sink nodes. Each is assigned an
            ALSA card index by enumeration order, so the node-id ordering maps to the
            mocked alsaaudio cards (SomeCard=0, AnotherCard=1).
        builtin_ids: PipeWire node ids for built-in (jack) Audio/Sink nodes.
    """
    objects = []
    dev_seq = iter(range(1000, 2000))

    def add(node_id, is_usb, alsa_card=None):
        dev_id = next(dev_seq)
        if is_usb:
            dev_props = {
                "media.class": "Audio/Device",
                "device.bus": "usb",
                "device.name": f"alsa_card.usb-Device-{node_id}",
            }
            node_name = f"alsa_output.usb-Device-{node_id}.analog-stereo"
            sink_props = {
                "media.class": "Audio/Sink",
                "node.name": node_name,
                "device.id": dev_id,
                "api.alsa.pcm.card": alsa_card,
                "api.alsa.path": f"hw:{alsa_card}",
            }
        else:
            dev_props = {
                "media.class": "Audio/Device",
                "device.bus-path": "platform-sound",
                "device.form-factor": "internal",
            }
            node_name = f"alsa_output.platform-sound.Sink-{node_id}"
            sink_props = {
                "media.class": "Audio/Sink",
                "node.name": node_name,
                "device.id": dev_id,
            }
        objects.append({"id": dev_id, "info": {"props": dev_props}})
        objects.append({"id": node_id, "info": {"props": sink_props}})

    for alsa_card, node_id in enumerate(usb_ids):
        add(node_id, is_usb=True, alsa_card=alsa_card)
    for node_id in builtin_ids:
        add(node_id, is_usb=False)
    return objects


@pytest.fixture(autouse=True)
def mock_pw_dump():
    """Patch pw-dump to a configurable payload (defaults to two USB speakers)."""
    state = {"objects": build_pw_dump(usb_ids=(50, 60))}

    def fake_run(cmd, *args, **kwargs):
        result = MagicMock()
        result.stdout = json.dumps(state["objects"])
        return result

    def configure(usb_ids=(), builtin_ids=()):
        state["objects"] = build_pw_dump(usb_ids=usb_ids, builtin_ids=builtin_ids)

    with patch("arduino.app_peripherals.speaker.utils.subprocess.run", side_effect=fake_run):
        yield configure
