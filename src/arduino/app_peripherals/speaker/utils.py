# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import subprocess

from .errors import SpeakerOpenError

_MEDIA_CARRIER = "media-carrier"


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


def nth_plugged_speaker(idx: int) -> str:
    """
    Find the n-th available physically connected speaker.

    The precedence is USB speakers first. Resolution falls back to jack
    speakers if no USB speaker is available at the requested position
    and the platform supports them.

    Args:
        idx (int): Index of the speaker to select (0-based).

    Returns:
        str: Identifier of the n-th available speaker, "usb:X" or "jack:X",
            where X is the 1-based ordinal index within its type.

    Raises:
        SpeakerOpenError: If no matching speaker is found.
    """
    usb_spkrs, builtin_spkrs = list_audio_sinks()

    if idx < len(usb_spkrs):
        return f"usb:{idx + 1}"

    if has_media_carrier() and idx < len(builtin_spkrs):
        return f"jack:{idx + 1}"

    raise SpeakerOpenError("No available speakers found")


def list_audio_sinks() -> tuple[list[dict], list[dict]]:
    """
    Discover audio playback devices via pw-dump, partitioned into USB and
    built-in, each ordered by ascending PipeWire node id (lowest id first).

    Returns:
        tuple[list[dict], list[dict]]: (usb_sinks, builtin_sinks)
    """
    objects = _pw_dump()

    devices = {obj["id"]: _props(obj) for obj in objects if _props(obj).get("media.class") == "Audio/Device"}

    sinks = [obj for obj in objects if _props(obj).get("media.class") == "Audio/Sink"]
    sinks.sort(key=lambda obj: obj["id"])

    usb, builtin = [], []
    for sink in sinks:
        (usb if _is_usb_sink(sink, devices) else builtin).append(sink)
    return usb, builtin


def node_description(node_name: str) -> str | None:
    """
    Return a PipeWire node's human-readable description, if available.

    Returns None when the node can't be found or pw-dump fails.

    Args:
        node_name (str): PipeWire node name ("node.name" property).

    Returns:
        str | None: The node's "node.description" (or "node.nick"), or None.
    """
    try:
        objects = _pw_dump()
    except SpeakerOpenError:
        return None
    for obj in objects:
        props = _props(obj)
        if props.get("node.name") == node_name:
            return props.get("node.description") or props.get("node.nick")
    return None


def _pw_dump() -> list:
    """Run pw-dump and parse its JSON output."""
    try:
        result = subprocess.run(
            ["pw-dump"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        raise SpeakerOpenError(f"Failed to enumerate audio devices via pw-dump: {e}")


def _is_usb_sink(sink: dict, devices: dict) -> bool:
    """Tell whether an Audio/Sink node is backed by a USB device."""
    props = _props(sink)
    parent = devices.get(props.get("device.id"), {})
    if parent.get("device.bus") == "usb":
        return True
    return False


def _props(obj: dict) -> dict:
    """Return the properties dict of a pw-dump object, or an empty dict."""
    return obj.get("info", {}).get("props", {})
