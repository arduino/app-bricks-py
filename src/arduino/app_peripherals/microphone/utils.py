# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import subprocess

from ..device_registry import DeviceRegistry
from .errors import MicrophoneOpenError

_MEDIA_CARRIER = "media-carrier"

microphone_registry = DeviceRegistry()
"""Tracks the microphones assigned to auto-selected Microphone instances."""


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


def claim_nth_available_microphone(idx: int) -> str:
    """
    Find and claim the n-th available physically connected microphone.

    The precedence is USB microphones first, then jack microphones if
    supported by the platform. Microphones already claimed by other
    auto-selected instances are skipped; when every plugged microphone is
    claimed, the n-th plugged one is reused. The claim must be released back
    to microphone_registry, either explicitly or by binding it to its owner.

    Args:
        idx (int): Index of the microphone to select among the available ones (0-based).

    Returns:
        str: Identifier of the n-th available microphone, "usb:X" or "jack:X",
            where X is the 1-based ordinal index within its type.

    Raises:
        MicrophoneOpenError: If no matching microphone is found.
    """
    usb_mics, builtin_mics = list_audio_sources()

    candidates = [f"usb:{i + 1}" for i in range(len(usb_mics))]
    if has_media_carrier():
        candidates += [f"jack:{i + 1}" for i in range(len(builtin_mics))]

    device = microphone_registry.select(idx, lambda: candidates)
    if device is None:
        raise MicrophoneOpenError("No available microphones found")
    return device


def nth_plugged_microphone(idx: int) -> str:
    """
    Find the n-th available physically connected microphone.

    The precedence is USB microphones first. Resolution falls back to jack
    microphones if no USB microphone is available at the requested position
    and the platform supports them.

    Args:
        idx (int): Index of the microphone to select (0-based).

    Returns:
        str: Identifier of the n-th available microphone, "usb:X" or "jack:X",
            where X is the 1-based ordinal index within its type.

    Raises:
        MicrophoneOpenError: If no matching microphone is found.
    """
    usb_mics, builtin_mics = list_audio_sources()

    if idx < len(usb_mics):
        return f"usb:{idx + 1}"

    if has_media_carrier() and idx < len(builtin_mics):
        return f"jack:{idx + 1}"

    raise MicrophoneOpenError("No available microphones found")


def list_audio_sources() -> tuple[list[dict], list[dict]]:
    """
    Discover audio capture devices via pw-dump, partitioned into USB and
    built-in, each ordered by ascending PipeWire node id (lowest id first).

    Returns:
        tuple[list[dict], list[dict]]: (usb_sources, builtin_sources)
    """
    objects = _pw_dump()

    devices = {obj["id"]: _props(obj) for obj in objects if _props(obj).get("media.class") == "Audio/Device"}

    sources = [obj for obj in objects if _props(obj).get("media.class") == "Audio/Source"]
    sources.sort(key=lambda obj: obj["id"])

    usb, builtin = [], []
    for source in sources:
        (usb if _is_usb_source(source, devices) else builtin).append(source)
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
    except MicrophoneOpenError:
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
        raise MicrophoneOpenError(f"Failed to enumerate audio devices via pw-dump: {e}")


def _is_usb_source(source: dict, devices: dict) -> bool:
    """Tell whether an Audio/Source node is backed by a USB device."""
    props = _props(source)
    parent = devices.get(props.get("device.id"), {})
    if parent.get("device.bus") == "usb":
        return True
    return False


def _props(obj: dict) -> dict:
    """Return the properties dict of a pw-dump object, or an empty dict."""
    return obj.get("info", {}).get("props", {})
