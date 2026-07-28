# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import subprocess

from ..device_registry import DeviceRegistry
from .errors import SpeakerOpenError

_MEDIA_CARRIER = "media-carrier"

_speaker_registry = DeviceRegistry()
"""Tracks the speakers assigned to auto-selected Speaker instances."""


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


def _claim_first_available_speaker() -> str:
    """
    Find and claim the first plugged speaker not assigned to another instance.

    USB speakers take precedence over jack ones, if supported by the platform.
    The claim is keyed on the speaker's stable reference so it survives device
    reordering, and must be released back to _speaker_registry, either
    explicitly or by binding it to its owner.

    Returns:
        str: Stable reference of the claimed speaker, either
            "plughw:CARD=<name>,DEV=<n>" or "pipewire:NODE=<node.name>".

    Raises:
        SpeakerOpenError: If no speaker is plugged or all are already in use.
    """
    from .alsa_speaker import ALSASpeaker

    device = _speaker_registry.select(ALSASpeaker.list_usb_devices, ALSASpeaker.list_jack_devices)
    if device is None:
        raise SpeakerOpenError("No available speakers found: either none is plugged or all are already in use")
    return device


def _nth_plugged_speaker(idx: int) -> str:
    """
    Find the n-th plugged speaker, regardless of whether it is already in use.

    The index spans USB speakers first, then jack speakers, if supported
    by the current platform.

    Args:
        idx (int): Index of the speaker to select (0-based).

    Returns:
        str: Identifier of the n-th plugged speaker, "usb:X" or "jack:X",
            where X is the 1-based ordinal index within its type.

    Raises:
        SpeakerOpenError: If no speaker is plugged at the given index.
    """
    usb_spkrs, builtin_spkrs = list_audio_sinks()

    usb_count = len(usb_spkrs)
    if idx < usb_count:
        return f"usb:{idx + 1}"

    jack_count = len(builtin_spkrs) if has_media_carrier() else 0
    if idx - usb_count < jack_count:
        return f"jack:{idx - usb_count + 1}"

    raise SpeakerOpenError(f"No speaker found at index {idx}: only {usb_count + jack_count} speaker(s) plugged")


def list_audio_sinks() -> tuple[list[dict], list[dict]]:
    """
    Discover audio playback devices via pw-dump, partitioned into USB and
    built-in. USB sinks are ordered by ascending PipeWire node id (lowest
    id first); built-in ones by their ALSA path, which is stable across
    reboots.

    Sinks are categorized by transport: USB, Bluetooth, HDMI or built-in.
    Bluetooth and HDMI sinks are not supported yet, so they are excluded
    from the returned lists.

    Returns:
        tuple[list[dict], list[dict]]: (usb_sinks, builtin_sinks)
    """
    objects = _pw_dump()

    devices = {obj["id"]: obj for obj in objects if _props(obj).get("media.class") == "Audio/Device"}

    sinks = [obj for obj in objects if _props(obj).get("media.class") == "Audio/Sink"]
    sinks.sort(key=lambda obj: obj["id"])

    usb, builtin = [], []
    for sink in sinks:
        category = _categorize_node(sink, devices)
        if category == _USB:
            usb.append(sink)
        elif category == _BUILTIN:
            builtin.append(sink)
    builtin.sort(key=_alsa_path_order)
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


_USB = "usb"
_BLUETOOTH = "bluetooth"
_HDMI = "hdmi"
_BUILTIN = "builtin"


def _categorize_node(node: dict, devices: dict) -> str:
    """Categorize an audio node by its transport: USB, Bluetooth, HDMI or built-in."""
    device = devices.get(_props(node).get("device.id"), {})
    device_props = _props(device)
    if device_props.get("device.bus") == "usb":
        return _USB
    if device_props.get("device.bus") == "bluetooth" or device_props.get("device.api") == "bluez5":
        return _BLUETOOTH
    if _routes_through_hdmi(node, device):
        return _HDMI
    return _BUILTIN


def _alsa_path_order(node: dict) -> tuple[str, int]:
    """Boot-stable ordering key: the node's ALSA card path with its numeric device suffix."""
    path = _props(node).get("api.alsa.path", "")
    card, sep, device = path.rpartition(",")
    if sep and device.isdigit():
        return card, int(device)
    return path, -1


def _routes_through_hdmi(node: dict, device: dict) -> bool:
    """Tell whether an audio node is routed through an HDMI port of its device."""
    profile_device = _props(node).get("card.profile.device")
    if profile_device is None:
        return False
    for route in device.get("info", {}).get("params", {}).get("EnumRoute", []):
        if profile_device in route.get("devices", []):
            # Route info is a flat [count, key, value, ...] list
            info = route.get("info") or []
            route_props = dict(zip(info[1::2], info[2::2]))
            if route_props.get("port.type") == "hdmi":
                return True
    return False


def _props(obj: dict) -> dict:
    """Return the properties dict of a pw-dump object, or an empty dict."""
    return obj.get("info", {}).get("props", {})
