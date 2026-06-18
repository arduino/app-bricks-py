# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import subprocess

from .errors import MicrophoneOpenError

_MEDIA_CARRIER = "media-carrier"


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


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
