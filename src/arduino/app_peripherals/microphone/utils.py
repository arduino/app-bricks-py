# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import subprocess
from collections import deque
from typing import Any

import numpy as np

from ..device_registry import DeviceRegistry
from .errors import MicrophoneOpenError

_MEDIA_CARRIER = "media-carrier"

_microphone_registry = DeviceRegistry()
"""Tracks the microphones assigned to auto-selected Microphone instances."""


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


def _claim_first_available_microphone() -> str:
    """
    Find and claim the first plugged microphone not assigned to another instance.

    USB microphones take precedence over jack ones, if supported by the
    platform. The claim is keyed on the microphone's stable reference so it
    survives device reordering, and must be released back to
    _microphone_registry, either explicitly or by binding it to its owner.

    Returns:
        str: Stable reference of the claimed microphone, either
            "plughw:CARD=<name>,DEV=<n>" or "pipewire:NODE=<node.name>".

    Raises:
        MicrophoneOpenError: If no microphone is plugged or all are already in use.
    """
    from .alsa_microphone import ALSAMicrophone

    device = _microphone_registry.select(ALSAMicrophone.list_usb_devices, ALSAMicrophone.list_jack_devices)
    if device is None:
        raise MicrophoneOpenError("No available microphones found: either none is plugged or all are already in use")
    return device


def _nth_plugged_microphone(idx: int) -> str:
    """
    Find the n-th plugged microphone, regardless of whether it is already in use.

    The index spans USB microphones first, then jack microphones, if supported
    by the current platform.

    Args:
        idx (int): Index of the microphone to select (0-based).

    Returns:
        str: Identifier of the n-th plugged microphone, "usb:X" or "jack:X",
            where X is the 1-based ordinal index within its type.

    Raises:
        MicrophoneOpenError: If no microphone is plugged at the given index.
    """
    usb_mics, builtin_mics = list_audio_sources()

    usb_count = len(usb_mics)
    if idx < usb_count:
        return f"usb:{idx + 1}"

    jack_count = len(builtin_mics) if has_media_carrier() else 0
    if idx - usb_count < jack_count:
        return f"jack:{idx - usb_count + 1}"

    raise MicrophoneOpenError(
        f"No microphone found at index {idx}: only {usb_count + jack_count} microphone(s) plugged",
        hint="Connect a microphone (or check the audio configuration) and restart the app.",
    )


def list_audio_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Discover audio capture devices via pw-dump, partitioned into USB and
    built-in. USB sources are ordered by ascending PipeWire node id (lowest
    id first); built-in ones by their ALSA path, which is stable across
    reboots.

    Sources are categorized by transport: USB, Bluetooth, HDMI or built-in.
    Bluetooth and HDMI sources are not supported yet, so they are excluded
    from the returned lists.

    Returns:
        tuple[list[dict], list[dict]]: (usb_sources, builtin_sources)
    """
    objects = _pw_dump()

    devices = {obj["id"]: obj for obj in objects if _props(obj).get("media.class") == "Audio/Device"}

    usb, builtin = [], []
    for source in (obj for obj in objects if _props(obj).get("media.class") == "Audio/Source"):
        category = _categorize_node(source, devices)
        if category == _USB:
            usb.append(source)
        elif category == _BUILTIN:
            builtin.append(source)

    usb.sort(key=lambda obj: obj["id"])  # Discovery order: hot-plugged devices append at the end
    builtin.sort(key=_alsa_path_order)  # Profile-defined order, stable across reboots
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


def _pw_dump() -> list[dict[str, Any]]:
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


_USB = "usb"
_BLUETOOTH = "bluetooth"
_HDMI = "hdmi"
_BUILTIN = "builtin"


def _categorize_node(node: dict[str, Any], devices: dict[Any, dict[str, Any]]) -> str:
    """Categorize an audio node by its transport: USB, Bluetooth, HDMI or built-in."""
    device = devices.get(_props(node).get("device.id"), {})
    device_props = _props(device)
    if device_props.get("device.bus") == "usb":
        return _USB
    if device_props.get("device.bus") == "bluetooth":
        return _BLUETOOTH
    if _routes_through_hdmi(node, device):
        return _HDMI
    return _BUILTIN


def _alsa_path_order(node: dict[str, Any]) -> tuple[str, int]:
    """Boot-stable ordering key: the node's ALSA card path with its numeric device suffix."""
    path = _props(node).get("api.alsa.path", "")
    card, sep, device = path.rpartition(",")
    if sep and device.isdigit():
        return card, int(device)
    return path, -1


def _routes_through_hdmi(node: dict[str, Any], device: dict[str, Any]) -> bool:
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


def _props(obj: dict[str, Any]) -> dict[str, Any]:
    """Return the properties dict of a pw-dump object, or an empty dict."""
    return obj.get("info", {}).get("props", {})


def chunk_level(chunk: np.ndarray, is_packed: bool = False) -> float:
    """
    RMS level of a PCM chunk on a 0..1 full-scale range, whatever its sample format.

    Args:
        chunk (np.ndarray): PCM samples as returned by ``BaseMicrophone.capture()``:
            signed or unsigned integers, or floats in -1..1. Interleaved channels are fine.
        is_packed (bool): True for 24-bit samples packed in int32
            (``BaseMicrophone.format_is_packed``). Default: False.

    Returns:
        float: 0.0 for silence, 1.0 for a full-scale square wave.
    """
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float64)
    kind = chunk.dtype.kind
    if kind == "i":
        bits = 24 if is_packed else 8 * chunk.dtype.itemsize
        x /= float(2 ** (bits - 1))
    elif kind == "u":
        half = float(2 ** (8 * chunk.dtype.itemsize - 1))
        x = (x - half) / half
    return float(np.sqrt(np.mean(x * x)))


class PauseDetector:
    """
    Tells speech pauses from speech in a stream of PCM chunks, adapting to the room noise.

    A chunk is quiet when its level is within ``quiet_over_floor_db`` of the noise
    floor, or below ``quiet_min_dbfs``. The noise floor is the ``floor_percentile``-th
    percentile of the chunk levels of the last ``floor_window_s`` seconds: speech pauses
    at least that often, so it follows the room noise, quiet or noisy, without climbing
    to the speech level. A pause is a run of quiet chunks at least ``pause_s`` long.

    Keep one detector per consumer of the audio: it holds the recent levels of the
    stream it is fed.

    Example:
        detector = PauseDetector(pause_s=0.25)
        for chunk in mic.stream():
            if detector.update(chunk, mic.sample_rate, mic.channels, mic.format_is_packed) and detector.paused:
                print("pause")
    """

    def __init__(
        self,
        pause_s: float = 0.25,
        floor_window_s: float = 10.0,
        floor_percentile: float = 5,
        quiet_over_floor_db: float = 10.0,
        quiet_min_dbfs: float = -70.0,
    ) -> None:
        """
        Args:
            pause_s (float): Minimum length of a quiet run to count as a pause, in seconds.
                Default: 0.25.
            floor_window_s (float): Length of the history the noise floor is taken from,
                in seconds. Default: 10.0.
            floor_percentile (float): Percentile of the history levels taken as noise
                floor. Default: 5.
            quiet_over_floor_db (float): A chunk up to this many dB above the floor is quiet.
                Default: 10.0.
            quiet_min_dbfs (float): A chunk below this level is always quiet (digital
                silence). Default: -70.0.
        """
        self.pause_s = pause_s
        self.floor_window_s = floor_window_s
        self.floor_percentile = floor_percentile
        self._quiet_ratio = 10 ** (quiet_over_floor_db / 20)
        self._quiet_min = 10 ** (quiet_min_dbfs / 20)
        self._levels: deque[tuple[float, float]] = deque()  # (level, seconds)
        self._levels_s = 0.0
        self._quiet_s = 0.0
        self._quiet = True

    @property
    def quiet(self) -> bool:
        """Whether the last chunk was quiet."""
        return self._quiet

    @property
    def quiet_s(self) -> float:
        """Length of the current run of quiet chunks, in seconds (0 after a chunk with speech)."""
        return self._quiet_s

    @property
    def paused(self) -> bool:
        """Whether the current quiet run is at least ``pause_s`` long."""
        return self._quiet_s >= self.pause_s

    def update(self, chunk: np.ndarray, sample_rate: int, channels: int = 1, is_packed: bool = False) -> bool:
        """
        Account one PCM chunk.

        Args:
            chunk (np.ndarray): PCM samples, interleaved if multichannel.
            sample_rate (int): Sample rate of the chunk, in Hz.
            channels (int): Number of interleaved channels. Default: 1.
            is_packed (bool): True for 24-bit samples packed in int32. Default: False.

        Returns:
            bool: True if the chunk is quiet.
        """
        return self.update_level(chunk_level(chunk, is_packed), chunk.size / (sample_rate * channels))

    def update_level(self, level: float, duration_s: float) -> bool:
        """
        Account one chunk given its level, as computed by :func:`chunk_level`.

        Args:
            level (float): RMS level of the chunk, 0..1 full scale.
            duration_s (float): Duration of the chunk, in seconds.

        Returns:
            bool: True if the chunk is quiet.
        """
        self._levels.append((level, duration_s))
        self._levels_s += duration_s
        while self._levels_s > self.floor_window_s and len(self._levels) > 1:
            self._levels_s -= self._levels.popleft()[1]
        floor = float(np.percentile([lv for lv, _ in self._levels], self.floor_percentile))
        self._quiet = level < max(floor * self._quiet_ratio, self._quiet_min)
        self._quiet_s = self._quiet_s + duration_s if self._quiet else 0.0
        return self._quiet
