# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import math
import os
import subprocess
from collections import deque
from typing import Any

import numpy as np

from .errors import MicrophoneOpenError

_MEDIA_CARRIER = "media-carrier"


def has_media_carrier() -> bool:
    """Tell whether the media carrier is currently configured on the board."""
    return os.environ.get("CONFIGURED_CARRIERS") == _MEDIA_CARRIER


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


class SpeechSegmenter:
    """
    Finds where a stretch of speech ends in a stream of PCM chunks, at a pause of the speaker.

    A segment starts at the first chunk that is not quiet, and ends at the first pause of at
    least ``pause_s`` once it is ``min_s`` long, or after ``max_s`` whatever the speaker does.
    A segment holding less than ``min_speech_s`` of speech never ends: a click or a cough is
    not a segment. Quiet chunks and pauses are told by a :class:`PauseDetector`, which adapts
    to the room noise; ``reset()`` starts the next segment and keeps the noise it learned.

    Example:
        segmenter = SpeechSegmenter(pause_s=1.0, max_s=10.0, min_speech_s=0.3)
        for chunk in mic.stream():
            if segmenter.update(chunk, mic.sample_rate, mic.channels, mic.format_is_packed):
                break  # the speaker paused
    """

    def __init__(self, pause_s: float = 0.25, min_s: float = 0.0, max_s: float = math.inf, min_speech_s: float = 0.0) -> None:
        """
        Args:
            pause_s (float): Length of the pause that ends a segment, in seconds. Default: 0.25.
            min_s (float): Shortest segment a pause can end, in seconds. Default: 0.0.
            max_s (float): Longest segment, ended even without a pause, in seconds. Default: no limit.
            min_speech_s (float): Speech a segment needs before it can end, in seconds. Default: 0.0.
        """
        self.min_s = min_s
        self.max_s = max_s
        self.min_speech_s = min_speech_s
        self.detector = PauseDetector(pause_s=pause_s)
        """The detector telling quiet chunks and pauses; feed it directly to teach it the room noise without starting a segment."""
        self.reset()

    def reset(self) -> None:
        """Start a new segment, keeping the room noise learned so far."""
        self._in_segment = False
        self._segment_s = 0.0
        self._speech_s = 0.0

    @property
    def in_segment(self) -> bool:
        """Whether a segment has started: a chunk that is not quiet arrived since the last reset."""
        return self._in_segment

    @property
    def segment_s(self) -> float:
        """Length of the current segment, in seconds."""
        return self._segment_s

    @property
    def speech_s(self) -> float:
        """Length of the chunks of the current segment that are not quiet, in seconds."""
        return self._speech_s

    def update(self, chunk: np.ndarray, sample_rate: int, channels: int = 1, is_packed: bool = False) -> bool:
        """
        Account one PCM chunk.

        Args:
            chunk (np.ndarray): PCM samples, interleaved if multichannel.
            sample_rate (int): Sample rate of the chunk, in Hz.
            channels (int): Number of interleaved channels. Default: 1.
            is_packed (bool): True for 24-bit samples packed in int32. Default: False.

        Returns:
            bool: True when the segment ends with this chunk.
        """
        return self.update_level(chunk_level(chunk, is_packed), chunk.size / (sample_rate * channels))

    def update_level(self, level: float, duration_s: float) -> bool:
        """
        Account one chunk given its level, as computed by :func:`chunk_level`.

        Args:
            level (float): RMS level of the chunk, 0..1 full scale.
            duration_s (float): Duration of the chunk, in seconds.

        Returns:
            bool: True when the segment ends with this chunk.
        """
        quiet = self.detector.update_level(level, duration_s)
        if not self._in_segment:
            if quiet:
                return False
            self._in_segment = True
        self._segment_s += duration_s
        if not quiet:
            self._speech_s += duration_s
        if self._speech_s < self.min_speech_s:
            return False
        return (self._segment_s >= self.min_s and self.detector.paused) or self._segment_s >= self.max_s
