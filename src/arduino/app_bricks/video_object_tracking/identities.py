# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""One identity for the tracks the tracker opens on the same object."""

from dataclasses import dataclass

from arduino.app_internal.edge_impulse.client import Box
from arduino.app_internal.edge_impulse.video_stream import iou
from arduino.app_utils import Logger

logger = Logger("VideoObjectTracking")

MIN_IOU = 0.35  # intersection over union a new track must have with the last box of an object to continue it


@dataclass
class _Object:
    label: str
    box: tuple[float, float, float, float]
    last_seen: float
    tracks: set[int]


class ObjectIdentities:
    """Gives the tracks of one object one identity: the number and the label of its first track.

    `memory_sec` is the memory the brick keeps of an object missing from the results, the same one that keeps it
    in the area and in the counts: within it a new track can continue the object, after it the object is forgotten
    with its tracks and a track of it that comes back is a new object. A new track continues an object that no
    track represents in the result when the intersection over union of its box and the last box of the object is
    at least `min_iou`; among several such objects, the one it overlaps the most. Of two live tracks of one
    object, the one that appeared last represents it.
    """

    def __init__(self, memory_sec: float, min_iou: float) -> None:
        self._memory_sec = memory_sec
        self._min_iou = min_iou
        self._objects: dict[int, _Object] = {}  # by number, the id of their first track
        self._object_of: dict[int, int] = {}  # track id -> number of its object
        self._appearance: dict[int, int] = {}  # track id -> rank of its first appearance
        self._appeared = 0

    def resolve(self, tracks: list[Box], now: float) -> list[tuple[int, str, Box]]:
        """The objects of one result, as (number, label, the track representing it), in the order of the tracks.

        Args:
            tracks (list[Box]): The tracks of the result, each with an id.
            now (float): The instant of the result, in seconds of the clock the results are stamped with.

        Returns:
            list[tuple[int, str, Box]]: One entry per object.
        """
        self._prune(now)
        represented = {self._object_of[track.id] for track in tracks if track.id in self._object_of}
        candidates = {number: known for number, known in self._objects.items() if number not in represented}
        for track in tracks:
            if track.id is None or track.id in self._object_of:
                continue
            self._appeared += 1
            self._appearance[track.id] = self._appeared
            number = self._continued(track, candidates)
            if number is None:
                self._objects[track.id] = _Object(track.label, _xyxy(track), now, {track.id})
                number = track.id
            else:
                self._objects[number].tracks.add(track.id)
                logger.debug(f"Track {track.id} ({track.label}) continues object {number} ({self._objects[number].label})")
            self._object_of[track.id] = number
        representing: dict[int, Box] = {}
        for track in tracks:
            if track.id is None:
                continue
            number = self._object_of[track.id]
            current = representing.get(number)
            if current is None or current.id is None or self._appearance[track.id] > self._appearance[current.id]:
                representing[number] = track
        for number, track in representing.items():
            self._objects[number].box, self._objects[number].last_seen = _xyxy(track), now
        return [(number, self._objects[number].label, track) for number, track in representing.items()]

    def forget(self) -> None:
        """Forget every object and track, for a tracker that numbers its tracks from zero again."""
        self._objects.clear()
        self._object_of.clear()
        self._appearance.clear()

    def _continued(self, track: Box, candidates: dict[int, _Object]) -> int | None:
        """The number of the candidate object the new track continues, None for a new object."""
        box = _xyxy(track)
        best, best_overlap = None, 0.0
        for number, known in candidates.items():
            overlap = iou(box, known.box)
            if overlap >= self._min_iou and overlap > best_overlap:
                best, best_overlap = number, overlap
        return best

    def _prune(self, now: float) -> None:
        for number, known in list(self._objects.items()):
            if now - known.last_seen > self._memory_sec:
                del self._objects[number]
                for track_id in known.tracks:
                    self._object_of.pop(track_id, None)
                    self._appearance.pop(track_id, None)


def _xyxy(track: Box) -> tuple[float, float, float, float]:
    return (track.x, track.y, track.x + track.w, track.y + track.h)
