# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""One identity for the tracks the tracker opens on the same object."""

from dataclasses import dataclass

from arduino.app_internal.ei_inference import Box
from arduino.app_utils import Logger

logger = Logger("VideoObjectTracking")

OVERLAP = 0.5  # share of the smaller box that a new track must cover of the last box of an object to continue it
SIZE_RATIO = 0.5  # smaller area over larger area: boxes further apart in size belong to different objects


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
    track represents in the result when its box covers the last box of the object by at least `overlap` of the
    smaller box and the two boxes are within `size_ratio` in area; among several such objects, the most covered
    one. Of two live tracks of one object, the one that appeared last represents it.
    """

    def __init__(self, memory_sec: float, overlap: float, size_ratio: float) -> None:
        self._memory_sec = memory_sec
        self._overlap = overlap
        self._size_ratio = size_ratio
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
        area = _area(box)
        best, best_covered = None, 0.0
        for number, known in candidates.items():
            smaller, larger = sorted((area, _area(known.box)))
            if smaller < self._size_ratio * larger or smaller <= 0:
                continue
            covered = _intersection(box, known.box) / smaller
            if covered >= self._overlap and covered > best_covered:
                best, best_covered = number, covered
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


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)
