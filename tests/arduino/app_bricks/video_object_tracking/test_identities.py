# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path

from arduino.app_bricks.video_object_tracking.identities import MIN_IOU, ObjectIdentities
from arduino.app_bricks.video_object_tracking.memory import ObjectMemory
from arduino.app_internal.edge_impulse.client import Box

RECORDED_TIE = Path(__file__).parent / "data" / "tie_640x480.json"


def _box(object_id: int, label: str = "cup", x: float = 100.0, y: float = 100.0, w: float = 60.0, h: float = 60.0) -> Box:
    return Box(label=label, score=0.9, x=x, y=y, w=w, h=h, id=object_id)


class _Results:
    """Feeds the identities and the memory they read one result at a time, as the brick does."""

    def __init__(self, keep_grace: int = 3) -> None:
        self.memory = ObjectMemory(memory_sec=1.0)
        self.identities = ObjectIdentities(self.memory, window_sec=1.0, min_iou=MIN_IOU)
        self._keep_grace = keep_grace

    def resolve(self, tracks: list[Box], now: float) -> list[tuple[int, str, int | None]]:
        """The objects of the result, as (number, label, id of the track representing it)."""
        resolved = self.identities.resolve(tracks, now)
        self.identities.forget_objects(self.memory.update([number for number, _, _ in resolved], now, self._keep_grace))
        return [(number, label, track.id) for number, label, track in resolved]


def test_a_new_track_over_a_lost_object_continues_it():
    results = _Results()
    results.resolve([_box(1)], 0.0)
    results.resolve([], 0.1)

    assert results.resolve([_box(2, label="vase", x=110.0)], 0.5) == [(1, "cup", 2)]


def test_an_object_still_represented_is_not_continued():
    results = _Results()
    results.resolve([_box(1)], 0.0)

    assert results.resolve([_box(1), _box(2, label="vase", x=105.0)], 0.1) == [(1, "cup", 1), (2, "vase", 2)]


def test_an_object_unseen_for_more_than_the_window_is_not_continued():
    results = _Results()
    results.resolve([_box(1)], 0.0)
    assert results.resolve([_box(2)], 1.0) == [(1, "cup", 2)], "a second is still within the window"

    results = _Results()
    results.resolve([_box(1)], 0.0)
    assert results.resolve([_box(2)], 1.01) == [(2, "cup", 2)]


def test_a_track_back_while_its_object_is_remembered_is_still_that_object():
    results = _Results(keep_grace=50)
    results.resolve([_box(1)], 0.0)
    results.resolve([], 0.1)
    results.resolve([_box(2, label="vase", x=110.0)], 0.5)
    results.resolve([], 1.0)

    assert results.resolve([_box(2, label="vase", x=110.0)], 2.0) == [(1, "cup", 2)], "1.5 s unseen, within keep_grace"


def test_the_same_id_back_after_its_object_is_forgotten_is_a_new_object():
    results = _Results(keep_grace=3)
    results.resolve([_box(1)], 0.0)
    for t in (0.5, 1.0, 1.5, 2.0):
        results.resolve([], t)

    assert results.resolve([_box(1, label="vase")], 2.1) == [(1, "vase", 1)], "the object was forgotten with its track"


def test_a_box_overlapping_too_little_does_not_continue():
    results = _Results()
    results.resolve([_box(1)], 0.0)
    assert results.resolve([_box(2, w=30.0, h=30.0)], 0.1) == [(2, "cup", 2)], "900 of 3600 pixels shared: 0.25"

    results = _Results()
    results.resolve([_box(1)], 0.0)
    assert results.resolve([_box(2, x=135.0)], 0.1) == [(2, "cup", 2)], "25 of 95 columns shared: 0.26"


def test_the_most_overlapped_object_wins():
    results = _Results()
    results.resolve([_box(1, x=100.0), _box(2, x=140.0)], 0.0)
    results.resolve([], 0.1)

    assert results.resolve([_box(3, x=125.0)], 0.2) == [(2, "cup", 3)], "0.41 with the first, 0.60 with the second"


def test_of_two_live_tracks_the_latest_represents_the_object():
    results = _Results()
    results.resolve([_box(1)], 0.0)
    results.resolve([], 0.1)
    results.resolve([_box(2, x=104.0)], 0.2)

    assert results.resolve([_box(1, x=101.0), _box(2, x=105.0)], 0.3) == [(1, "cup", 2)]
    assert results.resolve([_box(1, x=101.0)], 0.4) == [(1, "cup", 1)], "alone again, the first track represents it"


def test_two_new_tracks_over_a_lost_object_both_continue_it():
    results = _Results()
    results.resolve([_box(1)], 0.0)
    results.resolve([], 0.1)

    assert results.resolve([_box(2, label="potted plant", x=102.0), _box(3, label="vase", x=104.0)], 0.2) == [(1, "cup", 3)]
    assert results.resolve([_box(2, label="potted plant", x=102.0)], 0.3) == [(1, "cup", 2)]


def test_forgetting_every_object_starts_the_numbering_again():
    results = _Results()
    results.resolve([_box(1)], 0.0)

    results.identities.forget_objects(results.memory.clear())

    assert results.resolve([_box(1, label="vase")], 0.1) == [(1, "vase", 1)]


def test_recorded_tie_continues_five_tracks():
    recording = json.loads(RECORDED_TIE.read_text())
    fields = recording["track_fields"]
    results = _Results()
    continued: list[tuple[float, int | None, int]] = []
    for frame in recording["frames"]:
        reported = [dict(zip(fields, values, strict=True)) for values in frame["tracks"]]
        tracks = [
            Box(label=t["label"], score=t["value"], x=t["x"], y=t["y"], w=t["width"], h=t["height"], id=t["object_id"])
            for t in reported
            if t["label"] != "person"
        ]
        for number, _, track_id in results.resolve(tracks, frame["t"]):
            if track_id != number and track_id not in {continued_id for _, continued_id, _ in continued}:
                continued.append((frame["t"], track_id, number))

    assert continued == [(3.325, 39, 38), (29.122, 44, 43), (35.223, 46, 45), (58.62, 50, 49), (59.12, 51, 49)]
