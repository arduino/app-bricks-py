# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path

from arduino.app_bricks.video_object_tracking.identities import OVERLAP, SIZE_RATIO, ObjectIdentities
from arduino.app_internal.edge_impulse.client import Box

RECORDED_TIE = Path(__file__).parent / "data" / "tie_640x480.json"


def _box(object_id: int, label: str = "cup", x: float = 100.0, y: float = 100.0, w: float = 60.0, h: float = 60.0) -> Box:
    return Box(label=label, score=0.9, x=x, y=y, w=w, h=h, id=object_id)


def _identities() -> ObjectIdentities:
    return ObjectIdentities(memory_sec=1.0, overlap=OVERLAP, size_ratio=SIZE_RATIO)


def _numbers(resolved: list[tuple[int, str, Box]]) -> list[tuple[int, str, int | None]]:
    return [(number, label, track.id) for number, label, track in resolved]


def test_a_new_track_over_a_lost_object_continues_it():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    identities.resolve([], 0.1)

    assert _numbers(identities.resolve([_box(2, label="vase", x=110.0)], 0.5)) == [(1, "cup", 2)]


def test_an_object_still_represented_is_not_continued():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)

    assert _numbers(identities.resolve([_box(1), _box(2, label="vase", x=105.0)], 0.1)) == [(1, "cup", 1), (2, "vase", 2)]


def test_an_object_unseen_for_more_than_the_memory_is_not_continued():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    assert _numbers(identities.resolve([_box(2)], 1.0)) == [(1, "cup", 2)], "a second is still within the memory"

    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    assert _numbers(identities.resolve([_box(2)], 1.01)) == [(2, "cup", 2)]


def test_the_same_id_back_after_the_memory_is_a_new_object():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)

    assert _numbers(identities.resolve([_box(1, label="vase")], 1.5)) == [(1, "vase", 1)], "the object was forgotten with its track"


def test_a_box_of_another_size_or_covering_less_than_half_does_not_continue():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    assert _numbers(identities.resolve([_box(2, w=40.0, h=40.0)], 0.1)) == [(2, "cup", 2)], "1600 against 3600 pixels: below half"

    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    assert _numbers(identities.resolve([_box(2, x=135.0)], 0.1)) == [(2, "cup", 2)], "25 of 60 columns covered: below half"


def test_the_most_covered_object_wins():
    identities = _identities()
    identities.resolve([_box(1, x=100.0), _box(2, x=140.0)], 0.0)
    identities.resolve([], 0.1)

    assert _numbers(identities.resolve([_box(3, x=125.0)], 0.2)) == [(2, "cup", 3)], "35 columns of the first, 45 of the second"


def test_of_two_live_tracks_the_latest_represents_the_object():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    identities.resolve([], 0.1)
    identities.resolve([_box(2, x=104.0)], 0.2)

    assert _numbers(identities.resolve([_box(1, x=101.0), _box(2, x=105.0)], 0.3)) == [(1, "cup", 2)]
    assert _numbers(identities.resolve([_box(1, x=101.0)], 0.4)) == [(1, "cup", 1)], "alone again, the first track represents it"


def test_two_new_tracks_over_a_lost_object_both_continue_it():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)
    identities.resolve([], 0.1)

    assert _numbers(identities.resolve([_box(2, label="potted plant", x=102.0), _box(3, label="vase", x=104.0)], 0.2)) == [(1, "cup", 3)]
    assert _numbers(identities.resolve([_box(2, label="potted plant", x=102.0)], 0.3)) == [(1, "cup", 2)]


def test_forget_starts_the_numbering_again():
    identities = _identities()
    identities.resolve([_box(1)], 0.0)

    identities.forget()

    assert _numbers(identities.resolve([_box(1, label="vase")], 0.1)) == [(1, "vase", 1)]


def test_recorded_tie_continues_five_tracks():
    recording = json.loads(RECORDED_TIE.read_text())
    fields = recording["track_fields"]
    identities = _identities()
    continued: list[tuple[float, int | None, int]] = []
    for frame in recording["frames"]:
        reported = [dict(zip(fields, values, strict=True)) for values in frame["tracks"]]
        tracks = [
            Box(label=t["label"], score=t["value"], x=t["x"], y=t["y"], w=t["width"], h=t["height"], id=t["object_id"])
            for t in reported
            if t["label"] != "person"
        ]
        for number, _, track in identities.resolve(tracks, frame["t"]):
            if track.id != number and track.id not in {track_id for _, track_id, _ in continued}:
                continued.append((frame["t"], track.id, number))

    assert continued == [(3.325, 39, 38), (29.122, 44, 43), (35.223, 46, 45), (58.62, 50, 49), (59.12, 51, 49)]
