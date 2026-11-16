# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.video_object_tracking.memory import ObjectMemory


def test_an_object_missing_for_many_results_in_less_than_the_memory_seconds_is_kept():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7], 0.0, keep_grace=3)

    for i in range(1, 10):
        assert memory.update([], i * 0.1, keep_grace=3) == [], "9 results missing, 0.9 s"

    assert memory.last_seen(7) == 0.0


def test_an_object_missing_for_long_in_no_more_than_keep_grace_results_is_kept():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7], 0.0, keep_grace=3)

    for t in (2.0, 4.0, 6.0):
        assert memory.update([], t, keep_grace=3) == [], "3 results missing, the tracker may still follow it"


def test_an_object_missing_for_more_than_keep_grace_results_and_the_memory_seconds_is_forgotten_once():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7, 8], 0.0, keep_grace=3)
    for t in (0.4, 0.8, 1.2):
        assert memory.update([8], t, keep_grace=3) == []

    assert memory.update([8], 1.6, keep_grace=3) == [7], "4 results missing, 1.6 s"
    assert memory.last_seen(7) is None and memory.last_seen(8) == 1.6
    assert memory.update([8], 2.0, keep_grace=3) == []


def test_an_object_seen_again_starts_missing_anew():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7], 0.0, keep_grace=1)
    memory.update([], 0.9, keep_grace=1)
    memory.update([7], 1.8, keep_grace=1)

    assert memory.update([], 2.7, keep_grace=1) == [], "1 result missing since it came back"
    assert memory.update([], 3.6, keep_grace=1) == [7]


def test_keep_grace_is_the_one_of_each_result():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7], 0.0, keep_grace=50)
    for i in range(1, 30):
        assert memory.update([], i * 0.1, keep_grace=50) == [], "29 results missing, the tracker keeps the track for 50"

    assert memory.update([], 3.0, keep_grace=3) == [7], "lowered: 30 results missing are more than 3"


def test_clear_forgets_every_object():
    memory = ObjectMemory(memory_sec=1.0)
    memory.update([7, 8], 0.0, keep_grace=3)

    assert memory.clear() == [7, 8]
    assert memory.last_seen(7) is None and memory.last_seen(8) is None
