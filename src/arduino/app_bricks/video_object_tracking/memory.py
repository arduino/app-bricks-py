# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""How long the brick remembers an object missing from the results."""

from collections.abc import Iterable


class ObjectMemory:
    """Remembers when each object was last seen, and forgets the ones gone for good.

    An object is forgotten once it has been missing for more than `keep_grace` results, the ones the tracker
    keeps its track for, and for more than `memory_sec` seconds, the time a track the tracker still follows may
    stay out of the results. Until then an object that comes back is the one that left.
    """

    def __init__(self, memory_sec: float) -> None:
        self._memory_sec = memory_sec
        self._results = 0
        self._last_seen: dict[int, tuple[int, float]] = {}  # object id -> (number, instant) of the last result it was in

    def update(self, seen: Iterable[int], now: float, keep_grace: float) -> list[int]:
        """Record the objects of one result and forget the ones missing for too long.

        Args:
            seen (Iterable[int]): The ids of the objects in the result.
            now (float): The instant of the result, in seconds of the clock the results are stamped with.
            keep_grace (float): For how many results the tracker keeps the track of an object it no longer detects.

        Returns:
            list[int]: The ids forgotten with this result.
        """
        self._results += 1
        for object_id in seen:
            self._last_seen[object_id] = (self._results, now)
        forgotten = [
            object_id
            for object_id, (result, instant) in self._last_seen.items()
            if self._results - result > keep_grace and now - instant > self._memory_sec
        ]
        for object_id in forgotten:
            del self._last_seen[object_id]
        return forgotten

    def last_seen(self, object_id: int) -> float | None:
        """The instant of the last result the object was in, None for an object never seen or forgotten."""
        entry = self._last_seen.get(object_id)
        return entry[1] if entry is not None else None

    def clear(self) -> list[int]:
        """Forget every object and return their ids."""
        forgotten = list(self._last_seen)
        self._last_seen.clear()
        return forgotten
