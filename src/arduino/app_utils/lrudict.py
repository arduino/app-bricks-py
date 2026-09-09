# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from collections import OrderedDict
from typing import Any


class LRUDict[K, V](OrderedDict[K, V]):
    """A dictionary-like object with a fixed size that evicts the least recently used items."""

    def __init__(self, maxsize: int = 128, *args: Any, **kwargs: Any) -> None:
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: K) -> V:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            self.move_to_end(key)

        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            # Evict the least recently used item (the first item)
            self.popitem(last=False)
