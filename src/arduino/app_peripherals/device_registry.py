# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import weakref
from collections.abc import Callable, Sequence


class DeviceRegistry:
    """
    Process-wide registry of devices claimed by auto-selected peripherals.

    Auto-selection consults it to skip devices already assigned to other
    instances of the same peripheral type, so that e.g. two Camera() calls
    resolve to two distinct cameras instead of contending for the same one.

    Claims are counted: a device handed out again once the pool is exhausted
    stays claimed until every owner has released it. A claim bound to its
    owner via bind() is released automatically when the owner is garbage
    collected.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._claims: dict[str, int] = {}

    def select(self, idx: int, *device_groups: Callable[[], Sequence[str]]) -> str | None:
        """
        Atomically claim and return the idx-th available device.

        Devices already claimed are skipped. When fewer than idx+1 devices are
        available, the idx-th plugged one is reused, so devices are shared only
        once the pool is exhausted. Groups are enumerated lazily, in precedence
        order, only until the requested index is satisfied.

        Args:
            idx (int): Index among the available (unclaimed) devices (0-based).
            *device_groups: Callables returning candidate device identifiers,
                in precedence order.

        Returns:
            str | None: The claimed device identifier, or None if no device
                exists at the requested index.
        """
        if idx < 0:
            return None
        with self._lock:
            plugged: list[str] = []
            available: list[str] = []
            for group in device_groups:
                for device in group():
                    plugged.append(device)
                    if device not in self._claims:
                        available.append(device)
                if idx < len(available):
                    break
            if idx < len(available):
                device = available[idx]
            elif idx < len(plugged):
                device = plugged[idx]
            else:
                return None
            self._claims[device] = self._claims.get(device, 0) + 1
            return device

    def bind(self, device: str, owner: object) -> None:
        """Tie a claim on a device to its owner, releasing it when the owner is garbage collected."""
        weakref.finalize(owner, self.release, device)

    def release(self, device: str) -> None:
        """Release one claim on a device, making it available again once all claims are gone."""
        with self._lock:
            count = self._claims.pop(device, 0)
            if count > 1:
                self._claims[device] = count - 1

    def clear(self) -> None:
        """Drop all claims."""
        with self._lock:
            self._claims.clear()
