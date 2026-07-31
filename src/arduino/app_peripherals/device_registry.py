# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import weakref
from collections.abc import Callable, Sequence


class DeviceRegistry:
    """
    Process-wide registry of devices claimed by peripheral instances.

    Auto-selection consults it to pick a device not already assigned to another
    instance of the same peripheral type. Explicitly addressed devices are
    claimed too, so auto-selection routes around them; those claims never block
    an explicit selection, since a device can be legitimately reused by
    multiple instances (e.g. ALSA shared mode) and contention is only
    discovered when starting the peripheral.

    Claims are keyed on stable device identities (e.g. "/dev/v4l/by-id/..."
    links or "plughw:CARD=..." refs) so they survive device reordering, and
    are counted: a device reused by multiple instances stays claimed until
    every owner has released it. A claim bound to its owner via bind() is
    released automatically when the owner is garbage collected.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._claims: dict[str, int] = {}

    def select(self, *device_groups: Callable[[], Sequence[str]]) -> str | None:
        """
        Atomically claim and return the first available device.

        Devices already claimed are skipped. Groups are enumerated lazily, in
        precedence order, only until an available device is found.

        Args:
            *device_groups: Callables returning candidate device identifiers,
                in precedence order.

        Returns:
            str | None: The claimed device identifier, or None if every device
                is already claimed or none was listed.
        """
        with self._lock:
            for group in device_groups:
                for device in group():
                    if device not in self._claims:
                        self._claims[device] = 1
                        return device
        return None

    def claim(self, device: str) -> None:
        """
        Claim a device unconditionally, even if it is already claimed.

        Args:
            device (str): Identifier of the device to claim.
        """
        with self._lock:
            self._claims[device] = self._claims.get(device, 0) + 1

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
