# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Validates the registry that releases peripherals when the application shuts down: weak
# ownership, bulk stop under a wall-clock budget, and one-shot semantics.
import gc
import threading
import time

import pytest

from arduino.app_peripherals.device_registry import DeviceRegistry
from arduino.app_utils.peripheral_registry import PeripheralRegistry


@pytest.fixture
def registry():
    """Provides a fresh peripheral registry for each test."""
    return PeripheralRegistry()


class FakePeripheral:
    """Stands in for a real peripheral: same stop() contract, controllable timing."""

    def __init__(self, block: threading.Event | None = None, raises: bool = False, started: bool = True):
        self.stop_count = 0
        self.stopped = threading.Event()
        self._block = block
        self._raises = raises
        self._started = started

    def is_started(self) -> bool:
        return self._started

    def stop(self) -> None:
        if not self._started:
            return
        self.stop_count += 1
        if self._block is not None:
            self._block.wait(timeout=30)
        if self._raises:
            raise RuntimeError("stop() blew up")
        self._started = False
        self.stopped.set()


class TestRegister:
    def test_register_keeps_only_a_weak_reference(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)

        del peripheral
        gc.collect()

        assert registry.stop_all(timeout=1.0) == []

    def test_registering_twice_stops_once(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)
        registry.register(peripheral)

        registry.stop_all(timeout=1.0)

        assert peripheral.stop_count == 1

    def test_unregister_excludes_the_peripheral(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)
        registry.unregister(peripheral)

        registry.stop_all(timeout=1.0)

        assert peripheral.stop_count == 0

    def test_peripheral_that_cannot_be_weakly_referenced_is_ignored(self, registry):
        class Unreferenceable:
            __slots__ = ()

            def stop(self) -> None:
                pass

        # Must not raise: a peripheral we cannot track is better than a failed construction
        registry.register(Unreferenceable())

        assert registry.stop_all(timeout=1.0) == []

    def test_registration_does_not_block_device_claim_release(self, registry):
        """The reason the registry holds weak references.

        DeviceRegistry releases a device claim through weakref.finalize() on the owner, so holding
        a strong reference here would keep auto-selected cameras claimed forever.
        """
        devices = DeviceRegistry()
        peripheral = FakePeripheral()

        assert devices.select(lambda: ["/dev/video0"]) == "/dev/video0"
        devices.bind("/dev/video0", peripheral)
        registry.register(peripheral)

        del peripheral
        gc.collect()

        # The claim is gone, so the same device can be selected again
        assert devices.select(lambda: ["/dev/video0"]) == "/dev/video0"


class TestStopAll:
    def test_stops_every_registered_peripheral(self, registry):
        peripherals = [FakePeripheral() for _ in range(3)]
        for peripheral in peripherals:
            registry.register(peripheral)

        assert registry.stop_all(timeout=2.0) == []
        assert all(p.stop_count == 1 for p in peripherals)
        assert all(p.stopped.is_set() for p in peripherals)

    def test_empty_registry_is_a_noop(self, registry):
        assert registry.stop_all(timeout=1.0) == []

    def test_a_failing_stop_does_not_prevent_the_others(self, registry):
        first, boom, last = FakePeripheral(), FakePeripheral(raises=True), FakePeripheral()
        for peripheral in (first, boom, last):
            registry.register(peripheral)

        assert registry.stop_all(timeout=2.0) == []
        assert first.stopped.is_set()
        assert last.stopped.is_set()

    def test_never_started_peripheral_is_a_noop(self, registry):
        peripheral = FakePeripheral(started=False)
        registry.register(peripheral)

        registry.stop_all(timeout=1.0)

        assert peripheral.stop_count == 0

    def test_stop_is_concurrent_and_bounded_by_the_budget(self, registry):
        """Three peripherals that never finish must cost one budget, not three."""
        block = threading.Event()
        peripherals = [FakePeripheral(block=block) for _ in range(3)]
        for peripheral in peripherals:
            registry.register(peripheral)

        try:
            started_at = time.monotonic()
            pending = registry.stop_all(timeout=0.3)
            elapsed = time.monotonic() - started_at

            assert len(pending) == 3
            # Serial would be >= 0.9s; the ceiling leaves room for slow CI scheduling
            assert elapsed < 0.8, f"stop_all took {elapsed:.2f}s, peripherals were not stopped concurrently"
        finally:
            block.set()

    def test_a_blocked_peripheral_does_not_hold_up_the_others(self, registry):
        block = threading.Event()
        blocked = FakePeripheral(block=block)
        responsive = FakePeripheral()
        registry.register(blocked)
        registry.register(responsive)

        try:
            pending = registry.stop_all(timeout=0.5)

            assert pending == [blocked]
            assert responsive.stopped.is_set()
        finally:
            block.set()


class TestStopAllOnce:
    def test_stops_only_on_the_first_call(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)

        registry.stop_all_once(timeout=1.0)
        registry.stop_all_once(timeout=1.0)

        assert peripheral.stop_count == 1

    def test_reset_re_arms_the_latch(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)

        registry.stop_all_once(timeout=1.0)
        peripheral._started = True  # a restarted peripheral
        registry.reset()
        registry.stop_all_once(timeout=1.0)

        assert peripheral.stop_count == 2

    def test_clear_drops_registrations_and_re_arms(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)

        registry.clear()
        registry.stop_all_once(timeout=1.0)

        assert peripheral.stop_count == 0
