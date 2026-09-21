# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Validates the registry that releases peripherals when the application shuts down: weak
# ownership, bulk stop under a wall-clock budget, and one-shot semantics.
import gc
import threading
import time

import pytest

import arduino.app_utils.peripheral_registry as peripheral_registry_module
from arduino.app_peripherals.device_registry import DeviceRegistry
from arduino.app_utils.peripheral_registry import PeripheralRegistry


def _failing_start(when: int | None):
    """Returns a Thread.start that fails on the when-th stop thread, or on all of them.

    Mimics an interpreter that cannot create threads any more: thread exhaustion under memory
    pressure, or a finalizing interpreter. Only the registry's own stop threads are affected, so
    pytest's internals keep working.
    """
    real_start = threading.Thread.start
    calls = {"n": 0}

    def start(self):
        if self.name.startswith("stop-"):
            calls["n"] += 1
            if when is None or calls["n"] == when:
                raise RuntimeError("can't start new thread")
        return real_start(self)

    return start


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


class TestStopAllRobustness:
    """The sweep is the last chance a peripheral gets: it must never abort halfway."""

    def test_a_peripheral_that_cannot_get_a_thread_is_stopped_inline(self, registry, monkeypatch):
        """Condition: Thread.start() fails for a peripheral, e.g. no thread can be created.
        Expectation: it is stopped on the caller's thread instead of being skipped.
        """
        peripheral = FakePeripheral()
        registry.register(peripheral)
        monkeypatch.setattr(threading.Thread, "start", _failing_start(when=1))

        assert registry.stop_all(timeout=1.0) == []
        assert peripheral.stop_count == 1

    def test_a_failed_thread_start_does_not_skip_the_others(self, registry, monkeypatch):
        """Condition: the first Thread.start() of the sweep fails.
        Expectation: the sweep does not propagate and every peripheral is still stopped. It used
        to abort on the first failure, leaving every peripheral after it held.
        """
        peripherals = [FakePeripheral() for _ in range(3)]
        for peripheral in peripherals:
            registry.register(peripheral)
        monkeypatch.setattr(threading.Thread, "start", _failing_start(when=1))

        registry.stop_all(timeout=1.0)

        assert all(p.stop_count == 1 for p in peripherals), [p.stop_count for p in peripherals]

    def test_an_exhausted_interpreter_still_releases_every_peripheral(self, registry, monkeypatch):
        """Condition: no thread can be created at all, as in an interpreter that is finalizing.
        Expectation: every peripheral is released, serially, on the caller's thread.
        """
        peripherals = [FakePeripheral() for _ in range(3)]
        for peripheral in peripherals:
            registry.register(peripheral)
        monkeypatch.setattr(threading.Thread, "start", _failing_start(when=None))

        assert registry.stop_all(timeout=1.0) == []
        assert all(p.stopped.is_set() for p in peripherals)


class TestLateRegistrations:
    """A peripheral can be created while the sweep runs, e.g. by a brick's own stop()."""

    def test_a_peripheral_registered_during_the_sweep_is_released(self, registry):
        late = FakePeripheral()

        class Registering(FakePeripheral):
            def stop(self) -> None:
                super().stop()
                registry.register(late)

        first = Registering()
        registry.register(first)

        registry.stop_all(timeout=2.0)

        assert first.stop_count == 1
        assert late.stop_count == 1, "a peripheral registered during the sweep was never released"

    def test_late_registrations_are_bounded_by_max_passes(self, registry, monkeypatch):
        """A peripheral whose stop() keeps registering new ones must not loop forever."""
        monkeypatch.setattr(peripheral_registry_module, "MAX_STOP_PASSES", 3)
        created: list[FakePeripheral] = []

        class Breeding(FakePeripheral):
            def stop(self) -> None:
                super().stop()
                child = Breeding()
                created.append(child)
                registry.register(child)

        root = Breeding()
        registry.register(root)

        registry.stop_all(timeout=2.0)

        assert root.stop_count == 1

        # One new peripheral per pass, so the sweep stops after MAX_STOP_PASSES instead of spinning
        assert len(created) == 3

    def test_a_late_registration_does_not_extend_the_budget(self, registry):
        """The extra passes share the original deadline, they do not restart it."""
        block = threading.Event()
        late = FakePeripheral(block=block)

        class Registering(FakePeripheral):
            def stop(self) -> None:
                super().stop()
                registry.register(late)

        registering = Registering()
        registry.register(registering)

        try:
            started_at = time.monotonic()
            registry.stop_all(timeout=0.3)
            elapsed = time.monotonic() - started_at

            assert elapsed < 0.8, f"stop_all took {elapsed:.2f}s, the late pass restarted the budget"
            assert registering.stop_count == 1
        finally:
            block.set()

    def test_an_exhausted_budget_stops_re_reading_the_registry(self, registry):
        """Condition: the first pass already used the whole budget.
        Expectation: no further pass is attempted, the shutdown has to return to its caller.
        """
        block = threading.Event()
        late = FakePeripheral()

        class Registering(FakePeripheral):
            def stop(self) -> None:
                super().stop()
                registry.register(late)
                block.wait(timeout=30)

        registering = Registering()
        registry.register(registering)

        try:
            pending = registry.stop_all(timeout=0.2)

            assert len(pending) == 1
            assert late.stop_count == 0, "the budget was already spent, the late pass must be skipped"
        finally:
            block.set()


class TestStopAllOnceRobustness:
    def test_an_interrupted_sweep_leaves_the_latch_open(self, registry, monkeypatch):
        """Condition: the sweep is interrupted, e.g. by a second termination signal raised into
        the main thread while the shutdown is running.
        Expectation: the latch stays open so the interpreter-exit fallback retries. The latch used
        to be armed before the work, so an interrupted shutdown lost its peripherals for good.
        """
        peripheral = FakePeripheral()
        registry.register(peripheral)

        def interrupted(timeout):
            raise KeyboardInterrupt("signal during shutdown")

        monkeypatch.setattr(registry, "stop_all", interrupted)
        with pytest.raises(KeyboardInterrupt):
            registry.stop_all_once(timeout=0.2)

        monkeypatch.undo()
        registry.stop_all_once(timeout=1.0)  # the interpreter-exit fallback

        assert peripheral.stop_count == 1

    def test_a_completed_sweep_still_arms_the_latch(self, registry):
        peripheral = FakePeripheral()
        registry.register(peripheral)

        registry.stop_all_once(timeout=1.0)
        peripheral._started = True  # would be stopped again if the latch were still open
        registry.stop_all_once(timeout=1.0)

        assert peripheral.stop_count == 1

    def test_a_concurrent_call_does_not_start_a_second_sweep(self, registry):
        """The app shutdown and the interpreter-exit fallback can overlap: only one may sweep."""
        entered = threading.Event()
        block = threading.Event()
        peripheral = FakePeripheral()

        class Slow(FakePeripheral):
            def stop(self) -> None:
                entered.set()
                block.wait(timeout=30)
                super().stop()

        slow = Slow()
        registry.register(slow)
        registry.register(peripheral)

        sweeper = threading.Thread(target=registry.stop_all_once, args=(2.0,), daemon=True)
        sweeper.start()
        try:
            assert entered.wait(timeout=2.0)

            assert registry.stop_all_once(timeout=1.0) == [], "a second sweep was started"
        finally:
            block.set()
            sweeper.join(timeout=5)

        assert peripheral.stop_count == 1
