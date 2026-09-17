# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Process-wide registry of peripherals that must be released when the application shuts down.

Peripherals (cameras, microphones, speakers, remote sensors) are not bricks, so nothing in the
brick lifecycle guarantees they are stopped. Some of them hold resources that outlive the
process: a CSI camera assigned by cam-server stays assigned to a dead client and becomes
unopenable host-wide until the service is restarted. The application must therefore release them
before the process goes away, and must do so within the container stop grace period.

The registry holds weak references only, so registering a peripheral never keeps it alive and
never interferes with the ``weakref.finalize()`` device-claim release installed by
``DeviceRegistry.bind()``.
"""

import atexit
import threading
import time
import weakref

from .logger import Logger

logger = Logger("Peripherals")

PERIPHERAL_STOP_BUDGET_S = 2.0
"""Default wall-clock budget, in seconds, for releasing every registered peripheral."""

MAX_STOP_PASSES = 3
"""Max number of times stop_all() re-reads the registry to catch late registrations."""


class PeripheralRegistry:
    """Weakly-held set of peripherals that can be stopped in bulk under a deadline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peripherals: weakref.WeakSet = weakref.WeakSet()
        self._stopped = False
        self._stopping = False

    def register(self, peripheral: object) -> None:
        """Register a peripheral to be released when the application shuts down.

        Only a weak reference is kept, so registration never keeps the peripheral alive.
        Peripherals that cannot be weakly referenced or hashed are ignored.

        Args:
            peripheral (object): Any object exposing a ``stop()`` method.
        """
        try:
            with self._lock:
                self._peripherals.add(peripheral)
        except TypeError as e:
            # Objects using __slots__ without __weakref__, or defining __eq__ without __hash__
            logger.debug(f"Peripheral '{type(peripheral).__name__}' cannot be registered for automatic release: {e}")

    def unregister(self, peripheral: object) -> None:
        """Remove a peripheral from the registry, if present.

        Args:
            peripheral (object): The peripheral to forget.
        """
        try:
            with self._lock:
                self._peripherals.discard(peripheral)
        except TypeError:
            pass

    def stop_all(self, timeout: float = PERIPHERAL_STOP_BUDGET_S) -> list[object]:
        """Stop every registered peripheral, concurrently, within a single wall-clock budget.

        Peripherals are independent devices, so they are stopped in parallel: the budget covers
        the whole set rather than each peripheral. Every ``stop()`` runs in its own daemon thread
        so that a peripheral blocked on its internal lock, for instance a camera whose
        ``capture()`` is still in flight, cannot consume the budget of the others.

        The registry is re-read after each pass, up to MAX_STOP_PASSES times: a peripheral can be
        registered while the sweep runs, for instance by a brick's own ``stop()``, and the sweep
        is the last chance it gets. ``stop()`` is idempotent for every peripheral, so a device
        that shows up in two passes is simply asked twice.

        Args:
            timeout (float): Wall-clock budget, in seconds, for the whole set.

        Returns:
            list[object]: The peripherals whose ``stop()`` had not returned within the budget.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        pending: list[object] = []
        # What has already been swept, keyed by identity rather than equality: two distinct
        # devices of a class with value semantics must still be stopped separately. The values
        # are strong references, which also keeps the ids from being recycled under us.
        swept: dict[int, object] = {}

        for _ in range(MAX_STOP_PASSES):
            with self._lock:
                # Snapshot into strong references so the peripherals cannot be collected while
                # being stopped, and so the WeakSet is never iterated while the lock is held.
                targets = [peripheral for peripheral in self._peripherals if id(peripheral) not in swept]

            if not targets:
                break

            swept.update({id(peripheral): peripheral for peripheral in targets})
            pending.extend(self._stop_batch(targets, deadline))

            if time.monotonic() >= deadline:
                break

        return pending

    def _stop_batch(self, targets: list[object], deadline: float) -> list[object]:
        """Stop one batch of peripherals in parallel, joining them until the shared deadline.

        Args:
            targets (list[object]): The peripherals to stop. Must not be empty.
            deadline (float): Absolute time.monotonic() deadline shared by the whole batch.

        Returns:
            list[object]: The peripherals whose ``stop()`` had not returned by the deadline.
        """
        logger.info(f"Releasing {len(targets)} peripheral(s)")

        workers: list[tuple[object, threading.Thread]] = []
        inline: list[object] = []
        for peripheral in targets:
            thread = threading.Thread(
                target=self._stop_one,
                args=(peripheral,),
                name=f"stop-{type(peripheral).__name__}",
                daemon=True,
            )
            try:
                thread.start()
            except (RuntimeError, MemoryError) as e:
                # No thread to be had: exhaustion, or an interpreter that is already finalizing.
                # The sweep must not abort here, or every peripheral after this one would be
                # skipped, so this one is stopped on the caller's thread instead.
                logger.warning(f"Cannot release peripheral '{type(peripheral).__name__}' in a thread: {e}")
                inline.append(peripheral)
                continue
            workers.append((peripheral, thread))

        pending = []
        for peripheral, thread in workers:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                pending.append(peripheral)
                logger.warning(f"Peripheral '{type(peripheral).__name__}' was not released within the shutdown budget")

        # Joined first, so the peripherals that did get a thread keep their share of the budget.
        # These calls are unbounded: a device left behind is worse than a shutdown that overruns.
        for peripheral in inline:
            self._stop_one(peripheral)

        return pending

    def stop_all_once(self, timeout: float = PERIPHERAL_STOP_BUDGET_S) -> list[object]:
        """Stop every registered peripheral, at most once per process.

        Later calls are no-ops, so the application shutdown and the interpreter-exit fallback can
        both call this unconditionally.

        The latch is armed only once the sweep has completed: a sweep that was interrupted, by a
        second termination signal raised into the main thread or by the interpreter running out of
        resources, leaves the latch open so the interpreter-exit fallback still gets a chance at
        the peripherals it never reached.

        Args:
            timeout (float): Wall-clock budget, in seconds, for the whole set.

        Returns:
            list[object]: The peripherals whose ``stop()`` had not returned within the budget.
        """
        with self._lock:
            if self._stopped or self._stopping:
                return []
            self._stopping = True

        # The lock is released before any blocking work: a peripheral must never be stopped while
        # a registry lock is held.
        try:
            pending = self.stop_all(timeout)
        except BaseException:
            with self._lock:
                self._stopping = False
            raise

        with self._lock:
            self._stopped = True
            self._stopping = False

        return pending

    def reset(self) -> None:
        """Re-arm the one-shot latch so a restarted application can release peripherals again."""
        with self._lock:
            self._stopped = False
            self._stopping = False

    def clear(self) -> None:
        """Drop every registration and re-arm the one-shot latch. Intended for tests."""
        with self._lock:
            self._peripherals.clear()
            self._stopped = False
            self._stopping = False

    @staticmethod
    def _stop_one(peripheral: object) -> None:
        try:
            peripheral.stop()
        except Exception as e:
            logger.warning(f"Failed to release peripheral '{type(peripheral).__name__}': {e}")


Peripherals = PeripheralRegistry()
"""Process-wide peripheral registry."""


def _stop_peripherals_at_exit() -> None:
    """Release peripherals at interpreter exit when the application shutdown never ran.

    This covers framework-managed runs, where a framework such as Streamlit owns the process
    lifecycle and ``App.run()`` returns immediately, and plain ``sys.exit()`` paths. It is a no-op
    when the application shutdown already released the peripherals.

    It cannot help on SIGKILL or ``os._exit()``, so the bounded application shutdown remains the
    primary guarantee.
    """
    Peripherals.stop_all_once()


atexit.register(_stop_peripherals_at_exit)
