# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# CASE 6: time-boxed shutdown
# Validates that every brick is stopped within one global budget, that the peripherals are
# released only afterwards, and that neither a hanging brick nor a hanging worker thread can
# prevent the release. The app runs in a container with a stop grace period: overrunning it gets
# the process killed, which leaves exclusive peripherals (e.g. a CSI camera) held by a dead
# process and unusable until their service is restarted.
import subprocess
import sys
import threading
import time

import pytest

import arduino.app_utils.app as app
from arduino.app_utils import AppController, brick, peripheral_registry
from arduino.app_utils.peripheral_registry import PeripheralRegistry


BRICKS_BUDGET = 0.6
PERIPHERALS_BUDGET = 0.2
SLACK = 0.6

# Captured before the fast_budget fixture can shrink them, so the tests that are about the
# production timings can put them back and assert on the real numbers.
REAL_BRICKS_BUDGET = app.SHUTDOWN_BRICKS_BUDGET_S
REAL_PERIPHERALS_BUDGET = app.SHUTDOWN_PERIPHERALS_BUDGET_S
REAL_LOCK_BUDGET = app.SHUTDOWN_LOCK_BUDGET_S


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


@pytest.fixture
def peripherals(monkeypatch):
    """Provides a fresh peripheral registry, reachable by the AppController under test."""
    registry = PeripheralRegistry()
    monkeypatch.setattr(peripheral_registry, "Peripherals", registry)
    return registry


@pytest.fixture(autouse=True)
def fast_budget(monkeypatch):
    """Shrinks the shutdown budgets so the tests stay quick, keeping their proportions."""
    monkeypatch.setattr(app, "SHUTDOWN_BRICKS_BUDGET_S", BRICKS_BUDGET)
    monkeypatch.setattr(app, "SHUTDOWN_PERIPHERALS_BUDGET_S", PERIPHERALS_BUDGET)
    monkeypatch.setattr(app, "SHUTDOWN_LOCK_BUDGET_S", 0.1)


@pytest.fixture
def real_budgets(monkeypatch):
    """Puts the production budgets back, overriding the autouse fast_budget fixture.

    Used by the tests that exist to check the shutdown fits in the grace period, which is a
    statement about the real numbers and says nothing if the budgets are shrunk first.
    """
    monkeypatch.setattr(app, "SHUTDOWN_BRICKS_BUDGET_S", REAL_BRICKS_BUDGET)
    monkeypatch.setattr(app, "SHUTDOWN_PERIPHERALS_BUDGET_S", REAL_PERIPHERALS_BUDGET)
    monkeypatch.setattr(app, "SHUTDOWN_LOCK_BUDGET_S", REAL_LOCK_BUDGET)


# Test class definitions


class RecordingPeripheral:
    """Stands in for a peripheral, recording when it was released."""

    def __init__(self, timeline: list[str]):
        self._timeline = timeline
        self.stop_count = 0
        self.stopped = threading.Event()

    def stop(self) -> None:
        self.stop_count += 1
        self._timeline.append("peripheral")
        self.stopped.set()


@brick
class RecordingBrick:
    """A brick that records the order in which it was stopped."""

    def __init__(self, name: str, timeline: list[str]):
        self.name = name
        self._timeline = timeline
        self.stop_called = threading.Event()

    def stop(self) -> None:
        self._timeline.append(self.name)
        self.stop_called.set()


@brick
class StuckLoopBrick:
    """A brick whose worker thread cannot be interrupted: stop() returns, the loop does not."""

    def __init__(self, name: str = ""):
        self.name = name
        self.stop_called = threading.Event()

    def stop(self) -> None:
        self.stop_called.set()

    def loop(self) -> None:
        time.sleep(30)


@brick
class HangingStopBrick:
    """A brick whose stop() never returns, the case that used to block the whole shutdown."""

    def __init__(self, release: threading.Event):
        self._release = release
        self.stop_entered = threading.Event()

    def stop(self) -> None:
        self.stop_entered.set()
        self._release.wait(timeout=30)


class BlockingPeripheral:
    """A peripheral whose stop() never returns, e.g. a camera stuck in its driver."""

    def __init__(self, release: threading.Event):
        self._release = release
        self.stop_entered = threading.Event()

    def stop(self) -> None:
        self.stop_entered.set()
        self._release.wait(timeout=30)


class Terminated(BaseException):
    """Stands in for the signal exception loop() raises into the main thread."""


def _raiser(exc: BaseException):
    """Returns a user loop that raises exc on its first iteration."""

    def user_loop() -> None:
        raise exc

    return user_loop


@brick
class ManyWorkersBrick:
    """A brick with several uninterruptible worker threads, all sharing one brick budget."""

    def __init__(self) -> None:
        self.stop_called = threading.Event()

    def stop(self) -> None:
        self.stop_called.set()

    def loop(self) -> None:
        time.sleep(30)

    @brick.loop
    def second_loop(self) -> None:
        time.sleep(30)

    def execute(self) -> None:
        time.sleep(30)


def _run(app_instance) -> threading.Thread:
    """Starts the app in the background and waits for its bricks to be running."""
    thread = threading.Thread(target=app_instance.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not app_instance._running:
        time.sleep(0.01)
    time.sleep(0.05)  # let the worker threads reach their loops

    return thread


def _run_until_it_returns(app_instance, user_loop=None) -> BaseException | None:
    """Runs the whole App.run() on a worker thread and returns whatever escaped it.

    run() is kept off the main thread on purpose: there it would install a process-wide SIGTERM
    handler, which has no business outliving a test.
    """
    escaped: list[BaseException] = []

    def target() -> None:
        try:
            app_instance.run(user_loop)
        except BaseException as e:
            # Catching BaseException is the point: what escapes run() is what the test asserts on
            escaped.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), "App.run() never returned"
    return escaped[0] if escaped else None


# Test cases


def test_case_6_peripherals_are_released_after_every_brick(app_instance, peripherals):
    """Condition: several bricks and a peripheral are registered.
    Expectation: bricks are stopped in reverse start order, and the peripheral is released last.
    """
    timeline: list[str] = []
    first, second, third = (RecordingBrick(f"brick-{n}", timeline) for n in (1, 2, 3))
    peripheral = RecordingPeripheral(timeline)
    peripherals.register(peripheral)

    _run(app_instance)
    app_instance._shutdown()

    assert timeline == ["brick-3", "brick-2", "brick-1", "peripheral"]
    # State the ordering guarantee independently of how many bricks there are
    assert timeline.index("peripheral") == len(timeline) - 1
    assert first.stop_called.is_set()
    assert second.stop_called.is_set()
    assert third.stop_called.is_set()
    assert peripheral.stop_count == 1


def test_case_6_uninterruptible_bricks_share_one_global_budget(app_instance, peripherals):
    """Condition: four bricks whose worker threads never return.
    Expectation: the shutdown costs one budget in total, not one per brick, and every brick is
    still asked to stop.
    """
    bricks = [StuckLoopBrick(f"brick-{n}") for n in range(4)]
    peripheral = RecordingPeripheral([])
    peripherals.register(peripheral)

    _run(app_instance)

    started_at = time.monotonic()
    app_instance._shutdown()
    elapsed = time.monotonic() - started_at

    # Per-brick joins would cost 4x the join timeout; the budget is shared instead
    assert elapsed < BRICKS_BUDGET + PERIPHERALS_BUDGET + SLACK, f"shutdown took {elapsed:.2f}s"
    assert all(b.stop_called.is_set() for b in bricks), "every brick must still be asked to stop"
    assert peripheral.stopped.is_set(), "the peripheral must still be released"


def test_case_6_a_hanging_brick_stop_does_not_starve_the_peripherals(app_instance, peripherals):
    """Condition: a brick whose stop() never returns.
    Expectation: the shutdown gives up on it and still releases the peripherals. Before the
    budget existed, this shutdown never returned at all.
    """
    release = threading.Event()
    hanging = HangingStopBrick(release)
    peripheral = RecordingPeripheral([])
    peripherals.register(peripheral)

    _run(app_instance)

    try:
        started_at = time.monotonic()
        app_instance._shutdown()
        elapsed = time.monotonic() - started_at

        assert hanging.stop_entered.is_set()
        assert elapsed < BRICKS_BUDGET + PERIPHERALS_BUDGET + SLACK, f"shutdown took {elapsed:.2f}s"
        assert peripheral.stopped.is_set(), "a hanging brick must not prevent the release"
    finally:
        release.set()


def test_case_6_several_worker_threads_share_the_brick_budget(app_instance, peripherals):
    """Condition: one brick with three uninterruptible worker threads.
    Expectation: they share the brick's slice of the budget instead of each costing a full join.
    """
    ManyWorkersBrick()
    peripheral = RecordingPeripheral([])
    peripherals.register(peripheral)

    _run(app_instance)

    started_at = time.monotonic()
    app_instance._shutdown()
    elapsed = time.monotonic() - started_at

    assert elapsed < BRICKS_BUDGET + PERIPHERALS_BUDGET + SLACK, f"shutdown took {elapsed:.2f}s"
    assert peripheral.stopped.is_set()


def test_case_6_responsive_bricks_still_stop_promptly(app_instance, peripherals):
    """Condition: bricks that stop immediately.
    Expectation: the shutdown does not wait out a budget it did not need.
    """
    timeline: list[str] = []
    for n in range(3):
        RecordingBrick(f"brick-{n}", timeline)
    peripheral = RecordingPeripheral(timeline)
    peripherals.register(peripheral)

    _run(app_instance)

    started_at = time.monotonic()
    app_instance._shutdown()
    elapsed = time.monotonic() - started_at

    assert elapsed < BRICKS_BUDGET, f"shutdown burned {elapsed:.2f}s for bricks that stop at once"
    assert peripheral.stopped.is_set()


def test_case_6_shutdown_is_idempotent(app_instance, peripherals):
    """Condition: _shutdown() is called twice.
    Expectation: bricks and peripherals are stopped exactly once.
    """
    timeline: list[str] = []
    recording = RecordingBrick("brick-1", timeline)
    peripheral = RecordingPeripheral(timeline)
    peripherals.register(peripheral)

    _run(app_instance)
    app_instance._shutdown()
    app_instance._shutdown()

    assert timeline.count("brick-1") == 1
    assert peripheral.stop_count == 1
    assert recording.stop_called.is_set()


def test_case_6_bricks_cannot_be_started_while_shutting_down(app_instance, peripherals):
    """Condition: a brick is registered after the shutdown began.
    Expectation: it is refused, since the shutdown has already snapshotted what it will stop.
    """
    timeline: list[str] = []

    _run(app_instance)
    app_instance._shutdown()

    late = RecordingBrick("late", timeline)

    assert late not in app_instance._waiting_queue
    assert late not in app_instance._running_queue


def test_case_6_atexit_fallback_releases_after_a_framework_managed_run(app_instance, peripherals, monkeypatch):
    """Condition: the process lifecycle is owned by a framework, so _shutdown() never runs.
    Expectation: the interpreter-exit fallback releases the peripherals instead, exactly once.
    """
    monkeypatch.setattr(app_instance, "_is_framework_managed", lambda: True)
    peripheral = RecordingPeripheral([])
    peripherals.register(peripheral)

    app_instance.run()

    assert not peripheral.stopped.is_set(), "a framework-managed run must not stop peripherals itself"

    peripheral_registry._stop_peripherals_at_exit()

    assert peripheral.stop_count == 1


def test_case_6_atexit_fallback_is_a_noop_after_a_normal_shutdown(app_instance, peripherals):
    """Condition: the app shut down normally and the interpreter then exits.
    Expectation: the peripherals are not released a second time.
    """
    peripheral = RecordingPeripheral([])
    peripherals.register(peripheral)

    _run(app_instance)
    app_instance._shutdown()
    peripheral_registry._stop_peripherals_at_exit()

    assert peripheral.stop_count == 1


def test_case_6_the_shutdown_runs_when_the_user_loop_calls_sys_exit(app_instance, peripherals):
    """Condition: the user loop calls sys.exit(), so a SystemExit escapes the app loop.
    Expectation: the bricks are stopped and the peripherals released, in that order, before the
    SystemExit carries on. The shutdown used to be skipped entirely on this path, leaving the
    release to the interpreter-exit fallback, which stops no bricks at all.
    """
    timeline: list[str] = []
    recording = RecordingBrick("brick-1", timeline)
    peripheral = RecordingPeripheral(timeline)
    peripherals.register(peripheral)

    escaped = _run_until_it_returns(app_instance, _raiser(SystemExit(7)))

    assert isinstance(escaped, SystemExit), f"escaped={escaped!r}"
    assert escaped.code == 7, "the exit code must reach the interpreter unchanged"
    assert timeline == ["brick-1", "peripheral"]
    assert recording.stop_called.is_set()
    assert not app_instance._running


def test_case_6_the_shutdown_runs_when_a_base_exception_escapes_the_loop(app_instance, peripherals):
    """Condition: a BaseException is raised into the app loop, as a termination signal would be.
    Expectation: the shutdown still runs, and the exception propagates untouched.
    """
    timeline: list[str] = []
    RecordingBrick("brick-1", timeline)
    peripheral = RecordingPeripheral(timeline)
    peripherals.register(peripheral)

    terminated = Terminated("SIGTERM")
    escaped = _run_until_it_returns(app_instance, _raiser(terminated))

    assert escaped is terminated
    assert timeline == ["brick-1", "peripheral"]


def test_case_6_a_failing_shutdown_does_not_replace_the_propagating_exception(app_instance, peripherals, monkeypatch):
    """Condition: the app is already terminating on an exception and the shutdown itself fails.
    Expectation: the original exception is the one that reaches the top level. The global
    excepthook (see app_utils/errors.py) reports whatever gets there, so a shutdown failure
    taking its place would replace the error the user has to read with an unrelated one.
    """

    def broken_shutdown() -> None:
        raise RuntimeError("shutdown blew up")

    monkeypatch.setattr(app_instance, "_shutdown", broken_shutdown)

    original = Terminated("the failure the user needs to see")
    escaped = _run_until_it_returns(app_instance, _raiser(original))

    assert escaped is original, f"the excepthook would have reported {escaped!r} instead"


def test_case_6_a_failing_peripheral_release_does_not_abort_the_shutdown(app_instance, peripherals, monkeypatch):
    """Condition: releasing the peripherals raises, e.g. the interpreter cannot allocate.
    Expectation: the shutdown still completes and reports itself done, instead of leaving the app
    marked as running.
    """
    timeline: list[str] = []
    recording = RecordingBrick("brick-1", timeline)

    def broken(timeout):
        raise RuntimeError("the registry blew up")

    monkeypatch.setattr(peripherals, "stop_all_once", broken)

    _run(app_instance)
    app_instance._shutdown()

    assert recording.stop_called.is_set(), "the bricks must still have been stopped"
    assert timeline == ["brick-1"]
    assert not app_instance._running, "the shutdown did not complete"


def test_case_6_a_peripheral_registered_by_a_brick_stop_is_still_released(app_instance, peripherals):
    """Condition: a brick registers a peripheral from its own stop().
    Expectation: it is released too, and still after every brick. This is what pins the release
    snapshot to after the brick sweep: taking it any earlier would silently drop this peripheral,
    and the interpreter-exit fallback cannot cover it either once the latch is armed.
    """
    timeline: list[str] = []
    late = RecordingPeripheral(timeline)

    @brick
    class LateRegisteringBrick:
        def stop(self) -> None:
            timeline.append("brick")
            peripheral_registry.Peripherals.register(late)

    LateRegisteringBrick()

    _run(app_instance)
    app_instance._shutdown()

    assert timeline == ["brick", "peripheral"]
    assert late.stop_count == 1


def test_case_6_the_budgets_fit_in_the_grace_period():
    """The budgets are what keeps the process from being killed mid-release, so their sum is an
    invariant and not a preference: raising one without lowering another has to fail here.
    """
    assert REAL_BRICKS_BUDGET > 0, "the bricks must get a budget"
    assert REAL_PERIPHERALS_BUDGET > 0, "the peripherals must get a budget"

    total = REAL_BRICKS_BUDGET + REAL_PERIPHERALS_BUDGET + app.SHUTDOWN_HEADROOM_S
    assert total <= app.SHUTDOWN_GRACE_PERIOD_S, (
        f"the shutdown budgets add up to {total:.1f}s, over the {app.SHUTDOWN_GRACE_PERIOD_S:.1f}s grace period"
    )

    # The lock wait is nested inside the brick budget, so it must stay a small part of it
    assert REAL_LOCK_BUDGET <= REAL_BRICKS_BUDGET / 2

    # The interpreter-exit fallback stands in for the shutdown and is on the same clock
    assert peripheral_registry.PERIPHERAL_STOP_BUDGET_S <= REAL_PERIPHERALS_BUDGET


def test_case_6_a_pathological_shutdown_fits_in_the_grace_period(app_instance, peripherals, real_budgets):
    """Condition: the worst case, with the production budgets - a brick whose stop() never
    returns, bricks whose worker threads never return, and a peripheral whose stop() never
    returns.
    Expectation: _shutdown() returns inside the grace period, leaving the headroom for the
    interpreter teardown. This is the end-to-end statement that the budgets are sized right:
    overrunning the grace period is what gets the process killed while a device is still held.
    """
    release = threading.Event()
    block = threading.Event()

    hanging = HangingStopBrick(release)
    for n in range(3):
        StuckLoopBrick(f"stuck-{n}")
    ManyWorkersBrick()

    blocked = BlockingPeripheral(block)
    peripherals.register(blocked)

    _run(app_instance)

    try:
        started_at = time.monotonic()
        app_instance._shutdown()
        elapsed = time.monotonic() - started_at

        assert elapsed <= app.SHUTDOWN_GRACE_PERIOD_S, f"shutdown took {elapsed:.2f}s, the process would have been killed"
        # Fires before the grace-period assert above, with a clearer reason: the shutdown is meant
        # to spend its budgets and stop, so the only slack here is CI scheduling
        assert elapsed <= REAL_BRICKS_BUDGET + REAL_PERIPHERALS_BUDGET + 0.4, f"shutdown took {elapsed:.2f}s, it overran its budgets"
        assert hanging.stop_entered.is_set(), "the brick was never asked to stop"
        assert blocked.stop_entered.is_set(), "the peripheral was never asked to release"
    finally:
        release.set()
        block.set()


@pytest.mark.integration
def test_case_6_atexit_hook_is_armed_in_a_real_interpreter():
    """Condition: a process registers a peripheral and exits without stopping it.
    Expectation: the hook installed at import time releases it anyway.
    """
    script = (
        "from arduino.app_utils.peripheral_registry import Peripherals\n"
        "class P:\n"
        "    def stop(self):\n"
        "        print('RELEASED', flush=True)\n"
        "p = P()\n"
        "Peripherals.register(p)\n"
        "raise SystemExit(0)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert "RELEASED" in result.stdout, f"stdout={result.stdout!r} stderr={result.stderr!r}"
