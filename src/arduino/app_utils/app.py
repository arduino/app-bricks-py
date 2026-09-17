# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import signal
import sys
import threading
from collections import deque
import time

from . import peripheral_registry
from .utils import _has_callable_method, _brick_name
from .logger import Logger
from collections.abc import Callable
from types import FrameType
from typing import Any, Never

logger = Logger("App")

# The whole shutdown must fit in the container stop grace period: a process killed while still
# holding an exclusive peripheral can leave it unusable for everyone else, so budgets are
# deliberately conservative and leave headroom for the interpreter teardown.
SHUTDOWN_BRICKS_BUDGET_S = 5.0
"""Wall-clock budget, in seconds, for stopping every brick. Shared globally, not per brick."""

SHUTDOWN_PERIPHERALS_BUDGET_S = 2.0
"""Wall-clock budget, in seconds, for releasing every peripheral, once the bricks are stopped."""

SHUTDOWN_LOCK_BUDGET_S = 1.0
"""Max time, in seconds, spent waiting for the app lock before stopping bricks without it."""

WORKER_JOIN_TIMEOUT_S = 5.0
"""Per-worker-thread join timeout used when no global deadline applies, i.e. by stop_brick()."""


class AppController:
    """AppController orchestrates the entire application lifecycle by managing brick startup, shutdown, and their
    loops execution in a controlled, structured way.

    It discovers methods named 'loop'/'execute' or decorated with @loop/@execute and runs each in a separate thread.
    Also methods named 'start' and 'stop' are called automatically depending on the App's lifecycle.

    Bricks that are instantiated before App.run() is called will be started/stopped automatically.
    Bricks that are started manually via App.start_brick() must have their lifecycle managed manually by the user.

    When App.run() exits, all bricks, including those started manually, will be stopped to ensure a clean shutdown.
    """

    def __init__(self) -> None:
        self._waiting_queue = deque()
        self._running_queue = deque()
        self._brick_states: dict[any, list[tuple[threading.Thread, threading.Event]]] = {}
        self._app_lock = threading.Lock()
        self._running = False
        self._stopping = False

    def register(self, brick: object) -> None:
        """Registers a brick for being managed automatically by the AppController.

        If the brick is not running, it will be auto-started when App.run() will be called.
        If the brick is already running, this method does nothing.
        """
        if self._stopping:
            logger.warning(f"Ignoring registration of brick '{_brick_name(brick)}': the app is shutting down")
            return

        with self._app_lock:
            if brick in self._running_queue:
                return

            if brick not in self._waiting_queue:
                self._waiting_queue.append(brick)
                logger.debug(f"Registered brick '{_brick_name(brick)}' to start on next App.run().")

    def unregister(self, brick: object) -> None:
        """Unregisters a brick from being managed automatically by the AppController.

        If the brick is not running, it won't be auto-started anymore when App.run() will be called.
        If the brick is already running, this method does nothing.
        """
        with self._app_lock:
            if brick in self._running_queue:
                return

            if brick in self._waiting_queue:
                self._waiting_queue.remove(brick)
                logger.debug(f"Unregistered brick '{_brick_name(brick)}' from starting on next App.run().")

    def start_bricks(self) -> None:
        """Starts the application and all registered bricks.

        Use this method if you don't want to block the main thread and handle it as you wish.

        The bricks should be manually managed by the user by calling App.stop_bricks().
        """
        self._start_managed_bricks()

    def start_brick(self, brick: object) -> None:
        """Immediately starts a single brick and all its runnable methods.

        This brick should be manually managed by the user by calling App.stop_brick().
        """
        # Bricks may be manually started before App.run() is called, ensure they don't appear in the waiting queue
        self.unregister(brick)
        with self._app_lock:
            self._start(brick)

    def stop_bricks(self) -> None:
        """Stops the application and all running bricks.

        All the bricks are stopped within a single global time budget. This does not release the
        registered peripherals, which are released by the App's own shutdown.
        """
        self._stop_all_bricks()

    def stop_brick(self, brick: object) -> None:
        """Immediately stops a single running brick."""
        with self._app_lock:
            self._stop(brick)

    def run(self, user_loop: callable = None) -> None:
        """Starts all registered bricks and keeps the main thread alive, waiting for a shutdown signal (Ctrl+C).

        If a user_loop callable is provided, it will be executed instead of the default infinite loop.

        When running inside a framework that manages the process lifecycle (e.g. Streamlit),
        bricks are started but the blocking loop is skipped. The framework is responsible for
        keeping the process alive; brick daemon threads terminate automatically with the process.

        Args:
            user_loop (callable, optional): A user-defined function to run instead of the default infinite loop.
        """
        # Idempotent: if already running (e.g. Streamlit re-runs the script), just return.
        if self._running:
            return

        self._running = True
        self._stopping = False
        # Re-arm the one-shot peripheral release so a restarted app releases its peripherals again
        peripheral_registry.Peripherals.reset()
        self._start_managed_bricks()
        logger.info("App started")

        if self._is_framework_managed():
            logger.info("Running in framework-managed mode (process lifecycle handled externally)")
            return

        try:
            exit_code = self.loop(user_loop)
        except BaseException:
            # loop() handles Exception itself, so getting here means SystemExit from user code or
            # a BaseException raised into the main thread. The shutdown is the only thing that
            # releases the peripherals in order, so it has to run on this path too.
            self._shutdown_quietly()
            raise

        self._shutdown()

        if exit_code:
            sys.exit(exit_code)

    def _shutdown_quietly(self) -> None:
        """Runs the shutdown while an exception is already propagating, swallowing its failures."""
        try:
            self._shutdown()
        except Exception as e:
            logger.exception(f"Shutdown failed while the app was already terminating: {e}")

    def _shutdown(self) -> None:
        """Performs a clean, time-bounded shutdown of all bricks and then of all peripherals.

        The whole sequence is bounded by SHUTDOWN_BRICKS_BUDGET_S + SHUTDOWN_PERIPHERALS_BUDGET_S
        so that it always completes within the container stop grace period. Overrunning it gets
        the process killed, and a process killed while holding an exclusive peripheral can leave
        that peripheral unusable until its driver or service is restarted.
        """
        if not self._running:
            return

        logger.info("App is shutting down")
        self._stopping = True
        started_at = time.monotonic()

        try:
            self._stop_all_bricks()
        finally:
            # Peripherals are released only once every brick has been stopped: bricks own the
            # capture loops, and a peripheral's stop() has to wait for an in-flight capture to
            # return. Releasing them in a finally means a failure while stopping the bricks can
            # never leave a device behind.
            self._stop_all_peripherals()

        self._running = False
        logger.debug(f"Shutdown completed in {time.monotonic() - started_at:.2f}s")
        print("======== App shutdown completed =====================", flush=True)

    def _stop_all_peripherals(self) -> None:
        """Releases every registered peripheral, at most once per process.

        This must never run while holding _app_lock: stopping a peripheral can block on that
        peripheral's own lock, and _app_lock is not reentrant, so holding it here would stall
        any concurrent register()/start_brick() for the whole peripheral budget.
        """
        try:
            peripheral_registry.Peripherals.stop_all_once(SHUTDOWN_PERIPHERALS_BUDGET_S)
        except Exception as e:
            # Releasing the peripherals is the last step of the shutdown: a failure here must not
            # abort it, or _running would stay set and the completion never be reported.
            # BaseException is deliberately not caught: it has to propagate, and the registry
            # leaves its one-shot latch open so the interpreter-exit fallback can try again.
            logger.exception(f"Failed to release the peripherals: {e}")

    def _is_framework_managed(self) -> bool:
        """Detect if running inside a framework that manages the process lifecycle.

        Returns True when the script is being executed in a worker thread by a framework
        like Streamlit, which owns the main thread and the process lifecycle.
        """
        if threading.current_thread() is threading.main_thread():
            return False

        # Explicitly detect Streamlit's ScriptRunContext (definitive signal)
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            if get_script_run_ctx(suppress_warning=True) is not None:
                return True
        except ImportError:
            pass

        return False

    def loop(self, user_loop: callable = None) -> int:
        """This method keeps the application running, blocking until a KeyboardInterrupt (Ctrl+C) occurs.

        If a user_loop callable is provided, it will be executed inside an infinite loop and
        called repeatedly every iteration.

        Args:
            user_loop (callable, optional): A user-defined function to run inside an infinite loop.

        Returns:
            int: The exit code describing why the loop terminated:
                - 0 for a clean termination
                - 128 + signal number for termination signals
                - a code < 128 for other errors
        """

        class SignalReceived(BaseException):
            def __init__(self, signum: int) -> None:
                self.signum = signum

        def handle_signal(signum: int, frame: FrameType | None) -> Never:
            raise SignalReceived(signum)

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, handle_signal)

        try:
            if user_loop:
                while True:
                    user_loop()
            else:
                while True:
                    time.sleep(10)
        except StopIteration:
            logger.debug("StopIteration received from user loop")
            return 0
        except KeyboardInterrupt:
            logger.debug("KeyboardInterrupt received")
            return min(128 + signal.SIGINT, 255)
        except SignalReceived as signal_received:
            logger.debug(f"Termination signal {signal_received.signum} received")
            return min(128 + signal_received.signum, 255)
        except Exception:
            logger.exception("Unhandled exception in application loop")
            return 1

    def _start_managed_bricks(self) -> None:
        with self._app_lock:
            while self._waiting_queue:
                brick = self._waiting_queue.popleft()
                self._start(brick)
        logger.debug("All managed bricks started")

    def _stop_all_bricks(self) -> None:
        """Stops every running brick, in reverse start order, within a single global budget.

        The budget is global rather than per brick: a brick that refuses to stop consumes its own
        fair share and is then abandoned, so the bricks after it, and the peripheral release that
        follows, still get their time.
        """
        deadline = time.monotonic() + SHUTDOWN_BRICKS_BUDGET_S

        # The budget starts before the lock is taken, so waiting on a concurrent register() or
        # start_brick() cannot push the shutdown past the deadline. If the lock cannot be taken
        # in time the bricks are stopped anyway: a racing queue mutation is a much smaller
        # problem than the process being killed while a peripheral is still held.
        acquired = self._app_lock.acquire(timeout=max(0.0, min(SHUTDOWN_LOCK_BUDGET_S, deadline - time.monotonic())))
        if not acquired:
            logger.warning("App lock still held at shutdown, stopping bricks without it")

        try:
            bricks_to_stop = list(reversed(self._running_queue))
            total = len(bricks_to_stop)
            for index, brick in enumerate(bricks_to_stop):
                # Fair share of whatever is left. Slack left unused by a fast brick is reclaimed
                # by the following ones, since the remaining time is recomputed every iteration.
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    self._stop(brick, deadline=time.monotonic() + remaining / (total - index))
                except Exception as e:
                    # Never let one brick abort the sweep: the bricks after it, and the peripheral
                    # release, still have to happen.
                    logger.exception(f"Failed to stop brick '{_brick_name(brick)}': {e}")
        finally:
            if acquired:
                self._app_lock.release()

        logger.debug("All bricks stopped")

    def _discover_runnable_methods(self, brick: object) -> list[tuple[Callable[..., object], str]]:
        """Discovers and validates all methods marked with @loop/@execute or named loop/execute."""
        methods = []
        processed_names = set()

        for name in dir(brick):
            if name.startswith("__") or name in processed_names:
                continue

            try:
                attr = getattr(brick, name)
                is_loop = hasattr(attr, "_is_loop") or name == "loop"
                is_execute = hasattr(attr, "_is_execute") or name == "execute"

                if is_loop or is_execute:
                    if _has_callable_method(brick, name):
                        method_type = "loop" if is_loop else "execute"
                        methods.append((attr, method_type))
                        processed_names.add(name)
            except AttributeError:
                # Some attributes from dir() might not be gettable, just skip them
                continue

        return methods

    def _start(self, brick: Any) -> None:  # noqa: ANN401
        """Starts a single brick and its worker threads. Must be called while holding _app_lock."""
        if brick in self._running_queue:
            # TODO: we should raise an exception here
            logger.warning(f"Brick '{_brick_name(brick)}' is already running")
            return

        if self._stopping:
            # A brick started now would not be in the snapshot the shutdown is walking, so it
            # would never be stopped.
            logger.warning(f"Refusing to start brick '{_brick_name(brick)}': the app is shutting down")
            return

        try:
            if _has_callable_method(brick, "start"):
                logger.debug(f"Calling start() for brick: '{_brick_name(brick)}'")
                brick.start()

            runnable_methods = self._discover_runnable_methods(brick)
            if runnable_methods:
                self._brick_states[brick] = []
                for method, method_type in runnable_methods:
                    brick_is_running = threading.Event()
                    brick_is_running.set()

                    thread_name = f"{brick.__class__.__name__}.{method.__name__}"
                    thread = threading.Thread(
                        target=self._method_runner,
                        args=(brick, method, method_type, brick_is_running),
                        name=thread_name,
                        daemon=True,
                    )
                    thread.start()

                    self._brick_states[brick].append((thread, brick_is_running))

            self._running_queue.append(brick)
        except Exception as e:
            # TODO: we should raise an exception here
            logger.exception(f"Failed to start brick '{_brick_name(brick)}': {e}")

    def _stop(self, brick: Any, deadline: float | None = None) -> None:  # noqa: ANN401
        """Stops a single brick and its worker threads. Must be called while holding _app_lock.

        Args:
            brick: The brick to stop.
            deadline (float | None): Absolute time.monotonic() deadline for this brick. When None,
                stop() is called synchronously and unbounded and every worker thread is joined for
                WORKER_JOIN_TIMEOUT_S, which is the behaviour the single-brick API relies on.
        """
        if brick not in self._running_queue:
            # TODO: we should raise an exception here
            logger.warning(f"Brick '{_brick_name(brick)}' is not running")
            return

        # Call the brick's stop method right away. This might cause the loop method to be called even after the stop
        # has been issued but this is a guarantee that we can't provide. We might as well call stop right away and gain
        # the possibility to better handle blocking bricks which contain long-running tasks that can be stopped only
        # externally and would otherwise result in a timeout when joining the worker thread.
        self._call_brick_stop(brick, deadline)

        if brick in self._brick_states:
            states = self._brick_states.pop(brick)

            # Signal every worker before joining any of them: clearing the events one at a time
            # would leave the remaining loops spinning for the whole duration of each join.
            for _, brick_is_running in states:
                brick_is_running.clear()

            threads_left = len(states)
            for thread, _ in states:
                if deadline is None:
                    timeout = WORKER_JOIN_TIMEOUT_S
                else:
                    remaining = deadline - time.monotonic()
                    # Budget already spent: don't wait, but still report whether the thread died.
                    timeout = remaining / threads_left if remaining > 0 else 0.0
                thread.join(timeout=timeout)
                threads_left -= 1
                if thread.is_alive():
                    logger.warning(f"Worker thread '{thread.name}' for '{_brick_name(brick)}' did not terminate in time")

        if brick in self._running_queue:
            self._running_queue.remove(brick)

        logger.debug(f"Brick '{_brick_name(brick)}' stopped successfully")

    def _call_brick_stop(self, brick: Any, deadline: float | None) -> None:  # noqa: ANN401
        """Calls the brick's stop() method, bounded by the given deadline.

        stop() is brick or user code and may block for an arbitrarily long time, on a network
        call, an external process or a lock it does not own, and a Python thread cannot be
        interrupted. So when a deadline applies the call is dispatched to a short-lived daemon
        thread and abandoned if it overruns: that brick loses its clean stop, but the remaining
        bricks and, most importantly, the peripheral release still get their budget.

        A consequence worth knowing: during an app shutdown stop() no longer runs on the caller's
        thread, and a brick whose stop() overran may still be running while the next brick is
        stopped. stop_brick() keeps calling it synchronously.

        Args:
            brick: The brick to stop.
            deadline (float | None): Absolute time.monotonic() deadline, or None to call stop()
                synchronously and wait for it indefinitely.
        """
        if not _has_callable_method(brick, "stop"):
            return

        def call_stop() -> None:
            try:
                logger.debug(f"Calling stop() for brick: '{_brick_name(brick)}'")
                brick.stop()
            except Exception as e:
                logger.exception(f"Failed to stop brick '{_brick_name(brick)}': {e}")

        if deadline is None:
            call_stop()
            return

        # Dispatch even with no budget left: stop() is where the brick releases its resources and
        # it normally returns immediately, so it is always worth starting. We just don't wait.
        stopper = threading.Thread(target=call_stop, name=f"stop-{_brick_name(brick)}", daemon=True)
        stopper.start()
        stopper.join(timeout=max(0.0, deadline - time.monotonic()))
        if stopper.is_alive():
            logger.warning(f"stop() for brick '{_brick_name(brick)}' exceeded the shutdown budget, abandoning it")

    def _method_runner(self, brick: object, method: Callable[..., object], method_type: str, brick_is_running: threading.Event) -> None:
        """Target function for worker threads, running a brick's method."""
        try:
            if method_type == "execute":
                logger.debug(f"Executing blocking execute method '{method.__name__}' of '{_brick_name(brick)}'")
                if brick_is_running.is_set():
                    method()
            elif method_type == "loop":
                logger.debug(f"Starting non-blocking loop method '{method.__name__}' of '{_brick_name(brick)}'")
                while brick_is_running.is_set():
                    method()
        except StopIteration:
            logger.debug(f"Loop method '{method.__name__}' for brick '{_brick_name(brick)}' stopped iterating")
        except Exception as e:
            logger.exception(f"Exception in worker for brick '{_brick_name(brick)}', method '{method.__name__}': {e}")

        logger.debug(f"Worker for '{_brick_name(brick)}', method '{method.__name__}' terminated")


App = AppController()
