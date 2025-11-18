# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
from collections import deque
import time

from .utils import _has_callable_method, _brick_name
from .logger import Logger

logger = Logger("App")


class AppController:
    """AppController orchestrates the entire application lifecycle by managing brick startup, shutdown, and their
    loops execution in a controlled, structured way.

    It discovers methods named 'loop'/'execute' or decorated with @loop/@execute and runs each in a separate thread.
    Also methods named 'start' and 'stop' are called automatically depending on the App's lifecycle.

    Bricks that are instantiated before App.run() is called will be started/stopped automatically.
    Bricks that are started manually via App.start_brick() must have their lifecycle managed manually by the user.

    When App.run() exits, all bricks, including those started manually, will be stopped to ensure a clean shutdown.
    """

    def __init__(self):
        self._waiting_queue = deque()
        self._running_queue = deque()
        self._brick_states: dict[any, list[tuple[threading.Thread, threading.Event]]] = {}
        self._app_lock = threading.Lock()

    def register(self, brick):
        """Registers a brick for being managed automatically by the AppController.

        If the brick is not running, it will be auto-started when App.run() will be called.
        If the brick is already running, this method does nothing.
        """
        with self._app_lock:
            if brick in self._running_queue:
                return

            if brick not in self._waiting_queue:
                self._waiting_queue.append(brick)
                logger.debug(f"Registered brick '{_brick_name(brick)}' to start on next App.run().")

    def unregister(self, brick):
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

    def start_bricks(self):
        """Starts the application and all registered bricks.

        Use this method if you don't want to block the main thread and handle it as you wish.

        The bricks should be manually managed by the user by calling App.stop_bricks().
        """
        self._start_managed_bricks()

    def start_brick(self, brick):
        """Immediately starts a single brick and all its runnable methods.

        This brick should be manually managed by the user by calling App.stop_brick().
        """
        # Bricks may be manually started before App.run() is called, ensure they don't appear in the waiting queue
        self.unregister(brick)
        with self._app_lock:
            self._start(brick)

    def stop_bricks(self):
        """Stops the application and all running bricks."""
        self._stop_all_bricks()

    def stop_brick(self, brick):
        """Immediately stops a single running brick."""
        with self._app_lock:
            self._stop(brick)

    def run(self, user_loop: callable = None):
        """Starts all registered bricks and keeps the main thread alive, waiting for a shutdown signal (Ctrl+C).

        If a user_loop callable is provided, it will be executed instead of the default infinite loop.

        Args:
            user_loop (callable, optional): A user-defined function to run instead of the default infinite loop.
        """
        print("======== App is starting ============================", flush=True)
        self._start_managed_bricks()
        logger.info("App started")
        self.loop(user_loop)
        logger.info("App is shutting down")
        self._stop_all_bricks()
        print("======== App shutdown completed =====================", flush=True)

    def loop(self, user_loop: callable = None):
        """This method keeps the application running, blocking until a KeyboardInterrupt (Ctrl+C) occurs.

        If a user_loop callable is provided, it will be executed inside an infinite loop and
        called repeatedly every iteration.

        Args:
            user_loop (callable, optional): A user-defined function to run inside an infinite loop.
        """
        try:
            if user_loop:
                while True:
                    user_loop()
            else:
                while True:
                    time.sleep(10)
        except StopIteration:
            logger.debug("StopIteration received from user loop")
        except KeyboardInterrupt:
            logger.debug("KeyboardInterrupt received")

    def _start_managed_bricks(self):
        with self._app_lock:
            while self._waiting_queue:
                brick = self._waiting_queue.popleft()
                self._start(brick)
        logger.debug("All managed bricks started")

    def _stop_all_bricks(self):
        with self._app_lock:
            bricks_to_stop = list(self._running_queue)
            for brick in reversed(bricks_to_stop):
                self._stop(brick)
        logger.debug("All bricks stopped")

    def _discover_runnable_methods(self, brick):
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

    def _start(self, brick):
        """Starts a single brick and its worker threads. Must be called while holding _app_lock."""
        if brick in self._running_queue:
            # TODO: we should raise an exception here
            logger.warning(f"Brick '{_brick_name(brick)}' is already running")
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

    def _stop(self, brick):
        """Stops a single brick and its worker threads. Must be called while holding _app_lock."""
        if brick not in self._running_queue:
            # TODO: we should raise an exception here
            logger.warning(f"Brick '{_brick_name(brick)}' is not running")
            return

        # Call the brick's stop method right away. This might cause the loop method to be called even after the stop
        # has been issued but this is a guarantee that we can't provide. We might as well call stop right away and gain
        # the possibility to better handle blocking bricks which contain long-running tasks that can be stopped only
        # externally and would otherwise result in a timeout when joining the worker thread.
        if _has_callable_method(brick, "stop"):
            try:
                logger.debug(f"Calling stop() for brick: '{_brick_name(brick)}'")
                brick.stop()
            except Exception as e:
                logger.exception(f"Failed to stop brick '{_brick_name(brick)}': {e}")

        if brick in self._brick_states:
            for thread, brick_is_running in self._brick_states.pop(brick):
                brick_is_running.clear()
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Worker thread '{thread.name}' for '{_brick_name(brick)}' did not terminate in time")

        if brick in self._running_queue:
            self._running_queue.remove(brick)

        logger.debug(f"Brick '{_brick_name(brick)}' stopped successfully")

    def _method_runner(self, brick, method, method_type, brick_is_running):
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
