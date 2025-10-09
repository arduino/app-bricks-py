# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 4: managed brick with multiple loop() and execute() methods
# Validates that all decorated methods and default methods run correctly.
import time
import threading

import pytest

import arduino.app_utils.app as app
from arduino.app_utils import AppController, brick


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


# Test brick definition
@brick
class MultiMethodBrick:
    def __init__(self):
        self.start_called = threading.Event()
        self.stop_called = threading.Event()

        # Counters for loop methods
        self.default_loop_count = 0
        self.decorated_loop_count = 0

        # Events for execute methods
        self.default_execute_called = threading.Event()
        self.decorated_execute_called = threading.Event()

    def start(self):
        self.start_called.set()

    def stop(self):
        self.stop_called.set()

    # Default non-blocking loop
    def loop(self):
        self.default_loop_count += 1
        time.sleep(0.05)

    # Decorated non-blocking loop
    @brick.loop
    def other_loop(self):
        self.decorated_loop_count += 1
        time.sleep(0.05)

    # Default blocking execute
    def execute(self):
        self.default_execute_called.set()

    # Decorated blocking execute
    @brick.execute
    def other_execute(self):
        self.decorated_execute_called.set()


# Test case
def test_case_4_instance_before_run(app_instance):
    """Condition: instance is created BEFORE App.run().
    Expectation: framework should automatically call default and decorated methods respecting their blocking/non-blocking semantics.
    """
    instance = MultiMethodBrick()

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()

    # Wait for start and both execute methods to be called
    assert instance.start_called.wait(timeout=1), "start() was not called"
    assert instance.default_execute_called.wait(timeout=1), "execute() was not called"
    assert instance.decorated_execute_called.wait(timeout=1), "@execute method was not called"

    # Let loops run for a moment
    time.sleep(0.2)

    # Stop the app
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    # Assert stop was called
    assert instance.stop_called.is_set(), "stop() was not called"

    # Assert loops ran multiple times
    assert instance.default_loop_count > 1, "Default loop did not run multiple times"
    assert instance.decorated_loop_count > 1, "Decorated loop did not run multiple times"

    # Verify loops are no longer running
    final_default_count = instance.default_loop_count
    final_decorated_count = instance.decorated_loop_count
    time.sleep(0.1)
    assert instance.default_loop_count == final_default_count, "Default loop did not stop"
    assert instance.decorated_loop_count == final_decorated_count, "Decorated loop did not stop"


def test_case_4_instance_after_run(app_instance):
    """Condition: Instance is created AFTER App.run().
    Expectation: framework should call default and decorated methods respecting their blocking/non-blocking semantics only after manual start.
    """
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    instance = MultiMethodBrick()

    # Verify that no methods have been called yet, just from instantiation
    assert not instance.default_execute_called.is_set(), "execute() should not be called before start_brick()"
    assert not instance.decorated_execute_called.is_set(), "@execute method should not be called before start_brick()"
    assert instance.default_loop_count == 0, "Default loop should not run before start_brick()"
    assert instance.decorated_loop_count == 0, "Decorated loop should not run before start_brick()"

    # Manually start the brick
    app_instance.start_brick(instance)

    # Wait for start and both execute methods to be called
    assert instance.start_called.wait(timeout=1), "start() was not called"
    assert instance.default_execute_called.wait(timeout=1), "execute() was not called"
    assert instance.decorated_execute_called.wait(timeout=1), "@execute method was not called"

    # Let loops run for a moment
    time.sleep(0.2)

    # Stop the app
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    # Assert stop was called
    assert instance.stop_called.is_set(), "stop() was not called"

    # Assert loops ran multiple times
    assert instance.default_loop_count > 1, "Default loop did not run multiple times"
    assert instance.decorated_loop_count > 1, "Decorated loop did not run multiple times"

    # Verify loops are no longer running
    final_default_count = instance.default_loop_count
    final_decorated_count = instance.decorated_loop_count
    time.sleep(0.1)
    assert instance.default_loop_count == final_default_count, "Default loop did not stop"
    assert instance.decorated_loop_count == final_decorated_count, "Decorated loop did not stop"
