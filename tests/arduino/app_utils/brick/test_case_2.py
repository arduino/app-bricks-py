# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 2: managed brick with non-blocking loop() method
# Validates that start/stop are called once, and loop() is called multiple times.
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
class LoopingBrick:
    def __init__(self):
        self.start_called_count = 0
        self.stop_called_count = 0
        self.loop_called_count = 0

    def start(self):
        self.start_called_count += 1

    def stop(self):
        self.stop_called_count += 1

    def loop(self):
        self.loop_called_count += 1
        time.sleep(0.05)  # Simulate work to avoid busy-waiting


# Test cases
def test_case_2_instance_before_run(app_instance):
    """Condition: instance is created BEFORE App.run().
    Expectation: framework should automatically call loop() multiple times.
    """
    instance = LoopingBrick()

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.start_called_count == 1, "Start should be called once"
    assert instance.stop_called_count == 1, "Stop should be called once"
    assert instance.loop_called_count > 1, "Loop should be called multiple times"


def test_case_2_instance_after_run(app_instance):
    """Condition: instance is created AFTER App.run().
    Expectation: it must be started manually, after which loop() should be called multiple times.
    """
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    instance = LoopingBrick()
    assert instance.loop_called_count == 0

    # Manually start the brick
    app_instance.start_brick(instance)
    time.sleep(0.2)  # Allow time for the loop to run a few times
    assert instance.start_called_count == 1
    assert instance.loop_called_count > 1, "Loop should run after start_brick()"

    # Stop the app
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.stop_called_count == 1, "Stop should be called on app shutdown"
    final_loop_count = instance.loop_called_count
    time.sleep(0.1)
    assert instance.loop_called_count == final_loop_count, "Loop should stop after shutdown"
