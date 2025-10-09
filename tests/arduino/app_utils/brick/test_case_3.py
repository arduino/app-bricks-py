# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 3: managed brick with blocking execute() method
# Validates that start(), stop(), and execute() are all called exactly once.
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
class BlockingExecuteBrick:
    def __init__(self):
        self.start_called_count = 0
        self.stop_called_count = 0
        self.execute_called_count = 0
        self.execute_finished = threading.Event()

    def start(self):
        self.start_called_count += 1

    def stop(self):
        self.stop_called_count += 1

    def execute(self):
        self.execute_called_count += 1
        self.execute_finished.set()  # Signal that execute has run


# Test cases
def test_case_3_instance_before_run(app_instance):
    """Condition: instance is created BEFORE App.run().
    Expectation: framework should automatically call execute() exactly once.
    """
    instance = BlockingExecuteBrick()

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()

    # Wait for the execute method to signal it has run
    finished_in_time = instance.execute_finished.wait(timeout=1)
    assert finished_in_time, "Execute method did not run in time"

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.start_called_count == 1, "Start should be called once"
    assert instance.stop_called_count == 1, "Stop should be called once"
    assert instance.execute_called_count == 1, "Execute should be called once"


def test_case_3_instance_after_run(app_instance):
    """Condition: Instance is created AFTER App.run().
    Expectation: framework should call execute() exactly once after manual start.
    """
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    instance = BlockingExecuteBrick()
    assert instance.execute_called_count == 0

    # Manually start the brick
    app_instance.start_brick(instance)

    finished_in_time = instance.execute_finished.wait(timeout=1)
    assert finished_in_time, "Execute method did not run in time after start_brick()"

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.start_called_count == 1, "Start should be called once"
    assert instance.stop_called_count == 1, "Stop should be called once"
    assert instance.execute_called_count == 1, "Execute should be called once"
