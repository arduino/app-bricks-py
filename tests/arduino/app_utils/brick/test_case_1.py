# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 1: managed brick without loop() or execute()
# Validates that only start() and stop() are called once.
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
class StartStopBrick:
    def __init__(self):
        self.start_called_count = 0
        self.stop_called_count = 0

    def start(self):
        self.start_called_count += 1

    def stop(self):
        self.stop_called_count += 1


# Test cases
def test_case_1_instance_before_run(app_instance):
    """Condition: instance is created BEFORE App.run().
    Expectation: framework should automatically call start() and stop() once.
    """
    instance = StartStopBrick()

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.start_called_count == 1, "Start should be called once"
    assert instance.stop_called_count == 1, "Stop should be called once"


def test_case_1_instance_after_run(app_instance):
    """Condition: instance is created AFTER App.run().
    Expectation: it must be started manually with start_brick().
    """
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    instance = StartStopBrick()
    # Brick is registered but not auto-started, so counts are 0
    assert instance.start_called_count == 0
    assert instance.stop_called_count == 0

    # Manually start the brick
    app_instance.start_brick(instance)
    time.sleep(0.1)
    assert instance.start_called_count == 1, "Start should be called after start_brick()"
    assert instance.stop_called_count == 0, "Stop should not be called yet"

    # Stop the app, which should stop the manually started brick
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert instance.start_called_count == 1, "Start should only be called once"
    assert instance.stop_called_count == 1, "Stop should be called on app shutdown"
