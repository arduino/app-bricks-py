# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 0: unmanaged brick
# Validates that a non-brick (simple class) is ignored by the framework.
import time
import threading

import pytest

import arduino.app_utils.app as app
from arduino.app_utils import AppController


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


# Test class definition
class PlainClass:
    def __init__(self):
        self.start_called = False
        self.stop_called = False

    def start(self):
        self.start_called = True

    def stop(self):
        self.stop_called = True


# Test cases
def test_case_0_instance_before_run(app_instance):
    """Condition: instance is created BEFORE App.run().
    Expectation: the framework should NOT automatically call any of its methods because it's not a brick.
    """
    instance = PlainClass()

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert not instance.start_called, "Start should not be called on a plain class"
    assert not instance.stop_called, "Stop should not be called on a plain class"


def test_case_0_instance_after_run(app_instance):
    """Condition: instance is created AFTER App.run().
    Expectation: the framework should not manage the instance automatically, but should if started manually.
    """
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    instance = PlainClass()

    # Even after some time, the methods should not have been called automatically
    time.sleep(0.1)
    assert not instance.start_called, "Start should not be called automatically"
    assert not instance.stop_called, "Stop should not be called automatically"

    # Manually starting the plain class should still work
    app_instance.start_brick(instance)
    time.sleep(0.1)
    assert instance.start_called, "Start should be called after manual start_brick()"

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    # Final check after shutdown
    assert instance.start_called, "Start should remain called"
    assert instance.stop_called, "Stop should be called by _stop_all_bricks because the instance was manually started"
