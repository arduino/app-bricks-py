# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# CASE 5: Manual brick management before App.run()
# Validates that manually managing a brick before the main app loop starts
# does not interfere with the app's automatic lifecycle management.
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


# A simple brick for testing start/stop calls
@brick
class SimpleBrick:
    def __init__(self, name=""):
        self.name = name
        self.start_called = threading.Event()
        self.stop_called = threading.Event()

    def start(self):
        self.start_called.set()

    def stop(self):
        self.stop_called.set()

    def __repr__(self):
        return f"SimpleBrick(name='{self.name}')"


def test_case_5_manual_start_and_stop_before_run(app_instance):
    """Condition: manual_brick is created and manually started/stopped BEFORE App.run(), auto_brick is created and managed automatically.
    Expectation: auto_brick and manual_brick are not affected by each other's lifecycle.
    """
    # Brick to be managed automatically by App.run()
    auto_brick = SimpleBrick("Auto")

    # Brick to be managed manually
    manual_brick = SimpleBrick("Manual")

    # Manually start and stop the first brick
    app_instance.start_brick(manual_brick)
    assert manual_brick.start_called.wait(timeout=1), "Manual brick did not start"
    assert not manual_brick.stop_called.is_set()

    app_instance.stop_brick(manual_brick)
    assert manual_brick.stop_called.wait(timeout=1), "Manual brick did not stop"

    # Run the app, which should only start auto_brick
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()

    # Check that the auto brick was also started
    assert auto_brick.start_called.wait(timeout=1), "Auto brick was not started by App.run()"
    assert not auto_brick.stop_called.is_set(), "Auto brick should not be stopped yet"

    # Stop the app
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    # Verify final states
    assert auto_brick.stop_called.is_set(), "Auto brick was not stopped on app shutdown"
    assert manual_brick.start_called.is_set()
    assert manual_brick.stop_called.is_set()


def test_case_5_manual_start_is_stopped_by_run_exit(app_instance):
    """Condition: manual_brick is created and manually started BEFORE App.run(), auto_brick is created and managed automatically.
    Expectation: auto_brick and manual_brick are both automatically stopped when App.run() exits.
    """
    # Brick to be managed automatically by App.run()
    auto_brick = SimpleBrick("Auto")

    # Brick to be managed manually
    manual_brick = SimpleBrick("Manual")

    # Manually start the first brick, but do not stop it
    app_instance.start_brick(manual_brick)
    assert manual_brick.start_called.wait(timeout=1), "Manual brick did not start"
    assert not manual_brick.stop_called.is_set()

    # Run the app, which should only start auto_brick
    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()

    # Check that the auto brick was also started
    assert auto_brick.start_called.wait(timeout=1), "Auto brick was not started by App.run()"
    assert not auto_brick.stop_called.is_set(), "Auto brick should not be stopped yet"

    # Stop the app. This should stop ALL running bricks.
    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    # Verify final states
    assert auto_brick.stop_called.is_set(), "Auto brick was not stopped on app shutdown"
    assert manual_brick.stop_called.is_set(), "Manually started brick was not stopped on app shutdown"
