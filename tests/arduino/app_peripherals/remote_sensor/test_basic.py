# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Basic functionality tests for RemoteSensor.

Tests initialization, lifecycle (start/stop), configuration, and context manager support.
"""

import time
import pytest

from arduino.app_peripherals.remote_sensor import RemoteSensor, RemoteSensorConfigError


def test_remote_sensor_initialization():
    """Test RemoteSensor can be initialized with default parameters."""
    sensor = RemoteSensor()
    assert sensor.host == "0.0.0.0"
    assert sensor.port == 8080
    assert sensor.timeout == 10
    assert sensor.data_format == "json"
    assert not sensor.is_started()


def test_remote_sensor_invalid_format():
    """Test RemoteSensor raises error for invalid data format."""
    with pytest.raises(RemoteSensorConfigError):
        RemoteSensor(data_format="invalid")


def test_remote_sensor_custom_parameters():
    """Test RemoteSensor can be initialized with custom parameters."""
    sensor = RemoteSensor(host="127.0.0.1", port=9000, timeout=5, data_format="csv")
    assert sensor.host == "127.0.0.1"
    assert sensor.port == 9000
    assert sensor.timeout == 5
    assert sensor.data_format == "csv"


def test_remote_sensor_start_stop():
    """Test RemoteSensor can be started and stopped."""
    sensor = RemoteSensor(port=8766)

    # Should not be started initially
    assert not sensor.is_started()

    # Start the sensor
    sensor.start()
    time.sleep(0.5)  # Give server time to start
    assert sensor.is_started()

    # Stop the sensor
    sensor.stop()
    assert not sensor.is_started()


def test_remote_sensor_context_manager():
    """Test RemoteSensor works as a context manager."""
    with RemoteSensor(port=8767) as sensor:
        time.sleep(0.5)  # Give server time to start
        assert sensor.is_started()

    # Should be stopped after context exit
    assert not sensor.is_started()


def test_remote_sensor_multiple_start():
    """Test that calling start() multiple times is safe."""
    sensor = RemoteSensor(port=8768)

    sensor.start()
    time.sleep(0.5)
    assert sensor.is_started()

    # Calling start again should be safe (no-op)
    sensor.start()
    assert sensor.is_started()

    sensor.stop()
    assert not sensor.is_started()


def test_remote_sensor_multiple_stop():
    """Test that calling stop() multiple times is safe."""
    sensor = RemoteSensor(port=8769)

    sensor.start()
    time.sleep(0.5)
    sensor.stop()
    assert not sensor.is_started()

    # Calling stop again should be safe (no-op)
    sensor.stop()
    assert not sensor.is_started()


def assert_all_dict(data):
    assert all(isinstance(x, dict) for x in data)
