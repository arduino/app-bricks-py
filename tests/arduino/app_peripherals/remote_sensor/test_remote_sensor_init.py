# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""
Basic functionality tests for RemoteSensor.

Tests initialization, lifecycle (start/stop), configuration, and context manager support.
"""

import pytest

from arduino.app_peripherals.remote_sensor import RemoteSensor, RemoteSensorConfigError


def test_remote_sensor_initialization():
    """Test RemoteSensor can be initialized with default parameters."""
    sensor = RemoteSensor()
    assert sensor.url == "ws://0.0.0.0:8090"
    assert sensor.port == 8090
    assert sensor.timeout == 3
    assert sensor.secret is None
    assert sensor.encrypt is False
    assert sensor.auto_reconnect is True
    assert "none" in sensor.security_mode
    assert not sensor.is_started()


def test_remote_sensor_custom_parameters():
    """Test RemoteSensor can be initialized with custom parameters."""
    sensor = RemoteSensor(port=9000, timeout=5, secret="yolo", encrypt=True, auto_reconnect=False)
    assert sensor.url == "ws://0.0.0.0:9000"
    assert sensor.port == 9000
    assert sensor.timeout == 5
    assert sensor.secret == "yolo"
    assert sensor.encrypt is True
    assert sensor.auto_reconnect is False
    assert "encrypted" in sensor.security_mode


def test_remote_sensor_invalid_port():
    """Test RemoteSensor raises error for invalid port values."""
    with pytest.raises(RemoteSensorConfigError):
        RemoteSensor(port=-1)

    with pytest.raises(RemoteSensorConfigError):
        RemoteSensor(port=70000)


def test_remote_sensor_invalid_timeout():
    """Test RemoteSensor raises error for invalid timeout values."""
    with pytest.raises(RemoteSensorConfigError):
        RemoteSensor(timeout=0)

    with pytest.raises(RemoteSensorConfigError):
        RemoteSensor(timeout=-5)


def test_remote_sensor_start_stop():
    """Test RemoteSensor can be started and stopped."""
    sensor = RemoteSensor(port=0)
    sensor.auto_reconnect_delay = 0

    # Should not be started initially
    assert not sensor.is_started()

    # Start the sensor
    sensor.start()
    assert sensor.is_started()

    # Stop the sensor
    sensor.stop()
    assert not sensor.is_started()


def test_remote_sensor_context_manager():
    """Test RemoteSensor works as a context manager."""
    with RemoteSensor(port=0) as sensor:
        assert sensor.is_started()

    # Should be stopped after context exit
    assert not sensor.is_started()


def test_remote_sensor_multiple_start():
    """Test that calling start() multiple times is safe."""
    sensor = RemoteSensor(port=0)
    sensor.start()

    assert sensor.is_started()

    # Calling start again should be safe (no-op)
    sensor.start()
    assert sensor.is_started()

    sensor.stop()
    assert not sensor.is_started()


def test_remote_sensor_multiple_stop():
    """Test that calling stop() multiple times is safe."""
    sensor = RemoteSensor(port=0)

    sensor.start()
    sensor.stop()
    assert not sensor.is_started()

    # Calling stop again should be safe (no-op)
    sensor.stop()
    assert not sensor.is_started()
