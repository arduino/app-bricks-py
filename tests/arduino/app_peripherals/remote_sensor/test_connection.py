# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
Connection and client management tests for RemoteSensor.

Tests callback functionality, single client limitation, multiple messages, and welcome messages.
"""

import pytest
import json
import asyncio
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from arduino.app_internal.core.peripherals import BPPCodec


@pytest.fixture
def codec() -> BPPCodec:
    """Fixture to provide a BPPCodec instance."""
    return BPPCodec()


@pytest.mark.asyncio
async def test_on_datapoint_callback(codec):
    """Test that the on_datapoint callback is called with received data."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        payload = json.loads(data.decode())
        received_data.append(payload)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        # Receive welcome message
        await ws.recv()

        # Send telemetry data
        data = {"temperature": -5.0, "humidity": 60.0}
        encoded = codec.encode(json.dumps(data).encode())
        await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()
    
    # Verify callback was called
    assert len(received_data) == 1
    assert "temperature" in received_data[0]
    assert received_data[0]["temperature"] == -5.0
    assert "humidity" in received_data[0]
    assert received_data[0]["humidity"] == 60.0


@pytest.mark.asyncio
async def test_single_client_limitation(codec):
    """Test that only one client can connect at a time."""
    sensor = RemoteSensor(port=0)
    sensor.start()

    # First client connects
    async with websockets.connect(sensor.url) as ws1:
        # Receive welcome message
        welcome = await ws1.recv()
        welcome_decoded = codec.decode(welcome)
        welcome_data = json.loads(welcome_decoded)
        assert welcome_data["status"] == "connected"

        # Second client tries to connect
        try:
            async with websockets.connect(sensor.url) as ws2:
                # Should receive rejection message
                rejection = await ws2.recv()
                rejection_decoded = codec.decode(rejection)
                rejection_data = json.loads(rejection_decoded)
                assert "error" in rejection_data

        except websockets.exceptions.ConnectionClosedOK:
            # Expected - server closed the connection
            pass
    
    sensor.stop()


@pytest.mark.asyncio
async def test_multiple_messages(codec):
    """Test that multiple messages from the same client are all received."""
    n_messages = 5
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        payload = json.loads(data.decode())
        received_data.append(payload)
        if len(received_data) == n_messages:
            loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        # Receive welcome message
        await ws.recv()

        # Send multiple messages
        for i in range(n_messages):
            data = {"sensor_id": i, "value": i * 10}
            encoded = codec.encode(json.dumps(data).encode())
            await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)

    sensor.stop()

    # Verify all messages were received
    assert len(received_data) == n_messages
    for i in range(n_messages):
        assert "sensor_id" in received_data[i]
        assert received_data[i]["sensor_id"] == i
        assert "value" in received_data[i]
        assert received_data[i]["value"] == i * 10


@pytest.mark.asyncio
async def test_welcome_message_content(codec):
    """Test that welcome message contains expected fields."""
    received_welcome = []
    test_done = asyncio.Event()

    sensor = RemoteSensor(port=0)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        welcome = await ws.recv()
        welcome_decoded = codec.decode(welcome)
        received_welcome.append(json.loads(welcome_decoded))
        test_done.set()

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_welcome) == 1
    assert "status" in received_welcome[0]
    assert received_welcome[0]["status"] == "connected"
    assert "security_mode" in received_welcome[0]
    assert received_welcome[0]["security_mode"] == "none"
    assert "message" in received_welcome[0]


@pytest.mark.asyncio
async def test_client_reconnection(codec):
    """Test that a client can reconnect after disconnecting."""
    received_data = []
    loop = asyncio.get_running_loop()
    task_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        payload = json.loads(data.decode())
        received_data.append(payload)
        if len(received_data) == 2:
            loop.call_soon_threadsafe(task_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    # First connection
    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        encoded = codec.encode(json.dumps({"msg": 1}).encode())
        await ws.send(encoded)

    # Give server time to clean up
    await asyncio.sleep(0.1)

    # Second connection (reconnect)
    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        encoded = codec.encode(json.dumps({"msg": 2}).encode())
        await ws.send(encoded)

    await asyncio.wait_for(task_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 2
    assert "msg" in received_data[0]
    assert received_data[0]["msg"] == 1
    assert "msg" in received_data[1]
    assert received_data[1]["msg"] == 2
