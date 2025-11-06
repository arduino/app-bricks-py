# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Connection and client management tests for RemoteSensor.

Tests callback functionality, single client limitation, multiple messages, and welcome messages.
"""

import time
import json
import asyncio
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from test_basic import assert_all_dict


def test_on_datapoint_callback():
    """Test that the on_datapoint callback is called with received data."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8771)
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)  # Give server time to start

    # Send data from a client
    async def send_data():
        uri = f"ws://127.0.0.1:8771"
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            await websocket.recv()

            # Send telemetry data
            test_data = {"temperature": 22.5, "humidity": 60.0}
            await websocket.send(json.dumps(test_data))

            # Give time for callback to be called
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_data())

        # Verify callback was called
        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["temperature"] == 22.5
        assert received_data[0]["humidity"] == 60.0
    finally:
        sensor.stop()


def test_single_client_limitation():
    """Test that only one client can connect at a time."""
    sensor = RemoteSensor(port=8772)
    sensor.start()
    time.sleep(0.5)  # Give server time to start

    async def test_connections():
        uri = f"ws://127.0.0.1:8772"

        # First client connects
        async with websockets.connect(uri) as ws1:
            # Receive welcome message
            welcome = await ws1.recv()
            welcome_data = json.loads(welcome)
            assert welcome_data["status"] == "connected"

            # Second client tries to connect
            try:
                async with websockets.connect(uri) as ws2:
                    # Should receive rejection message
                    rejection = await ws2.recv()
                    rejection_data = json.loads(rejection)
                    assert rejection_data["error"] == "Server busy"

                    # Connection should close
                    await asyncio.sleep(0.1)
            except websockets.exceptions.ConnectionClosedOK:
                # Expected - server closed the connection
                pass

    try:
        asyncio.run(test_connections())
    finally:
        sensor.stop()


def test_multiple_messages():
    """Test that multiple messages from the same client are all received."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8773)
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)  # Give server time to start

    async def send_multiple():
        uri = f"ws://127.0.0.1:8773"
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            await websocket.recv()

            # Send multiple messages
            for i in range(5):
                data = {"sensor_id": i, "value": i * 10}
                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.1)

            # Give time for callbacks to be processed
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_multiple())

        # Verify all messages were received
        assert_all_dict(received_data)
        assert len(received_data) == 5
        for i in range(5):
            assert received_data[i]["sensor_id"] == i
            assert received_data[i]["value"] == i * 10
    finally:
        sensor.stop()


def test_callback_without_start():
    """Test that setting callback before start works."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8774)

    # Set callback before starting
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_data():
        uri = "ws://127.0.0.1:8774"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()
            await websocket.send(json.dumps({"test": "data"}))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_data())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["test"] == "data"
    finally:
        sensor.stop()


def test_welcome_message_content():
    """Test that welcome message contains expected fields."""
    received_welcome = []

    sensor = RemoteSensor(port=8775, data_format="json")
    sensor.start()
    time.sleep(0.5)

    async def check_welcome():
        uri = "ws://127.0.0.1:8775"
        async with websockets.connect(uri) as websocket:
            welcome = await websocket.recv()
            received_welcome.append(json.loads(welcome))

    try:
        asyncio.run(check_welcome())

        assert len(received_welcome) == 1
        assert received_welcome[0]["status"] == "connected"
        assert received_welcome[0]["data_format"] == "json"
        assert "message" in received_welcome[0]
    finally:
        sensor.stop()


def test_client_reconnection():
    """Test that a client can reconnect after disconnecting."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8776)
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def connect_disconnect_reconnect():
        uri = "ws://127.0.0.1:8776"

        # First connection
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Welcome
            await ws.send(json.dumps({"msg": 1}))
            await asyncio.sleep(0.3)

        # Give server time to clean up
        await asyncio.sleep(0.5)

        # Second connection (reconnect)
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Welcome
            await ws.send(json.dumps({"msg": 2}))
            await asyncio.sleep(0.3)

    try:
        asyncio.run(connect_disconnect_reconnect())

        assert_all_dict(received_data)
        assert len(received_data) == 2
        assert received_data[0]["msg"] == 1
        assert received_data[1]["msg"] == 2
    finally:
        sensor.stop()
