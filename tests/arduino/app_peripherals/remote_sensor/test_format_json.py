# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
JSON format tests for RemoteSensor.

Tests JSON data format with text frames, binary frames, and complex structures.
"""

import time
import json
import asyncio
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from test_basic import assert_all_dict


def test_json_format_text_frame():
    """Test JSON format with text frame (string)."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8781, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_json_text():
        uri = "ws://127.0.0.1:8781"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send JSON as text frame (str)
            test_data = {"temperature": 22.5, "humidity": 60.0, "pressure": 1013.25}
            await websocket.send(json.dumps(test_data))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_json_text())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["temperature"] == 22.5
        assert received_data[0]["humidity"] == 60.0
        assert received_data[0]["pressure"] == 1013.25
    finally:
        sensor.stop()


def test_json_format_binary_frame():
    """Test JSON format with binary frame (bytes)."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8782, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_json_binary():
        uri = "ws://127.0.0.1:8782"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send JSON as binary frame (bytes)
            test_data = {"sensor": "test", "value": 42}
            await websocket.send(json.dumps(test_data).encode("utf-8"))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_json_binary())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["sensor"] == "test"
        assert received_data[0]["value"] == 42
    finally:
        sensor.stop()


def test_json_format_complex_structure():
    """Test JSON format with nested data structure."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8793, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_complex_json():
        uri = "ws://127.0.0.1:8793"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            test_data = {"device": "sensor_01", "readings": {"temperature": 22.5, "humidity": 60.0}, "metadata": ["tag1", "tag2", "tag3"]}
            await websocket.send(json.dumps(test_data))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_complex_json())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["device"] == "sensor_01"
        assert received_data[0]["readings"]["temperature"] == 22.5
        assert received_data[0]["metadata"] == ["tag1", "tag2", "tag3"]
    finally:
        sensor.stop()


def test_json_format_array():
    """Test JSON format with array as root element not supported."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8800, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_json_array():
        uri = "ws://127.0.0.1:8800"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            test_data = [1, 2, 3, 4, 5]
            await websocket.send(json.dumps(test_data))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_json_array())
        assert len(received_data) == 0
    finally:
        sensor.stop()


def test_json_format_null_values():
    """Test JSON format with null values."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8801, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_json_with_null():
        uri = "ws://127.0.0.1:8801"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            test_data = {"value": None, "status": "ok"}
            await websocket.send(json.dumps(test_data))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_json_with_null())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["value"] is None
        assert received_data[0]["status"] == "ok"
    finally:
        sensor.stop()


def test_json_format_boolean_values():
    """Test JSON format with boolean values."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8802, data_format="json")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_json_booleans():
        uri = "ws://127.0.0.1:8802"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            test_data = {"enabled": True, "active": False}
            await websocket.send(json.dumps(test_data))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_json_booleans())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["enabled"] is True
        assert received_data[0]["active"] is False
    finally:
        sensor.stop()
