# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
Binary format tests for RemoteSensor.

Tests binary data handling, numpy arrays, multi-byte data, and byte order interpretation.
"""

import time
import asyncio
import numpy as np
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from test_basic import assert_all_dict


def test_binary_format_basic():
    """Test binary format."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8787, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_raw():
        uri = "ws://127.0.0.1:8787"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send binary data
            raw_data = bytes([0, 1, 2, 3, 4, 5, 255, 254, 253])
            await websocket.send(raw_data)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_raw())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert "binary" in received_data[0]

        raw_array = received_data[0]["binary"]
        assert isinstance(raw_array, np.ndarray)
        assert raw_array.dtype == np.uint8
        assert len(raw_array) == 9
        assert raw_array[0] == 0
        assert raw_array[4] == 4
        assert raw_array[6] == 255
        assert raw_array[8] == 253
    finally:
        sensor.stop()


def test_binary_format_numpy_array():
    """Test raw format with numpy array data."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8788, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_numpy():
        uri = "ws://127.0.0.1:8788"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send numpy array as bytes
            original_data = np.array([100, 200, 50, 75, 125], dtype=np.uint8)
            await websocket.send(original_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_numpy())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]

        expected = np.array([100, 200, 50, 75, 125], dtype=np.uint8)
        assert np.array_equal(raw_array, expected)
    finally:
        sensor.stop()


def test_binary_format_multibyte_data():
    """Test raw format with multi-byte integer data."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8789, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_multibyte():
        uri = "ws://127.0.0.1:8789"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send int16 array as bytes
            int16_data = np.array([1000, 2000, 3000, 4000], dtype=np.int16)
            await websocket.send(int16_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_multibyte())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]

        # Server receives as uint8, we need to reinterpret
        assert len(raw_array) == 8  # 4 int16 values = 8 bytes

        # Reinterpret as int16
        reconstructed = np.frombuffer(raw_array, dtype=np.int16)
        expected = np.array([1000, 2000, 3000, 4000], dtype=np.int16)
        assert np.array_equal(reconstructed, expected)
    finally:
        sensor.stop()


def test_binary_format_float_data():
    """Test raw format with float32 data."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8795, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_float():
        uri = "ws://127.0.0.1:8795"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send float32 array as bytes
            float_data = np.array([22.5, 60.0, 1013.25], dtype=np.float32)
            await websocket.send(float_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_float())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]

        # Reinterpret as float32
        reconstructed = np.frombuffer(raw_array, dtype=np.float32)
        expected = np.array([22.5, 60.0, 1013.25], dtype=np.float32)
        assert np.allclose(reconstructed, expected)
    finally:
        sensor.stop()


def test_binary_format_little_endian():
    """Test raw format with explicit little-endian interpretation."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8796, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_le_data():
        uri = "ws://127.0.0.1:8796"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send little-endian int16 data
            int16_data = np.array([256, 512, 1024], dtype="<i2")
            await websocket.send(int16_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_le_data())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]

        # Reinterpret as little-endian int16
        reconstructed = np.frombuffer(raw_array, dtype="<i2")
        expected = np.array([256, 512, 1024], dtype="<i2")
        assert np.array_equal(reconstructed, expected)
    finally:
        sensor.stop()


def test_binary_format_big_endian():
    """Test raw format with explicit big-endian interpretation."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8797, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_be_data():
        uri = "ws://127.0.0.1:8797"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send big-endian int16 data
            int16_data = np.array([256, 512, 1024], dtype=">i2")
            await websocket.send(int16_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_be_data())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]

        # Reinterpret as big-endian int16
        reconstructed = np.frombuffer(raw_array, dtype=">i2")
        expected = np.array([256, 512, 1024], dtype=">i2")
        assert np.array_equal(reconstructed, expected)
    finally:
        sensor.stop()


def test_binary_format_empty_data():
    """Test raw format with empty data."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8798, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_empty():
        uri = "ws://127.0.0.1:8798"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send empty bytes
            await websocket.send(b"")
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_empty())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]
        assert len(raw_array) == 0
    finally:
        sensor.stop()


def test_binary_format_large_data():
    """Test raw format with large data block."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8799, data_format="binary")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_large():
        uri = "ws://127.0.0.1:8799"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send 10KB of data
            large_data = np.random.randint(0, 256, size=10000, dtype=np.uint8)
            await websocket.send(large_data.tobytes())
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_large())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        raw_array = received_data[0]["binary"]
        assert len(raw_array) == 10000
    finally:
        sensor.stop()
