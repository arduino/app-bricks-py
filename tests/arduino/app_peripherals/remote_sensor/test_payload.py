# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""
Binary format tests for RemoteSensor.

Tests binary data handling, numpy arrays, multi-byte data, and byte order interpretation.
"""

import pytest
import asyncio
import numpy as np
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from arduino.app_internal.core.peripherals import BPPCodec


@pytest.fixture
def codec() -> BPPCodec:
    """Fixture to provide a BPPCodec instance."""
    return BPPCodec()


@pytest.mark.asyncio
async def test_binary_multibyte_data(codec):
    """Test raw format with multi-byte integer data."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome message

        int16_data = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
        encoded = codec.encode(int16_data.tobytes())
        await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    assert len(raw_array) == 8  # 4 int16 values = 8 bytes
    reconstructed = np.frombuffer(raw_array, dtype=np.int16)
    expected = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
    assert np.array_equal(reconstructed, expected)


@pytest.mark.asyncio
async def test_binary_format_float_data(codec):
    """Test raw format with float32 data."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()

        float_data = np.array([22.5, -60.0, 1013.25], dtype=np.float32)
        encoded = codec.encode(float_data.tobytes())
        await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    reconstructed = np.frombuffer(raw_array, dtype=np.float32)
    expected = np.array([22.5, -60.0, 1013.25], dtype=np.float32)
    assert np.allclose(reconstructed, expected)


@pytest.mark.asyncio
async def test_binary_format_little_endian(codec):
    """Test raw format with explicit little-endian interpretation."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()

        int16_data = np.array([256, -512, 1024], dtype="<i2")
        encoded = codec.encode(int16_data.tobytes())
        await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    reconstructed = np.frombuffer(raw_array, dtype="<i2")
    expected = np.array([256, -512, 1024], dtype="<i2")
    assert np.array_equal(reconstructed, expected)


@pytest.mark.asyncio
async def test_binary_format_big_endian(codec):
    """Test raw format with explicit big-endian interpretation."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as websocket:
        await websocket.recv()

        int16_data = np.array([256, 512, 1024], dtype=">i2")
        encoded = codec.encode(int16_data.tobytes())
        await websocket.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    reconstructed = np.frombuffer(raw_array, dtype=">i2")
    expected = np.array([256, 512, 1024], dtype=">i2")
    assert np.array_equal(reconstructed, expected)


@pytest.mark.asyncio
async def test_binary_format_empty_data(codec):
    """Test raw format with empty data."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as websocket:
        await websocket.recv()

        encoded = codec.encode(b"")
        await websocket.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    assert len(raw_array) == 0


@pytest.mark.asyncio
async def test_binary_format_large_data(codec):
    """Test raw format with large data block."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        received_data.append(data)
        loop.call_soon_threadsafe(test_done.set)

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()
        large_data = np.random.randint(0, 256, size=10000, dtype=np.uint8)
        encoded = codec.encode(large_data.tobytes())
        await ws.send(encoded)

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 1
    raw_array = received_data[0]
    assert len(raw_array) == 10000
