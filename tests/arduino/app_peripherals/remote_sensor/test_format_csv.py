# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
CSV format tests for RemoteSensor.

Tests CSV data format with single/multiple lines, different line endings, and binary frames.
"""

import time
import asyncio
import websockets

from arduino.app_peripherals.remote_sensor import RemoteSensor
from test_basic import assert_all_dict


def test_csv_format_single_line():
    """Test CSV format with single line."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8783, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_single():
        uri = "ws://127.0.0.1:8783"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send single CSV line
            csv_line = "temperature,22.5,humidity,60.0"
            await websocket.send(csv_line)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_single())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["csv"] == "temperature,22.5,humidity,60.0"
    finally:
        sensor.stop()


def test_csv_format_multiple_lines():
    """Test CSV format with multiple lines in one message."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8784, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_multiple():
        uri = "ws://127.0.0.1:8784"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send multiple CSV lines in one message
            csv_lines = "line1,value1\nline2,value2\nline3,value3"
            await websocket.send(csv_lines)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_multiple())

        # Should receive 3 callbacks, one per line
        assert_all_dict(received_data)
        assert len(received_data) == 3
        assert received_data[0]["csv"] == "line1,value1"
        assert received_data[1]["csv"] == "line2,value2"
        assert received_data[2]["csv"] == "line3,value3"
    finally:
        sensor.stop()


def test_csv_format_with_crlf():
    """Test CSV format with CRLF line endings."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8785, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_crlf():
        uri = "ws://127.0.0.1:8785"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send CSV with CRLF line endings
            csv_lines = "sensor1,100\r\nsensor2,200\r\nsensor3,300"
            await websocket.send(csv_lines)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_crlf())

        assert_all_dict(received_data)
        assert len(received_data) == 3
        assert received_data[0]["csv"] == "sensor1,100"
        assert received_data[1]["csv"] == "sensor2,200"
        assert received_data[2]["csv"] == "sensor3,300"
    finally:
        sensor.stop()


def test_csv_format_binary_frame():
    """Test CSV format sent as binary frame."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8786, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_binary():
        uri = "ws://127.0.0.1:8786"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()  # Welcome message

            # Send CSV as binary frame
            csv_line = "temp,25.5,hum,55.0"
            await websocket.send(csv_line.encode("utf-8"))
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_binary())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["csv"] == "temp,25.5,hum,55.0"
    finally:
        sensor.stop()


def test_csv_format_empty_lines():
    """Test CSV format with empty lines (should be ignored)."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8794, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_with_empty():
        uri = "ws://127.0.0.1:8794"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send CSV with empty lines
            csv_lines = "line1,val1\n\n\nline2,val2\n\n"
            await websocket.send(csv_lines)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_with_empty())

        # Should only receive 2 callbacks (empty lines ignored)
        assert_all_dict(received_data)
        assert len(received_data) == 2
        assert received_data[0]["csv"] == "line1,val1"
        assert received_data[1]["csv"] == "line2,val2"
    finally:
        sensor.stop()


def test_csv_format_tab_separator():
    """Test CSV format with tab separator."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8803, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_tabs():
        uri = "ws://127.0.0.1:8803"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send CSV with tab separators
            csv_line = "col1\tcol2\tcol3\tvalue"
            await websocket.send(csv_line)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_tabs())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["csv"] == "col1\tcol2\tcol3\tvalue"
    finally:
        sensor.stop()


def test_csv_format_space_separator():
    """Test CSV format with space separator."""
    received_data = []

    def callback(data):
        received_data.append(data)

    sensor = RemoteSensor(port=8804, data_format="csv")
    sensor.on_datapoint(callback)
    sensor.start()
    time.sleep(0.5)

    async def send_csv_spaces():
        uri = "ws://127.0.0.1:8804"
        async with websockets.connect(uri) as websocket:
            await websocket.recv()

            # Send CSV with space separators
            csv_line = "item1 item2 item3 100"
            await websocket.send(csv_line)
            await asyncio.sleep(0.5)

    try:
        asyncio.run(send_csv_spaces())

        assert_all_dict(received_data)
        assert len(received_data) == 1
        assert received_data[0]["csv"] == "item1 item2 item3 100"
    finally:
        sensor.stop()
