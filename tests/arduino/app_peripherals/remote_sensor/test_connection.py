# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
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
async def test_plaintext_connectivity(codec):
    """Test that a client can connect to the server on a plaintext connection."""
    loop = asyncio.get_running_loop()
    connected = asyncio.Event()
    disconnected = asyncio.Event()

    def status_callback(status: str, info: dict):
        if status == "connected":
            loop.call_soon_threadsafe(connected.set)
            assert info.get("client_address") is not None
        elif status == "disconnected":
            loop.call_soon_threadsafe(disconnected.set)
            assert info.get("client_address") is not None

    sensor = RemoteSensor(port=0)
    sensor.on_status_changed(status_callback)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        await ws.send(codec.encode(b"{}"))

    await asyncio.wait_for(connected.wait(), timeout=2)

    sensor.stop()
    await asyncio.wait_for(disconnected.wait(), timeout=2)

    assert connected.is_set()
    assert disconnected.is_set()


@pytest.mark.asyncio
async def test_tls_connectivity(codec):
    """Test that a client can connect to the server on a TLS connection."""
    loop = asyncio.get_running_loop()
    connected = asyncio.Event()
    disconnected = asyncio.Event()

    def status_callback(status: str, info: dict):
        if status == "connected":
            loop.call_soon_threadsafe(connected.set)
            assert info.get("client_address") is not None
        elif status == "disconnected":
            loop.call_soon_threadsafe(disconnected.set)
            assert info.get("client_address") is not None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        sensor = RemoteSensor(port=0, use_tls=True, certs_dir_path=tmp_dir)
        sensor.on_status_changed(status_callback)
        sensor.start()

        # Disable cert verification as we're using a self-signed one
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with websockets.connect(sensor.url, ssl=ssl_context) as ws:
            await ws.recv()  # Welcome
            await ws.send(codec.encode(b"{}"))

        await asyncio.wait_for(connected.wait(), timeout=2)

        sensor.stop()
        await asyncio.wait_for(disconnected.wait(), timeout=2)

    assert connected.is_set()
    assert disconnected.is_set()


TEST_SECRET = "test-secret-key"
SECURITY_MODES = [(None, False), (TEST_SECRET, False), (TEST_SECRET, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("secret, encrypt", SECURITY_MODES)
async def test_connected_at_first_valid_message(secret, encrypt):
    """Test that the client is connected when its first valid message arrives, not when the socket opens."""
    loop = asyncio.get_running_loop()
    connected = asyncio.Event()
    codec = BPPCodec(secret or "", encrypt)

    sensor = RemoteSensor(port=0, secret=secret, encrypt=encrypt)
    sensor.on_status_changed(lambda status, info: loop.call_soon_threadsafe(connected.set) if status == "connected" else None)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        assert sensor.status == "disconnected"

        await ws.send(codec.encode(b"{}"))
        await asyncio.wait_for(connected.wait(), timeout=2)
        assert sensor.status == "streaming"

    sensor.stop()


@pytest.mark.asyncio
async def test_stream_is_paused_when_idle_and_resumes(codec):
    """Test that the stream is paused when no datapoint arrives for the idle timeout and resumes at the next one."""
    loop = asyncio.get_running_loop()
    events = []
    reached = {"paused": asyncio.Event(), "streaming": asyncio.Event()}

    def status_callback(status: str, info: dict):
        events.append((status, info.get("client_name")))
        if status in reached:
            loop.call_soon_threadsafe(reached[status].set)

    sensor = RemoteSensor(port=0, idle_timeout=0.05)
    sensor.on_status_changed(status_callback)
    sensor.start()

    async with websockets.connect(sensor.url + "?client_name=phone") as ws:
        await ws.recv()  # Welcome

        await ws.send(codec.encode(b"{}"))
        await asyncio.wait_for(reached["paused"].wait(), timeout=2)

        reached["streaming"].clear()
        await ws.send(codec.encode(b"{}"))
        await asyncio.wait_for(reached["streaming"].wait(), timeout=2)

    sensor.stop()

    assert [status for status, _ in events[:4]] == ["connected", "streaming", "paused", "streaming"]
    assert events[-1][0] == "disconnected"
    assert all(client_name == "phone" for _, client_name in events)


@pytest.mark.asyncio
async def test_stream_is_not_paused_while_datapoints_arrive(codec):
    """Test that a client sending more often than the idle timeout is never paused."""
    events = []

    sensor = RemoteSensor(port=0, idle_timeout=0.2)
    sensor.on_status_changed(lambda status, info: events.append(status))
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome

        # Keep sending for twice the idle timeout
        for _ in range(8):
            await ws.send(codec.encode(b"{}"))
            await asyncio.sleep(0.05)

    sensor.stop()

    assert events == ["connected", "streaming", "disconnected"]


@pytest.mark.asyncio
@pytest.mark.parametrize("secret, encrypt", SECURITY_MODES)
async def test_silent_client_is_not_connected(secret, encrypt):
    """Test that a client that opens the socket and sends nothing causes no status change."""
    events = []

    sensor = RemoteSensor(port=0, secret=secret, encrypt=encrypt)
    sensor.on_status_changed(lambda status, info: events.append(status))
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome

    sensor.stop()

    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypt", [False, True])
async def test_wrong_secret_client_is_not_connected(encrypt):
    """Test that a client sending messages with a wrong secret causes no status change."""
    events = []
    wrong_codec = BPPCodec("wrong-secret", encrypt)

    sensor = RemoteSensor(port=0, secret=TEST_SECRET, encrypt=encrypt)
    sensor.on_status_changed(lambda status, info: events.append(status))
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        await ws.send(wrong_codec.encode(b"{}"))

    sensor.stop()

    assert events == []


@pytest.mark.asyncio
async def test_status_is_reported_after_restart(codec):
    """Test that status changes are still reported after the sensor is stopped and started again."""
    loop = asyncio.get_running_loop()
    connected = asyncio.Event()

    sensor = RemoteSensor(port=0)
    sensor.on_status_changed(lambda status, info: loop.call_soon_threadsafe(connected.set) if status == "connected" else None)
    sensor.start()
    sensor.stop()
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        await ws.recv()  # Welcome
        await ws.send(codec.encode(b"{}"))
        await asyncio.wait_for(connected.wait(), timeout=2)

    sensor.stop()


@pytest.mark.asyncio
async def test_client_reconnection(codec):
    """Test that a client can reconnect after disconnecting."""
    received_data = []
    loop = asyncio.get_running_loop()
    test_done = asyncio.Event()

    def callback(data):
        assert isinstance(data, bytes)
        payload = json.loads(data.decode())
        received_data.append(payload)
        if len(received_data) == 2:
            loop.call_soon_threadsafe(test_done.set)

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

    await asyncio.wait_for(test_done.wait(), timeout=2)
    sensor.stop()

    assert len(received_data) == 2
    assert "msg" in received_data[0]
    assert received_data[0]["msg"] == 1
    assert "msg" in received_data[1]
    assert received_data[1]["msg"] == 2


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
async def test_rejected_client_leaves_active_client_untouched(codec):
    """Test that a rejected client changes neither the name nor the data of the active one."""
    loop = asyncio.get_running_loop()
    received = asyncio.Queue()

    sensor = RemoteSensor(port=0)
    sensor.on_datapoint(lambda data: loop.call_soon_threadsafe(received.put_nowait, data))
    sensor.start()

    async with websockets.connect(sensor.url + "?client_name=first") as first:
        await first.recv()  # Welcome

        async with websockets.connect(sensor.url + "?client_name=second&raw=true") as second:
            rejection = json.loads(await second.recv())
            assert rejection["error"] == "Server busy"

        await first.send(codec.encode(b"datapoint"))
        assert await asyncio.wait_for(received.get(), timeout=2) == b"datapoint"
        assert sensor.name == "first"

    sensor.stop()


@pytest.mark.asyncio
async def test_welcome_goodbye_message_content(codec):
    """Test that welcome and goodbye messages contain expected fields."""
    sensor = RemoteSensor(port=0)
    sensor.start()

    async with websockets.connect(sensor.url) as ws:
        welcome_msg = await ws.recv()  # Welcome message
        welcome_decoded = codec.decode(welcome_msg)
        welcome = json.loads(welcome_decoded)

        assert "status" in welcome
        assert welcome["status"] == "connected"
        assert "message" in welcome
        assert "security_mode" in welcome
        assert welcome["security_mode"] == "none"

        import threading

        threading.Thread(target=sensor.stop, daemon=True).start()

        goodbye_msg = await ws.recv()  # Goodbye message
        goodbye_decoded = codec.decode(goodbye_msg)
        goodbye = json.loads(goodbye_decoded)

        assert "status" in goodbye
        assert goodbye["status"] == "disconnecting"
        assert "message" in goodbye


@pytest.mark.asyncio
async def test_on_single_message(codec):
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
