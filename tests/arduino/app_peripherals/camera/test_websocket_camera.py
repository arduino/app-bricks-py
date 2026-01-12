# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import asyncio
import base64
import json
import numpy as np
import cv2
import websockets

from arduino.app_internal.core.peripherals.bpp_codec import BPPCodec
from arduino.app_peripherals.camera import WebSocketCamera


@pytest.fixture
def codec() -> BPPCodec:
    """Create codec for encoding/decoding in tests."""
    return BPPCodec()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def encoded_frame_binary(codec, sample_frame) -> bytes:
    """Encode frame as binary."""
    _, buffer = cv2.imencode(".jpg", sample_frame)
    return codec.encode(buffer.tobytes())


@pytest.fixture
def encoded_frame_string(encoded_frame_binary) -> str:
    """Encode frame as base64 string."""
    return base64.b64encode(encoded_frame_binary).decode()


def test_websocket_camera_init_default():
    """Test WebSocketCamera initialization with default parameters."""
    camera = WebSocketCamera()
    assert camera.url == "ws://0.0.0.0:8080"
    assert camera.port == 8080
    assert camera.timeout == 3
    assert camera.resolution == (640, 480)
    assert camera.fps == 10
    assert camera.status == "disconnected"


def test_websocket_camera_init_custom():
    """Test WebSocketCamera initialization with custom parameters."""
    camera = WebSocketCamera(port=9090, timeout=30, resolution=(1920, 1080), fps=30)
    assert camera.url == "ws://0.0.0.0:9090"  # No env var is set, so uses default host
    assert camera.port == 9090
    assert camera.timeout == 30
    assert camera.resolution == (1920, 1080)
    assert camera.fps == 30
    assert camera.status == "disconnected"


def test_websocket_camera_start_stop():
    """Test start/stop WebSocket camera server."""
    camera = WebSocketCamera(port=0)
    assert not camera.is_started()

    try:
        camera.start()
    except Exception:
        pytest.fail("Camera start failed")

    assert camera.is_started()
    # Starting does not coincide with being connected in case of WebSocketCamera
    # as that depends on client activity
    assert camera.status == "disconnected"

    try:
        camera.stop()
    except Exception:
        pytest.fail("Camera stop failed")

    assert not camera.is_started()
    assert camera.status == "disconnected"


def test_websocket_camera_handle_binary_message(sample_frame, encoded_frame_binary):
    """Test parsing binary frame message."""
    camera = WebSocketCamera()

    frame = camera._parse_message(encoded_frame_binary)

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == sample_frame.shape


def test_websocket_camera_handle_base64_message(sample_frame, encoded_frame_string):
    """Test parsing binary message received as string using base64 encoding."""
    camera = WebSocketCamera()

    frame = camera._parse_message(encoded_frame_string)

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == sample_frame.shape


def test_websocket_camera_handle_message_invalid():
    """Test parsing invalid message."""
    camera = WebSocketCamera()

    frame = camera._parse_message("invalid base64 string")

    assert frame is None


def test_websocket_camera_read_frame_empty_queue():
    """Test reading frame when queue is empty."""
    with WebSocketCamera(port=0) as camera:
        frame = camera.capture()
        assert frame is None


@pytest.mark.asyncio
async def test_websocket_camera_capture_frame(encoded_frame_binary):
    """Test capturing frame from WebSocket camera."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Skip welcome message
            await ws.recv()

            await ws.send(encoded_frame_binary)

            await asyncio.sleep(0.1)

            frame = camera.capture()

            assert frame is not None
            assert isinstance(frame, np.ndarray)


@pytest.mark.asyncio
async def test_websocket_camera_single_client(codec):
    """Test WebSocket server accepts only one client at a time."""
    camera = WebSocketCamera(port=0)
    camera.start()

    try:
        # Connect first client
        async with websockets.connect(camera.url) as ws1:
            # First client should receive welcome message
            welcome = await ws1.recv()
            decoded_welcome = codec.decode(welcome)
            welcome_message = json.loads(decoded_welcome)
            assert welcome_message["status"] == "connected"

            # Try to connect second client while first is connected
            try:
                async with websockets.connect(camera.url) as ws2:
                    # Second client should receive rejection message
                    rejection = await asyncio.wait_for(ws2.recv(), timeout=1.0)
                    decoded_rejection = codec.decode(rejection)
                    rejection_message = json.loads(decoded_rejection)
                    assert "error" in rejection_message
            except websockets.exceptions.ConnectionClosed:
                # Connection closed immediately - also acceptable
                pass
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_welcome_message(codec):
    """Test that welcome message is sent to connected client."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Should receive welcome message
            welcome = await asyncio.wait_for(ws.recv(), timeout=1.0)
            decoded_welcome = codec.decode(welcome)
            welcome_message = json.loads(decoded_welcome)
            assert "message" in welcome_message
            assert welcome_message["status"] == "connected"
            assert tuple(welcome_message["resolution"]) == camera.resolution
            assert welcome_message["fps"] == camera.fps
            assert welcome_message["security_mode"] == camera.security_mode


@pytest.mark.asyncio
async def test_websocket_camera_receives_frames(encoded_frame_binary):
    """Test that server receives and queues frames from client."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Skip welcome message
            await ws.recv()

            # Send a frame
            await ws.send(encoded_frame_binary)

            # Give time for frame to be processed
            await asyncio.sleep(0.2)

            # Frame should be in queue
            assert camera.capture() is not None


@pytest.mark.asyncio
async def test_websocket_camera_disconnects_client_on_stop(codec):
    """Test that connected client is disconnected when camera stops."""
    camera = WebSocketCamera(port=0)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            # Client connected, receive welcome message
            welcome = await ws.recv()
            decoded_welcome = codec.decode(welcome)
            welcome_message = json.loads(decoded_welcome)
            assert welcome_message["status"] == "connected"

            # Stop the camera (runs in background thread via to_thread)
            await asyncio.to_thread(camera.stop)

            with pytest.raises(websockets.exceptions.ConnectionClosed):
                # Keep receiving until connection is closed
                while True:
                    goodbye = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    decoded_goodbye = codec.decode(goodbye)
                    goodbye_message = json.loads(decoded_goodbye)
                    if goodbye_message.get("status") == "disconnecting":
                        # Got goodbye message, connection should close soon
                        continue
    except websockets.exceptions.ConnectionClosed:
        # Connection was closed, which is expected
        pass

    assert not camera.is_started()


def test_websocket_camera_stop_without_client():
    """Test stopping server when no client is connected."""
    camera = WebSocketCamera(port=0)
    camera.start()

    # Stopping without any connected client should not raise an exception
    camera.stop()

    assert not camera.is_started()


@pytest.mark.asyncio
async def test_websocket_camera_backpressure(codec):
    """Test that old frames are dropped when new frames arrive faster than they're consumed."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            await ws.recv()  # Skip welcome message

            _, buffer1 = cv2.imencode(".jpg", np.ones((480, 640, 3), dtype=np.uint8) * 1)
            _, buffer2 = cv2.imencode(".jpg", np.ones((480, 640, 3), dtype=np.uint8) * 2)
            _, buffer3 = cv2.imencode(".jpg", np.ones((480, 640, 3), dtype=np.uint8) * 3)

            await ws.send(codec.encode(buffer1.tobytes()))
            await ws.send(codec.encode(buffer2.tobytes()))
            await ws.send(codec.encode(buffer3.tobytes()))

            await asyncio.sleep(0.1)

            frame = camera.capture()
            assert frame is not None

            mean_value = np.mean(frame)
            assert mean_value == 3  # Only the last one should be kept


def test_websocket_camera_with_adjustments(sample_frame):
    """Test WebSocket camera with frame adjustments."""

    def adjustment(frame):
        return frame + 50

    camera = WebSocketCamera(adjustments=adjustment)
    camera._frame_queue.put(sample_frame)
    camera._is_started = True

    # Capture uses adjustments
    frame = camera.capture()
    assert frame is not None

    # The adjustment is applied in capture()
    expected = sample_frame + 50
    assert np.array_equal(frame, expected)


@pytest.mark.asyncio
async def test_websocket_camera_client_events():
    """
    Test that WebSocket camera emits connection and disconnection events depending on client activity.
    """
    events = []
    main_loop = asyncio.get_running_loop()

    connected = asyncio.Event()
    disconnected = asyncio.Event()

    def event_listener(event_type, data):
        if event_type == "connected":
            main_loop.call_soon_threadsafe(connected.set)
            assert "client_address" in data
            assert "client_name" in data
            assert data["client_name"] == "test_client"
        if event_type == "disconnected":
            main_loop.call_soon_threadsafe(disconnected.set)
            assert "client_address" in data
            assert "client_name" in data
            assert data["client_name"] == "test_client"
        events.append((event_type, data))

    camera = WebSocketCamera(port=0)
    camera.on_status_changed(event_listener)
    camera.start()

    # This should emit connection and disconnection events
    async def client_task():
        async with websockets.connect(camera.url + "?client_name=test_client"):
            pass

    # Run client concurrently to properly test event handling
    client = asyncio.create_task(client_task())

    try:
        await asyncio.wait_for(connected.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Connection event was not emitted within timeout")
    try:
        await asyncio.wait_for(disconnected.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Disconnection event was not emitted within timeout")

    await client  # Ensure client task is finished and check for errors

    # The events list is modified from another thread, so a brief sleep
    # helps ensure the main thread sees the appended items before asserting.
    await asyncio.sleep(0.1)

    assert len(events) == 2
    assert "connected" in events[0][0]
    assert "disconnected" in events[1][0]

    camera.stop()  # This should not emit a disconnection

    await asyncio.sleep(0.1)

    # Check that stop() didn't emit additional events
    assert len(events) == 2
    assert "connected" in events[0][0]
    assert "disconnected" in events[1][0]


@pytest.mark.asyncio
async def test_websocket_camera_start_stop_events():
    """
    Test that WebSocket camera doesn't emit connection and disconnection events when started and
    stopped without any client connections.
    """
    events = []

    def event_listener(event_type, data):
        events.append((event_type, data))

    camera = WebSocketCamera(port=0)
    camera.on_status_changed(event_listener)
    camera.start()

    await asyncio.sleep(0.1)

    camera.stop()  # This should not emit a disconnection

    await asyncio.sleep(0.1)

    # Check that connection and disconnection events weren't emitted
    assert len(events) == 0


@pytest.mark.asyncio
async def test_websocket_camera_stop_event():
    """
    Test that WebSocket camera emits a disconnection event when stopped if
    there's an active client connection.
    """
    events = []

    connected = asyncio.Event()

    def event_listener(event_type, data):
        if event_type == "connected":
            connected.set()
        events.append((event_type, data))

    camera = WebSocketCamera(port=0, timeout=1)  # Reduced timeout for faster stop() call
    camera.on_status_changed(event_listener)
    camera.start()

    can_close = asyncio.Event()

    # This should emit a connection event but no disconnection event
    async def client_task():
        async with websockets.connect(camera.url):
            pass
        await can_close.wait()

    asyncio.create_task(client_task())

    try:
        await asyncio.wait_for(connected.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Connection event was not emitted within timeout")

    camera.stop()  # This should emit a disconnection
    can_close.set()

    # Check that connection and disconnection events weren't emitted
    assert len(events) == 2
    assert "connected" in events[0][0]
    assert "disconnected" in events[1][0]
