# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
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
def sample_frame() -> np.ndarray:
    """Create a sample frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def plain_codec() -> BPPCodec:
    """BPPCodec for MODE_NONE (no secret, no encryption)."""
    return BPPCodec("", enable_encryption=False)


@pytest.fixture
def bpp_frame_binary(sample_frame, plain_codec) -> bytes:
    """Encode frame as JPEG bytes wrapped in BPP MODE_NONE."""
    _, buffer = cv2.imencode(".jpg", sample_frame)
    return plain_codec.encode(buffer.tobytes())


@pytest.fixture
def bpp_frame_string(bpp_frame_binary) -> str:
    """Encode BPP-wrapped frame as base64 string."""
    return base64.b64encode(bpp_frame_binary).decode()


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


def test_websocket_camera_encrypt_without_secret_fails():
    """Test that encrypt=True without a secret raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Encryption requires a secret key"):
        WebSocketCamera(encrypt=True)


def test_websocket_camera_empty_string_secret_enables_bpp():
    """Test that secret="" is valid and enables BPP authentication."""
    camera = WebSocketCamera(port=0, secret="")
    assert camera.codec is not None
    assert camera.secret == ""


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


def test_websocket_camera_handle_binary_message(sample_frame, bpp_frame_binary):
    """Test parsing BPP-wrapped binary frame message."""
    camera = WebSocketCamera()

    frame = camera._parse_message(bpp_frame_binary)

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == sample_frame.shape


def test_websocket_camera_handle_base64_message(sample_frame, bpp_frame_string):
    """Test parsing BPP-wrapped message received as base64 string."""
    camera = WebSocketCamera()

    frame = camera._parse_message(bpp_frame_string)

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
async def test_websocket_camera_capture_frame(bpp_frame_binary):
    """Test capturing frame from WebSocket camera (BPP MODE_NONE)."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Skip welcome message
            await ws.recv()

            await ws.send(bpp_frame_binary)

            await asyncio.sleep(0.1)

            frame = camera.capture()

            assert frame is not None
            assert isinstance(frame, np.ndarray)


@pytest.mark.asyncio
async def test_websocket_camera_single_client():
    """Test WebSocket server accepts only one client at a time."""
    codec = BPPCodec("", enable_encryption=False)
    camera = WebSocketCamera(port=0)
    camera.start()

    try:
        # Connect first client
        async with websockets.connect(camera.url) as ws1:
            # First client should receive BPP-wrapped welcome message
            welcome_raw = await ws1.recv()
            welcome_message = json.loads(codec.decode(welcome_raw))
            assert welcome_message["status"] == "connected"

            # Try to connect second client while first is connected
            try:
                async with websockets.connect(camera.url) as ws2:
                    # Second client should receive BPP-wrapped rejection message
                    rejection_raw = await asyncio.wait_for(ws2.recv(), timeout=1.0)
                    rejection_message = json.loads(codec.decode(rejection_raw))
                    assert "error" in rejection_message
            except websockets.exceptions.ConnectionClosed:
                # Connection closed immediately - also acceptable
                pass
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_welcome_message():
    """Test that welcome message is sent to connected client (BPP-wrapped)."""
    codec = BPPCodec("", enable_encryption=False)
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Should receive BPP-wrapped welcome message
            welcome_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome_message = json.loads(welcome_payload)
            assert "message" in welcome_message
            assert welcome_message["status"] == "connected"
            assert tuple(welcome_message["resolution"]) == camera.resolution
            assert welcome_message["fps"] == camera.fps
            assert welcome_message["security_mode"] == camera.security_mode


@pytest.mark.asyncio
async def test_websocket_camera_receives_frames(bpp_frame_binary):
    """Test that server receives and queues BPP-wrapped frames from client."""
    with WebSocketCamera(port=0) as camera:
        async with websockets.connect(camera.url) as ws:
            # Skip welcome message
            await ws.recv()

            # Send a BPP-wrapped frame
            await ws.send(bpp_frame_binary)

            # Give time for frame to be processed
            await asyncio.sleep(0.2)

            # Frame should be in queue
            assert camera.capture() is not None


@pytest.mark.asyncio
async def test_websocket_camera_disconnects_client_on_stop():
    """Test that connected client is disconnected when camera stops."""
    codec = BPPCodec("", enable_encryption=False)
    camera = WebSocketCamera(port=0)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            # Client connected, receive BPP-wrapped welcome message
            welcome_raw = await ws.recv()
            welcome_message = json.loads(codec.decode(welcome_raw))
            assert welcome_message["status"] == "connected"

            # Stop the camera (runs in background thread via to_thread)
            await asyncio.to_thread(camera.stop)

            with pytest.raises(websockets.exceptions.ConnectionClosed):
                # Keep receiving until connection is closed
                while True:
                    goodbye_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    goodbye_message = json.loads(codec.decode(goodbye_raw))
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
async def test_websocket_camera_backpressure():
    """Test that old frames are dropped when new frames arrive faster than they're consumed."""
    codec = BPPCodec("", enable_encryption=False)
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

    camera = WebSocketCamera(port=0)

    def event_listener(event_type, data):
        if event_type == "connected":
            main_loop.call_soon_threadsafe(connected.set)
            assert "client_address" in data
            assert "client_name" in data
            assert data["client_name"] == "test_client"
            assert camera.name == "test_client"
        if event_type == "disconnected":
            main_loop.call_soon_threadsafe(disconnected.set)
            assert "client_address" in data
            assert "client_name" in data
            assert data["client_name"] == "test_client"
            assert camera.name == "test_client"
        events.append((event_type, data))

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


TEST_SECRET = "test-secret-key"


@pytest.mark.asyncio
async def test_websocket_camera_authenticated_mode():
    """Test sending and receiving frames with HMAC-SHA256 authentication (secret, no encryption)."""
    codec = BPPCodec(TEST_SECRET, enable_encryption=False)
    camera = WebSocketCamera(port=0, secret=TEST_SECRET, encrypt=False)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            # Welcome message should be BPP-encoded
            welcome_raw = await ws.recv()
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome = json.loads(welcome_payload)
            assert welcome["status"] == "connected"
            assert "authenticated" in welcome["security_mode"]

            # Send a BPP-encoded frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(codec.encode(buffer.tobytes()))

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is not None
            assert isinstance(captured, np.ndarray)
            assert np.mean(captured) == pytest.approx(42, abs=1)
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_authenticated_rejects_raw():
    """Test that authenticated mode rejects raw (non-BPP) messages."""
    camera = WebSocketCamera(port=0, secret=TEST_SECRET, encrypt=False)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            await ws.recv()  # Skip welcome

            # Send raw bytes (not BPP-encoded) — should be rejected
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(buffer.tobytes())

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is None
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_encrypted_mode():
    """Test sending and receiving frames with ChaCha20-Poly1305 encryption."""
    codec = BPPCodec(TEST_SECRET, enable_encryption=True)
    camera = WebSocketCamera(port=0, secret=TEST_SECRET, encrypt=True)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            # Welcome message should be BPP-encoded with encryption
            welcome_raw = await ws.recv()
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome = json.loads(welcome_payload)
            assert welcome["status"] == "connected"
            assert "encrypted" in welcome["security_mode"]

            # Send a BPP-encrypted frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 99
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(codec.encode(buffer.tobytes()))

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is not None
            assert isinstance(captured, np.ndarray)
            assert np.mean(captured) == pytest.approx(99, abs=1)
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_encrypted_rejects_raw():
    """Test that encrypted mode rejects raw (non-BPP) messages."""
    camera = WebSocketCamera(port=0, secret=TEST_SECRET, encrypt=True)
    camera.start()

    try:
        async with websockets.connect(camera.url) as ws:
            await ws.recv()  # Skip welcome

            # Send raw bytes — should be rejected
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(buffer.tobytes())

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is None
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_raw_mode():
    """Test that clients can bypass BPP with ?raw=true when security is disabled."""
    camera = WebSocketCamera(port=0)
    camera.start()

    try:
        # Connect with raw=true query parameter
        async with websockets.connect(camera.url + "?raw=true") as ws:
            # Welcome should be plain JSON (not BPP-wrapped)
            welcome = await asyncio.wait_for(ws.recv(), timeout=1.0)
            welcome_message = json.loads(welcome)
            assert welcome_message["status"] == "connected"

            # Send raw JPEG bytes (no BPP wrapping)
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 77
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(buffer.tobytes())

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is not None
            assert np.mean(captured) == pytest.approx(77, abs=1)
    finally:
        camera.stop()


@pytest.mark.asyncio
async def test_websocket_camera_raw_mode_ignored_with_secret():
    """Test that ?raw=true is ignored when security is enabled."""
    codec = BPPCodec(TEST_SECRET, enable_encryption=False)
    camera = WebSocketCamera(port=0, secret=TEST_SECRET)
    camera.start()

    try:
        # Connect with raw=true — should be ignored since secret is set
        async with websockets.connect(camera.url + "?raw=true") as ws:
            # Welcome should still be BPP-wrapped
            welcome_raw = await ws.recv()
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome = json.loads(welcome_payload)
            assert welcome["status"] == "connected"

            # Sending raw bytes should be rejected (BPP is enforced)
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(buffer.tobytes())

            await asyncio.sleep(0.2)

            captured = camera.capture()
            assert captured is None
    finally:
        camera.stop()
