# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import socket

import pytest
import asyncio
import websockets
import json
import base64
import numpy as np

from arduino.app_internal.core.peripherals.bpp_codec import BPPCodec
from arduino.app_peripherals.microphone import WebSocketMicrophone, MicrophoneOpenError


class TestWebSocketMicrophoneInit:
    """Test WebSocketMicrophone initialization and startup."""

    @pytest.mark.asyncio
    async def test_websocket_start_stop(self):
        mic = WebSocketMicrophone(port=0)

        mic.start()
        assert mic.is_started()
        assert mic._server is not None

        mic.stop()
        assert not mic.is_started()
        assert mic._server is None

    def test_encrypt_without_secret_fails(self):
        """Test that encrypt=True without a secret raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Encryption requires a secret key"):
            WebSocketMicrophone(encrypt=True)

    def test_empty_string_secret_enables_bpp(self):
        """Test that secret="" is valid and enables BPP authentication."""
        mic = WebSocketMicrophone(port=0, secret="")
        assert mic.codec is not None
        assert mic.secret == ""

    @pytest.mark.asyncio
    async def test_start_on_unavailable_port_fails(self):
        """Test that starting on an unavailable port fails gracefully."""
        # Occupy a port so the microphone server can't bind to it
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        occupied_port = blocker.getsockname()[1]

        try:
            mic = WebSocketMicrophone(port=occupied_port)
            mic._bind_ip = "127.0.0.1"

            with pytest.raises(MicrophoneOpenError):
                mic.start()
        finally:
            blocker.close()


class TestWebSocketPCMBinaryFormat:
    """Test receiving binary PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int16(self):
        """Test receiving binary PCM data as int16."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone()

        mic.start()

        # Create test PCM data
        test_audio = np.arange(1024, dtype=np.int16)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()  # Welcome message

            # Send BPP-encoded PCM bytes
            await ws.send(codec.encode(pcm_bytes))

            # Capture and validate
            received = mic.capture()

            assert received is not None
            assert isinstance(received, np.ndarray)
            assert received.dtype == np.int16
            assert len(received) == 1024
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int32(self):
        """Test receiving binary PCM data as int32."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0, format=np.int32)

        mic.start()

        test_audio = np.arange(512, dtype=np.int32)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(codec.encode(pcm_bytes))

            received = mic.capture()

            assert received is not None
            assert received.dtype == np.int32
            assert len(received) == 512
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_float32(self):
        """Test receiving binary PCM data as float32."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0, format=np.float32)

        mic.start()

        test_audio = np.random.randn(256).astype(np.float32)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(codec.encode(pcm_bytes))

            received = mic.capture()

            assert received is not None
            assert received.dtype == np.float32
            assert len(received) == 256
            np.testing.assert_array_almost_equal(received, test_audio)

        mic.stop()


class TestWebSocketPCMBase64Format:
    """Test receiving base64-encoded PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_base64_encoded_pcm(self):
        """Test receiving base64-encoded BPP data."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Create test PCM data, wrap in BPP, then base64-encode
        test_audio = np.arange(512, dtype=np.int16)
        bpp_encoded = codec.encode(test_audio.tobytes())
        base64_encoded = base64.b64encode(bpp_encoded).decode()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(base64_encoded)

            received = mic.capture()

            assert received is not None
            assert isinstance(received, np.ndarray)
            assert received.dtype == np.int16
            assert len(received) == 512
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_base64_with_padding(self):
        """Test receiving base64-encoded BPP data with padding."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Use size that requires padding in base64
        test_audio = np.arange(100, dtype=np.int16)
        bpp_encoded = codec.encode(test_audio.tobytes())
        base64_encoded = base64.b64encode(bpp_encoded).decode()

        # Verify it has padding
        assert "=" in base64_encoded or len(base64_encoded) % 4 == 0

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(base64_encoded)

            received = mic.capture()

            assert received is not None
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()


class TestWebSocketMultipleChunks:
    """Test receiving multiple PCM chunks sequentially."""

    @pytest.mark.asyncio
    async def test_receive_multiple_sequential_chunks(self):
        """Test receiving correctly multiple PCM chunks in sequence."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            # Send 5 chunks with different values
            sent_chunks = []
            for i in range(5):  # Internal queue holds up to 10 chunks
                chunk = np.full(128, i, dtype=np.int16)
                sent_chunks.append(chunk)
                await ws.send(codec.encode(chunk.tobytes()))

            received_chunks = []
            for _ in range(5):
                chunk = mic.capture()
                if chunk is not None:
                    received_chunks.append(chunk)

            assert len(received_chunks) > 0

            for chunk in received_chunks:
                assert isinstance(chunk, np.ndarray)
                assert chunk.dtype == np.int16

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_rapid_fire_chunks(self):
        """Test receiving chunks sent in rapid succession."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            # Send chunks rapidly without delay
            for i in range(10):  # Internal queue holds up to 10 chunks
                chunk = np.full(64, i, dtype=np.int16)
                await ws.send(codec.encode(chunk.tobytes()))

            # Should handle rapid chunks
            for i in range(10):
                received = mic.capture()
                assert received is not None

        mic.stop()


class TestWebSocketPCMDataIntegrity:
    """Test data integrity of received PCM streams."""

    @pytest.mark.asyncio
    async def test_pcm_values_preserved_exactly(self):
        """Test that PCM values are preserved exactly through transmission."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Create test pattern with known values
        test_audio = np.array([0, 100, -100, 32000, -32000, 1, -1], dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(codec.encode(test_audio.tobytes()))

            received = mic.capture()

            assert received is not None
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_pcm_byte_order_preserved(self):
        """Test that byte order is preserved in PCM transmission."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Test with values that would differ if byte order is wrong
        test_audio = np.array([256, 257, 258], dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(codec.encode(test_audio.tobytes()))

            received = mic.capture()

            np.testing.assert_array_equal(received, test_audio)

        mic.stop()


class TestWebSocketClientConnection:
    """Test WebSocket client connection handling."""

    @pytest.mark.asyncio
    async def test_client_receives_welcome_message(self):
        """Test that client receives welcome message on connection."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            welcome_raw = await ws.recv()

            welcome_data = json.loads(codec.decode(welcome_raw))

            assert "status" in welcome_data
            assert welcome_data["status"] == "connected"
            assert "security_mode" in welcome_data
            assert "none" in welcome_data["security_mode"]

        mic.stop()

    @pytest.mark.asyncio
    async def test_single_client_enforcement(self):
        """Test that only one client can connect at a time."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Connect first client
        async with websockets.connect(mic.url) as client1:
            welcome_raw = await client1.recv()  # Welcome
            welcome_data = json.loads(codec.decode(welcome_raw))
            assert "status" in welcome_data
            assert welcome_data["status"] == "connected"

            # Try to connect second client
            async with websockets.connect(mic.url) as client2:
                # Should receive rejection
                rejection_raw = await client2.recv()
                rejection_data = json.loads(codec.decode(rejection_raw))
                assert "error" in rejection_data

        mic.stop()

    @pytest.mark.asyncio
    async def test_client_disconnection_handled(self):
        """Test that client disconnection is handled gracefully."""
        codec = BPPCodec("", enable_encryption=False)
        loop = asyncio.get_running_loop()
        test_done = asyncio.Event()

        def callback(status, status_info):
            if status == "disconnected":
                assert mic.is_started()
                assert mic._server is not None
                assert mic._client is None
                loop.call_soon_threadsafe(test_done.set)

        mic = WebSocketMicrophone(port=0)
        mic.on_status_changed(callback)

        mic.start()

        test_audio = np.zeros(128, dtype=np.int16).tobytes()

        # Connect and disconnect
        async with websockets.connect(mic.url) as ws:
            await ws.recv()
            await ws.send(codec.encode(test_audio))

        await asyncio.wait_for(test_done.wait(), timeout=2)
        mic.stop()


class TestWebSocketClientEvents:
    """Test WebSocket client event emission with client_name."""

    @pytest.mark.asyncio
    async def test_client_events(self):
        """
        Test that WebSocket microphone emits connection and disconnection events
        with client_name extracted from URL query parameters.
        """
        events = []
        main_loop = asyncio.get_running_loop()

        connected = asyncio.Event()
        disconnected = asyncio.Event()

        mic = WebSocketMicrophone(port=0)

        def event_listener(event_type, data):
            if event_type == "connected":
                main_loop.call_soon_threadsafe(connected.set)
                assert "client_address" in data
                assert "client_name" in data
                assert data["client_name"] == "test_client"
                assert mic.name == "test_client"
            if event_type == "disconnected":
                main_loop.call_soon_threadsafe(disconnected.set)
                assert "client_address" in data
                assert "client_name" in data
                assert data["client_name"] == "test_client"
                assert mic.name == "test_client"
            events.append((event_type, data))

        mic.on_status_changed(event_listener)
        mic.start()

        # This should emit connection and disconnection events
        async def client_task():
            async with websockets.connect(mic.url + "?client_name=test_client"):
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

        mic.stop()  # This should not emit a disconnection

        await asyncio.sleep(0.1)

        # Check that stop() didn't emit additional events
        assert len(events) == 2
        assert "connected" in events[0][0]
        assert "disconnected" in events[1][0]


class TestWebSocketClientDisconnection:
    """Test WebSocket client disconnection handling."""

    @pytest.mark.asyncio
    async def test_client_disconnect_handled_gracefully(self):
        """Test that client disconnection is handled gracefully."""
        connected = asyncio.Event()
        disconnected = asyncio.Event()
        loop = asyncio.get_running_loop()

        def callback(status, status_info):
            if status == "connected":
                assert mic.is_started()
                assert mic._server is not None
                assert mic._client is not None
                loop.call_soon_threadsafe(connected.set)
            if status == "disconnected":
                assert mic.is_started()
                assert mic._server is not None
                assert mic._client is None
                loop.call_soon_threadsafe(disconnected.set)

        mic = WebSocketMicrophone(port=0)
        mic.on_status_changed(callback)

        mic.start()

        # Connect and disconnect
        async with websockets.connect(mic.url) as ws:
            await asyncio.wait_for(connected.wait(), timeout=2)
            await ws.recv()

        await asyncio.wait_for(disconnected.wait(), timeout=2)
        mic.stop()

        assert not mic.is_started()
        assert mic._server is None
        assert mic._client is None

    @pytest.mark.asyncio
    async def test_client_reconnect_after_disconnect(self):
        """Test that client can reconnect after disconnecting."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # First connection
        async with websockets.connect(mic.url) as ws:
            await ws.recv()

        # Second connection should work
        async with websockets.connect(mic.url) as ws:
            welcome_raw = await ws.recv()
            assert b"connected" in codec.decode(welcome_raw)

        mic.stop()

    @pytest.mark.asyncio
    async def test_client_abrupt_disconnect(self):
        """Test handling of abrupt client disconnect."""
        loop = asyncio.get_running_loop()
        test_done = asyncio.Event()

        def callback(status, status_info):
            if status == "disconnected":
                assert mic.is_started()
                assert mic._server is not None
                assert mic._client is None
                loop.call_soon_threadsafe(test_done.set)

        mic = WebSocketMicrophone(port=0)
        mic.on_status_changed(callback)

        mic.start()

        ws = await websockets.connect(mic.url)
        await ws.recv()

        # Abruptly close without proper shutdown
        await ws.close()

        await asyncio.wait_for(test_done.wait(), timeout=2)
        mic.stop()


class TestWebSocketMessageParsing:
    """Test message parsing and validation."""

    @pytest.mark.asyncio
    async def test_wrong_message_type_handled(self):
        """Test that wrong message type is handled."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            # Send text when expecting encoded data
            await ws.send("text message")

            # Should handle gracefully
            received = mic.capture()
            assert received is None

        mic.stop()


class TestWebSocketPCMStreaming:
    """Test continuous PCM streaming from WebSocket."""

    @pytest.mark.asyncio
    async def test_continuous_pcm_stream(self):
        """Test continuous PCM streaming from client."""
        codec = BPPCodec("", enable_encryption=False)
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async def stream_audio():
            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Stream 10 chunks then stop
                for i in range(10):
                    chunk = np.full(128, i, dtype=np.int16)
                    await ws.send(codec.encode(chunk.tobytes()))

        # Start streaming
        stream_task = asyncio.create_task(stream_audio())

        # Start capturing
        def collect_chunks():
            chunks = []
            stream = mic.stream()
            for i, chunk in enumerate(stream):
                chunks.append(chunk)
                if i >= 9:
                    break
            return chunks

        chunks = await asyncio.to_thread(collect_chunks)

        await stream_task

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16

        mic.stop()


class TestWebSocketRawMode:
    """Test raw mode that bypasses BPP framing."""

    @pytest.mark.asyncio
    async def test_raw_mode_bypasses_bpp(self):
        """Test that ?raw=true allows sending raw PCM bytes without BPP wrapping."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        test_audio = np.arange(256, dtype=np.int16)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url + "?raw=true") as ws:
            await ws.recv()  # Welcome message

            # Send raw PCM bytes directly (no BPP wrapping)
            await ws.send(pcm_bytes)

            received = mic.capture()

            assert received is not None
            assert isinstance(received, np.ndarray)
            assert received.dtype == np.int16
            assert len(received) == 256
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_raw_mode_ignored_with_secret(self):
        """Test that ?raw=true is ignored when a secret is set (BPP still enforced)."""
        mic = WebSocketMicrophone(port=0, secret="my-secret", encrypt=False)

        mic.start()

        test_audio = np.arange(256, dtype=np.int16)

        async with websockets.connect(mic.url + "?raw=true") as ws:
            await ws.recv()  # Welcome message

            # Send raw PCM bytes — should be rejected because BPP is enforced
            await ws.send(test_audio.tobytes())

            await asyncio.sleep(0.1)

            received = mic.capture()
            assert received is None

        mic.stop()


TEST_SECRET = "test-secret-key"


class TestWebSocketMicrophoneAuthenticated:
    """Test WebSocket microphone with HMAC-SHA256 authentication (secret, no encryption)."""

    @pytest.mark.asyncio
    async def test_receive_authenticated_pcm(self):
        """Test receiving BPP-authenticated PCM data."""
        codec = BPPCodec(TEST_SECRET, enable_encryption=False)
        mic = WebSocketMicrophone(port=0, secret=TEST_SECRET, encrypt=False)
        mic.start()

        test_audio = np.arange(512, dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            # Welcome should be BPP-encoded
            welcome_raw = await ws.recv()
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome = json.loads(welcome_payload)
            assert welcome["status"] == "connected"
            assert "authenticated" in welcome["security_mode"]

            # Send BPP-encoded audio
            await ws.send(codec.encode(test_audio.tobytes()))

            received = mic.capture()
            assert received is not None
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_authenticated_rejects_raw(self):
        """Test that authenticated mode rejects raw (non-BPP) messages."""
        mic = WebSocketMicrophone(port=0, secret=TEST_SECRET, encrypt=False)
        mic.start()

        test_audio = np.arange(512, dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            await ws.recv()  # Skip welcome

            # Send raw bytes — should be rejected
            await ws.send(test_audio.tobytes())

            await asyncio.sleep(0.1)

            received = mic.capture()
            assert received is None

        mic.stop()


class TestWebSocketMicrophoneEncrypted:
    """Test WebSocket microphone with ChaCha20-Poly1305 encryption."""

    @pytest.mark.asyncio
    async def test_receive_encrypted_pcm(self):
        """Test receiving BPP-encrypted PCM data."""
        codec = BPPCodec(TEST_SECRET, enable_encryption=True)
        mic = WebSocketMicrophone(port=0, secret=TEST_SECRET, encrypt=True)
        mic.start()

        test_audio = np.arange(256, dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            # Welcome should be BPP-encoded with encryption
            welcome_raw = await ws.recv()
            welcome_payload = codec.decode(welcome_raw)
            assert welcome_payload is not None
            welcome = json.loads(welcome_payload)
            assert welcome["status"] == "connected"
            assert "encrypted" in welcome["security_mode"]

            # Send BPP-encrypted audio
            await ws.send(codec.encode(test_audio.tobytes()))

            received = mic.capture()
            assert received is not None
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_encrypted_rejects_raw(self):
        """Test that encrypted mode rejects raw (non-BPP) messages."""
        mic = WebSocketMicrophone(port=0, secret=TEST_SECRET, encrypt=True)
        mic.start()

        test_audio = np.arange(256, dtype=np.int16)

        async with websockets.connect(mic.url) as ws:
            await ws.recv()  # Skip welcome

            # Send raw bytes — should be rejected
            await ws.send(test_audio.tobytes())

            await asyncio.sleep(0.1)

            received = mic.capture()
            assert received is None

        mic.stop()
