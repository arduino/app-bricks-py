# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import asyncio
import websockets
import json
import base64
import numpy as np

from arduino.app_internal.core.peripherals import BPPCodec
from arduino.app_peripherals.microphone import WebSocketMicrophone, MicrophoneOpenError


@pytest.fixture
def codec() -> BPPCodec:
    """Fixture to provide a codec if needed in future tests."""
    return BPPCodec()


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

    @pytest.mark.asyncio
    async def test_start_on_privileged_port_fails(self):
        """Test that starting on privileged port fails gracefully."""
        mic = WebSocketMicrophone(port=1)
        mic._bind_ip = "127.0.0.1"  # Workaround for MacOS

        with pytest.raises(MicrophoneOpenError):
            mic.start()


class TestWebSocketPCMBinaryFormat:
    """Test receiving binary PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int16(self, codec):
        """Test receiving binary PCM data as int16."""
        mic = WebSocketMicrophone()

        mic.start()

        # Create test PCM data
        test_audio = np.arange(1024, dtype=np.int16)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()  # Welcome message

            # Send binary PCM data
            encoded = codec.encode(pcm_bytes)
            await ws.send(encoded)

            # Capture and validate
            received = mic.capture()

            assert received is not None
            assert isinstance(received, np.ndarray)
            assert received.dtype == np.int16
            assert len(received) == 1024
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int32(self, codec):
        """Test receiving binary PCM data as int32."""
        mic = WebSocketMicrophone(port=0, format=np.int32)

        mic.start()

        test_audio = np.arange(512, dtype=np.int32)
        pcm_bytes = test_audio.tobytes()
        encoded = codec.encode(pcm_bytes)

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(encoded)

            received = mic.capture()

            assert received is not None
            assert received.dtype == np.int32
            assert len(received) == 512
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_float32(self, codec):
        """Test receiving binary PCM data as float32."""
        mic = WebSocketMicrophone(port=0, format=np.float32)

        mic.start()

        test_audio = np.random.randn(256).astype(np.float32)
        pcm_bytes = test_audio.tobytes()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            encoded = codec.encode(pcm_bytes)
            await ws.send(encoded)

            received = mic.capture()

            assert received is not None
            assert received.dtype == np.float32
            assert len(received) == 256
            np.testing.assert_array_almost_equal(received, test_audio)

        mic.stop()


class TestWebSocketPCMBase64Format:
    """Test receiving base64-encoded PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_base64_encoded_pcm(self, codec):
        """Test receiving base64-encoded PCM data."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Create and encode test PCM data
        test_audio = np.arange(512, dtype=np.int16)
        pcm_bytes = test_audio.tobytes()
        encoded = codec.encode(pcm_bytes)
        base64_encoded = base64.b64encode(encoded).decode()

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
    async def test_receive_base64_with_padding(self, codec):
        """Test receiving base64 data with padding."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Use size that requires padding in base64
        test_audio = np.arange(100, dtype=np.int16)
        pcm_bytes = test_audio.tobytes()
        encoded = codec.encode(pcm_bytes)
        base64_encoded = base64.b64encode(encoded).decode()

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
    async def test_receive_multiple_sequential_chunks(self, codec):
        """Test receiving correctly multiple PCM chunks in sequence."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            # Send 5 chunks with different values
            sent_chunks = []
            for i in range(5):  # Internal queue holds up to 10 chunks
                chunk = np.full(128, i, dtype=np.int16)
                sent_chunks.append(chunk)
                encoded = codec.encode(chunk.tobytes())
                await ws.send(encoded)

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
    async def test_receive_rapid_fire_chunks(self, codec):
        """Test receiving chunks sent in rapid succession."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            # Send chunks rapidly without delay
            for i in range(10):  # Internal queue holds up to 10 chunks
                chunk = np.full(64, i, dtype=np.int16)
                encoded = codec.encode(chunk.tobytes())
                await ws.send(encoded)

            # Should handle rapid chunks
            for i in range(10):
                received = mic.capture()
                assert received is not None

        mic.stop()


class TestWebSocketPCMDataIntegrity:
    """Test data integrity of received PCM streams."""

    @pytest.mark.asyncio
    async def test_pcm_values_preserved_exactly(self, codec):
        """Test that PCM values are preserved exactly through transmission."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Create test pattern with known values
        test_audio = np.array([0, 100, -100, 32000, -32000, 1, -1], dtype=np.int16)
        encoded = codec.encode(test_audio.tobytes())

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(encoded)

            received = mic.capture()

            assert received is not None
            np.testing.assert_array_equal(received, test_audio)

        mic.stop()

    @pytest.mark.asyncio
    async def test_pcm_byte_order_preserved(self, codec):
        """Test that byte order is preserved in PCM transmission."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Test with values that would differ if byte order is wrong
        test_audio = np.array([256, 257, 258], dtype=np.int16)
        encoded = codec.encode(test_audio.tobytes())

        async with websockets.connect(mic.url) as ws:
            await ws.recv()

            await ws.send(encoded)

            received = mic.capture()

            np.testing.assert_array_equal(received, test_audio)

        mic.stop()


class TestWebSocketClientConnection:
    """Test WebSocket client connection handling."""

    @pytest.mark.asyncio
    async def test_client_receives_welcome_message(self, codec):
        """Test that client receives welcome message on connection."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async with websockets.connect(mic.url) as ws:
            welcome = await ws.recv()

            decoded = codec.decode(welcome)
            welcome_data = json.loads(decoded)

            assert "status" in welcome_data
            assert welcome_data["status"] == "connected"
            assert "security_mode" in welcome_data
            assert "none" in welcome_data["security_mode"]

        mic.stop()

    @pytest.mark.asyncio
    async def test_single_client_enforcement(self, codec):
        """Test that only one client can connect at a time."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # Connect first client
        async with websockets.connect(mic.url) as client1:
            welcome = await client1.recv()  # Welcome
            decoded = codec.decode(welcome)
            welcome_data = json.loads(decoded)
            assert "status" in welcome_data
            assert welcome_data["status"] == "connected"

            # Try to connect second client
            async with websockets.connect(mic.url) as client2:
                # Should receive rejection
                rejection = await client2.recv()
                decoded = codec.decode(rejection)
                rejection_data = json.loads(decoded)
                assert "error" in rejection_data

        mic.stop()

    @pytest.mark.asyncio
    async def test_client_disconnection_handled(self, codec):
        """Test that client disconnection is handled gracefully."""
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
        encoded = codec.encode(test_audio)

        # Connect and disconnect
        async with websockets.connect(mic.url) as ws:
            await ws.recv()
            await ws.send(encoded)

        await asyncio.wait_for(test_done.wait(), timeout=2)
        mic.stop()


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
    async def test_client_reconnect_after_disconnect(self, codec):
        """Test that client can reconnect after disconnecting."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        # First connection
        async with websockets.connect(mic.url) as ws:
            await ws.recv()

        # Second connection should work
        async with websockets.connect(mic.url) as ws:
            welcome = await ws.recv()
            decoded = codec.decode(welcome)
            assert "connected" in decoded.decode()

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
    async def test_continuous_pcm_stream(self, codec):
        """Test continuous PCM streaming from client."""
        mic = WebSocketMicrophone(port=0)

        mic.start()

        async def stream_audio():
            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Stream 10 chunks then stop
                for i in range(10):
                    chunk = np.full(128, i, dtype=np.int16)
                    encoded = codec.encode(chunk.tobytes())
                    await ws.send(encoded)

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
