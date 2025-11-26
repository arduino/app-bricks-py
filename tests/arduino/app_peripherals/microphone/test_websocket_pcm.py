# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import asyncio
import websockets
import json
import base64
import numpy as np

from arduino.app_peripherals.microphone import WebSocketMicrophone


class TestWebSocketPCMBinaryFormat:
    """Test receiving binary PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int16(self):
        """Test receiving binary PCM data as int16."""
        mic = WebSocketMicrophone()

        try:
            mic.start()

            # Create test PCM data
            test_audio = np.arange(1024, dtype=np.int16)
            pcm_bytes = test_audio.tobytes()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()  # Welcome message

                # Send binary PCM data
                await ws.send(pcm_bytes)

                # Wait for processing
                await asyncio.sleep(0.1)

                # Capture and validate
                received = mic.capture()

                assert received is not None
                assert isinstance(received, np.ndarray)
                assert received.dtype == np.int16
                assert len(received) == 1024
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_int32(self):
        """Test receiving binary PCM data as int32."""
        mic = WebSocketMicrophone(port=0, audio_format="binary", format="S32_LE")

        try:
            mic.start()

            test_audio = np.arange(512, dtype=np.int32)
            pcm_bytes = test_audio.tobytes()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(pcm_bytes)
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                assert received.dtype == np.int32
                assert len(received) == 512
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_binary_pcm_float32(self):
        """Test receiving binary PCM data as float32."""
        mic = WebSocketMicrophone(port=0, audio_format="binary", format="FLOAT_LE")

        try:
            mic.start()

            test_audio = np.random.randn(256).astype(np.float32)
            pcm_bytes = test_audio.tobytes()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(pcm_bytes)
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                assert received.dtype == np.float32
                assert len(received) == 256
                np.testing.assert_array_almost_equal(received, test_audio)

        finally:
            mic.stop()


class TestWebSocketPCMBase64Format:
    """Test receiving base64-encoded PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_base64_encoded_pcm(self):
        """Test receiving base64-encoded PCM data."""
        mic = WebSocketMicrophone(port=0, audio_format="base64")

        try:
            mic.start()

            # Create and encode test PCM data
            test_audio = np.arange(512, dtype=np.int16)
            pcm_bytes = test_audio.tobytes()
            base64_encoded = base64.b64encode(pcm_bytes).decode("utf-8")

            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Send base64-encoded data as text
                await ws.send(base64_encoded)
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                assert isinstance(received, np.ndarray)
                assert received.dtype == np.int16
                assert len(received) == 512
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_base64_with_padding(self):
        """Test receiving base64 data with padding."""
        mic = WebSocketMicrophone(port=0, audio_format="base64")

        try:
            mic.start()

            # Use size that requires padding in base64
            test_audio = np.arange(100, dtype=np.int16)
            pcm_bytes = test_audio.tobytes()
            base64_encoded = base64.b64encode(pcm_bytes).decode("utf-8")

            # Verify it has padding
            assert "=" in base64_encoded or len(base64_encoded) % 4 == 0

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(base64_encoded)
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()


class TestWebSocketPCMJSONFormat:
    """Test receiving JSON-wrapped PCM streams."""

    @pytest.mark.asyncio
    async def test_receive_json_wrapped_pcm(self):
        """Test receiving PCM data wrapped in JSON."""
        mic = WebSocketMicrophone(port=0, audio_format="json")

        try:
            mic.start()

            # Create test PCM and wrap in JSON
            test_audio = np.arange(256, dtype=np.int16)
            pcm_bytes = test_audio.tobytes()
            base64_encoded = base64.b64encode(pcm_bytes).decode("utf-8")
            json_message = json.dumps({"audio": base64_encoded})

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(json_message)
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                assert isinstance(received, np.ndarray)
                assert received.dtype == np.int16
                assert len(received) == 256
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_json_with_metadata(self):
        """Test receiving JSON with PCM and additional metadata."""
        mic = WebSocketMicrophone(port=0, audio_format="json")

        try:
            mic.start()

            test_audio = np.arange(128, dtype=np.int16)
            pcm_bytes = test_audio.tobytes()
            base64_encoded = base64.b64encode(pcm_bytes).decode("utf-8")

            # JSON with metadata
            json_message = json.dumps({"audio": base64_encoded, "timestamp": 1234567890, "metadata": {"device": "mic1"}})

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(json_message)
                await asyncio.sleep(0.1)

                received = mic.capture()

                # Should extract audio correctly despite extra fields
                assert received is not None
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_json_missing_audio_field(self):
        """Test that JSON without 'audio' field is handled gracefully."""
        mic = WebSocketMicrophone(port=0, audio_format="json")

        try:
            mic.start()

            # JSON without audio field
            json_message = json.dumps({"data": "not_audio"})

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(json_message)
                await asyncio.sleep(0.1)

                # Should return None for invalid message
                received = mic.capture()
                assert received is None

        finally:
            mic.stop()


class TestWebSocketMultipleChunks:
    """Test receiving multiple PCM chunks sequentially."""

    @pytest.mark.asyncio
    async def test_receive_multiple_sequential_chunks(self):
        """Test receiving multiple PCM chunks in sequence."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Send 5 chunks with different values
                sent_chunks = []
                for i in range(5):
                    chunk = np.full(128, i, dtype=np.int16)
                    sent_chunks.append(chunk)
                    await ws.send(chunk.tobytes())
                    await asyncio.sleep(0.05)

                await asyncio.sleep(0.2)

                received_chunks = []
                for _ in range(5):
                    chunk = mic.capture()
                    if chunk is not None:
                        received_chunks.append(chunk)

                assert len(received_chunks) > 0

                for chunk in received_chunks:
                    assert isinstance(chunk, np.ndarray)
                    assert chunk.dtype == np.int16

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_receive_rapid_fire_chunks(self):
        """Test receiving chunks sent in rapid succession."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Send chunks rapidly without delay
                for i in range(10):
                    chunk = np.full(64, i, dtype=np.int16)
                    await ws.send(chunk.tobytes())

                await asyncio.sleep(0.2)

                # Should handle rapid chunks
                received = mic.capture()
                assert received is not None

        finally:
            mic.stop()


class TestWebSocketPCMDataIntegrity:
    """Test data integrity of received PCM streams."""

    @pytest.mark.asyncio
    async def test_pcm_values_preserved_exactly(self):
        """Test that PCM values are preserved exactly through transmission."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            # Create test pattern with known values
            test_audio = np.array([0, 100, -100, 32000, -32000, 1, -1], dtype=np.int16)

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(test_audio.tobytes())
                await asyncio.sleep(0.1)

                received = mic.capture()

                assert received is not None
                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_pcm_byte_order_preserved(self):
        """Test that byte order is preserved in PCM transmission."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            # Test with values that would differ if byte order is wrong
            test_audio = np.array([256, 257, 258], dtype=np.int16)

            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(test_audio.tobytes())
                await asyncio.sleep(0.1)

                received = mic.capture()

                np.testing.assert_array_equal(received, test_audio)

        finally:
            mic.stop()


class TestWebSocketClientConnection:
    """Test WebSocket client connection handling."""

    @pytest.mark.asyncio
    async def test_client_receives_welcome_message(self):
        """Test that client receives welcome message on connection."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            async with websockets.connect(mic.url) as ws:
                welcome = await ws.recv()
                welcome_data = json.loads(welcome)

                assert welcome_data["status"] == "connected"
                assert "audio_format" in welcome_data
                assert welcome_data["audio_format"] == "binary"

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_single_client_enforcement(self):
        """Test that only one client can connect at a time."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            # Connect first client
            async with websockets.connect(mic.url) as client1:
                await client1.recv()  # Welcome

                # Try to connect second client
                try:
                    async with websockets.connect(mic.url) as client2:
                        msg = await client2.recv()
                        # Should receive rejection
                        assert "error" in msg.lower() or "busy" in msg.lower()
                except websockets.exceptions.ConnectionClosed:
                    # Or connection should be closed
                    pass

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_client_disconnection_handled(self):
        """Test that client disconnection is handled gracefully."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            # Connect and disconnect
            async with websockets.connect(mic.url) as ws:
                await ws.recv()
                await ws.send(np.zeros(128, dtype=np.int16).tobytes())

            # Wait for disconnection to be processed
            await asyncio.sleep(0.2)

            # Server should still be running
            assert mic.is_started()
            assert mic._client is None

        finally:
            mic.stop()


class TestWebSocketMessageParsing:
    """Test message parsing and validation."""

    @pytest.mark.asyncio
    async def test_invalid_json_handled_gracefully(self):
        """Test that invalid JSON is handled without crashing."""
        mic = WebSocketMicrophone(port=0, audio_format="json")

        try:
            mic.start()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Send invalid JSON
                await ws.send("not valid json {]")
                await asyncio.sleep(0.1)

                # Should return None, not crash
                received = mic.capture()
                assert received is None

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_wrong_message_type_handled(self):
        """Test that wrong message type is handled."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            async with websockets.connect(mic.url) as ws:
                await ws.recv()

                # Send text when expecting binary
                await ws.send("text message")
                await asyncio.sleep(0.1)

                # Should handle gracefully
                received = mic.capture()
                assert received is None

        finally:
            mic.stop()


class TestWebSocketPCMStreaming:
    """Test continuous PCM streaming from WebSocket."""

    @pytest.mark.asyncio
    async def test_continuous_pcm_stream(self):
        """Test continuous PCM streaming from client."""
        mic = WebSocketMicrophone(port=0, audio_format="binary")

        try:
            mic.start()

            async def stream_audio():
                async with websockets.connect(mic.url) as ws:
                    await ws.recv()

                    # Stream 10 chunks then stop
                    for i in range(10):
                        chunk = np.full(128, i, dtype=np.int16)
                        await ws.send(chunk.tobytes())

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

        finally:
            mic.stop()
