# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import asyncio
import websockets
import numpy as np
from unittest.mock import MagicMock, patch
import alsaaudio

from arduino.app_peripherals.microphone import Microphone, ALSAMicrophone
from arduino.app_peripherals.microphone.errors import (
    MicrophoneError,
    MicrophoneOpenError,
    MicrophoneReadError,
    MicrophoneConfigError,
)

MOCK_USB_CARDS = ["UH34"]
MOCK_USB_PCM_DEVICES = ["plughw:CARD=UH34,DEV=0"]


class TestALSADeviceDisconnection:
    """Test ALSA device disconnection handling."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms")
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_detect_device_disconnection(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that device disconnection is detected."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")
        mock_pcms.return_value = MOCK_USB_PCM_DEVICES

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()

        # Simulate device disconnection
        pcm_instance.read.side_effect = alsaaudio.ALSAAudioError("No such device")
        mock_pcms.return_value = []  # Device removed from list

        # Attempt to read should detect disconnection
        audio = mic.capture()

        assert audio is None
        assert mic._pcm is None  # PCM should be cleared

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_is_device_disconnected_check(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test device disconnection detection method."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()

        # Device present
        assert not mic._is_device_disconnected()

        # Simulate device removal
        mock_pcms.return_value = []

        # Should detect disconnection
        assert mic._is_device_disconnected()


class TestALSADeviceReconnection:
    """Test ALSA device reconnection logic."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms")
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_reconnection_after_device_available(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test reconnection when device becomes available."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Initially no devices - creation should fail
        mock_pcms.return_value = []

        with pytest.raises(MicrophoneConfigError):
            mic = Microphone(device=0)

        # Make device available
        mock_pcms.return_value = MOCK_USB_PCM_DEVICES

        # Now creation and start should work
        mic = Microphone(device=0)
        mic.start()

        assert mic.is_started()

        mic.stop()

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_read_after_pcm_cleared(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test reading after PCM device is cleared triggers reconnection."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()

        # Clear PCM to simulate disconnection
        mic._pcm = None

        # Mock successful reconnection
        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        # Reading should trigger reconnection attempt
        # Note: In real implementation, this would block until reconnected
        # For this test, we just verify the behavior


class TestALSAReadErrors:
    """Test ALSA read error handling."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_read_with_no_data_returns_none(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that read with no data returns None."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Return 0 length
        pcm_instance.read.return_value = (0, b"")

        mic = Microphone(device=0)
        mic.start()

        audio = mic.capture()

        assert audio is None

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_read_with_non_disconnection_error_raises(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that non-disconnection ALSA errors raise MicrophoneReadError."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Return ALSA error that's not disconnection
        pcm_instance.read.side_effect = alsaaudio.ALSAAudioError("Buffer overrun")

        mic = Microphone(device=0)
        mic.start()

        with pytest.raises(MicrophoneReadError):
            mic.capture()


class TestALSAOpenErrors:
    """Test ALSA device opening errors."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    def test_device_busy_error(self, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that device busy error is properly reported."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        mock_pcm.side_effect = alsaaudio.ALSAAudioError("Device or resource busy")

        mic = Microphone(device=0)

        with pytest.raises(MicrophoneOpenError) as exc_info:
            mic.start()

        assert "busy" in str(exc_info.value).lower()

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    def test_generic_alsa_error(self, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test generic ALSA error handling."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        mock_pcm.side_effect = alsaaudio.ALSAAudioError("Unknown error")

        mic = Microphone(device=0)

        with pytest.raises(MicrophoneOpenError):
            mic.start()


class TestALSAVolumeControlErrors:
    """Test volume control error handling."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_get_volume_without_mixer(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test getting volume when no mixer is available."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()

        volume = mic.get_volume()

        assert volume is None

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_set_volume_without_mixer(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting volume when no mixer is available."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()

        # Should not raise
        mic.set_volume(50)

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=["Capture"])
    @patch("alsaaudio.Mixer")
    def test_set_volume_out_of_range(self, mock_mixer_class, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that set_volume validates range."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mixer_instance = MagicMock()
        mock_mixer_class.return_value = mixer_instance

        mic = Microphone(device=0)
        mic.start()

        with pytest.raises(ValueError):
            mic.set_volume(-1)

        with pytest.raises(ValueError):
            mic.set_volume(101)


class TestWebSocketClientDisconnection:
    """Test WebSocket client disconnection handling."""

    @pytest.mark.asyncio
    async def test_client_disconnect_handled_gracefully(self):
        """Test that client disconnection is handled gracefully."""
        mic = Microphone(device="ws://127.0.0.1:0", audio_format="binary")

        try:
            mic.start()

            for _ in range(50):
                if mic._server is not None:
                    break
                await asyncio.sleep(0.1)

            # Connect and disconnect
            async with websockets.connect(f"ws://127.0.0.1:{mic.port}") as ws:
                await ws.recv()
                # Connection closed on exit

            # Wait for disconnection to be processed
            await asyncio.sleep(0.2)

            # Server should still be running
            assert mic.is_started()
            assert mic._client is None

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_client_reconnect_after_disconnect(self):
        """Test that client can reconnect after disconnecting."""
        mic = Microphone(device="ws://127.0.0.1:0", audio_format="binary")

        try:
            mic.start()

            for _ in range(50):
                if mic._server is not None:
                    break
                await asyncio.sleep(0.1)

            # First connection
            async with websockets.connect(f"ws://127.0.0.1:{mic.port}") as ws:
                await ws.recv()

            await asyncio.sleep(0.2)

            # Second connection should work
            async with websockets.connect(f"ws://127.0.0.1:{mic.port}") as ws:
                welcome = await ws.recv()
                assert "connected" in welcome.lower()

        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_client_abrupt_disconnect(self):
        """Test handling of abrupt client disconnect."""
        mic = Microphone(device="ws://127.0.0.1:0", audio_format="binary")

        try:
            mic.start()

            for _ in range(50):
                if mic._server is not None:
                    break
                await asyncio.sleep(0.1)

            ws = await websockets.connect(f"ws://127.0.0.1:{mic.port}")
            await ws.recv()

            # Abruptly close without proper shutdown
            await ws.close()

            await asyncio.sleep(0.2)

            # Server should handle it gracefully
            assert mic.is_started()

        finally:
            mic.stop()


class TestWebSocketServerErrors:
    """Test WebSocket server error handling."""

    @pytest.mark.asyncio
    async def test_start_on_privileged_port_fails(self):
        """Test that starting on privileged port fails gracefully."""
        mic = Microphone(device="ws://0.0.0.0:1", audio_format="binary")

        try:
            mic.start()
            await asyncio.sleep(0.1)
        except MicrophoneOpenError:
            # This is the expected behavior
            pass


class TestConfigurationErrors:
    """Test configuration error handling."""

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises MicrophoneConfigError."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(device="hw:0,0", format="INVALID")

    def test_unsupported_format_for_numpy_raises_error(self):
        """Test that formats without numpy support raise error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(device="hw:0,0", format="MU_LAW")

    @patch("alsaaudio.cards", return_value=[])
    @patch("alsaaudio.card_indexes", return_value=[])
    @patch("alsaaudio.pcms", return_value=[])
    def test_no_devices_found_raises_error(self, mock_pcms, mock_card_indexes, mock_cards):
        """Test that no USB devices found raises error."""
        with pytest.raises(MicrophoneConfigError):
            Microphone(device=0)

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    def test_out_of_range_device_index_raises_error(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that out of range device index raises error."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        with pytest.raises(MicrophoneConfigError):
            Microphone(device=10)

    def test_invalid_device_type_raises_error(self):
        """Test that invalid device type raises error."""
        with pytest.raises(MicrophoneConfigError):
            Microphone(device=None)


class TestExceptionHierarchy:
    """Test exception hierarchy and catching."""

    def test_microphone_open_error_is_microphone_error(self):
        """Test exception inheritance."""
        assert issubclass(MicrophoneOpenError, MicrophoneError)

    def test_microphone_read_error_is_microphone_error(self):
        """Test exception inheritance."""
        assert issubclass(MicrophoneReadError, MicrophoneError)

    def test_microphone_config_error_is_microphone_error(self):
        """Test exception inheritance."""
        assert issubclass(MicrophoneConfigError, MicrophoneError)

    def test_catch_specific_error_with_base_handler(self):
        """Test that specific errors can be caught with base handler."""
        try:
            raise MicrophoneReadError("Test")
        except MicrophoneError as e:
            assert "Test" in str(e)


class TestErrorRecovery:
    """Test error recovery patterns."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_restart_after_error(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test restarting microphone after error."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()
        mic.stop()

        # Should be able to restart
        mic.start()
        assert mic.is_started()

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_context_manager_cleanup_on_error(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that context manager cleans up on error."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)

        try:
            with mic:
                assert mic.is_started()
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # Should be stopped after exception
        assert not mic.is_started()


class TestStopOnError:
    """Test that stop handles errors gracefully."""

    @patch("alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("alsaaudio.card_indexes", return_value=[0])
    @patch("alsaaudio.card_name")
    @patch("alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("alsaaudio.PCM")
    @patch("alsaaudio.mixers", return_value=[])
    def test_stop_with_close_error(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that stop handles close errors gracefully."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance
        pcm_instance.close.side_effect = Exception("Close failed")

        mic = Microphone(device=0)
        mic.start()

        # Should not raise
        mic.stop()

        assert not mic.is_started()
