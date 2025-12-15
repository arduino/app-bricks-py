# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import pytest
import threading
import time
import numpy as np
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.microphone import Microphone, BaseMicrophone, ALSAMicrophone, WebSocketMicrophone
from arduino.app_peripherals.microphone.config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from arduino.app_peripherals.microphone.errors import MicrophoneConfigError

# Mock data for ALSA
MOCK_CARDS = ["SomeCard", "AnotherCard"]
MOCK_PCMS = ["plughw:CARD=SomeCard,DEV=0", "hw:CARD=SomeCard,DEV=0", "plughw:CARD=AnotherCard,DEV=0", "hw:CARD=AnotherCard,DEV=0"]


class TestMicrophoneFactoryInstantiation:
    """Test factory instantiation of different microphone types."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_factory_creates_alsa_microphone_with_integer(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test factory creates ALSA microphone with integer device index."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device=0)

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_factory_creates_alsa_microphone_with_string_index(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test factory creates ALSA microphone with string device index."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="1")

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_factory_creates_alsa_microphone_with_device_name(self, mock_pcms, mock_card_name):
        """Test factory creates ALSA microphone with explicit device name."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="hw:0,0")

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "hw:CARD=SomeCard,DEV=0"

    def test_factory_creates_websocket_microphone_with_ws_url(self):
        """Test factory creates WebSocket microphone with ws:// URL."""
        mic = Microphone(device="ws://localhost:9234")

        assert isinstance(mic, WebSocketMicrophone)
        assert mic._bind_ip == "0.0.0.0"
        assert mic.port == 9234

    def test_factory_invalid_device_type_raises_error(self):
        """Test that invalid device type raises MicrophoneConfigError."""
        with pytest.raises(MicrophoneConfigError):
            Microphone(device=None)


class TestMicrophoneConfiguration:
    """Test microphone configuration and parameters."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_default_parameters(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that microphones use default parameters."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device=0)

        assert mic.sample_rate == RATE_16K
        assert mic.channels == CHANNELS_MONO
        assert mic.format == FORMAT_S16_LE
        assert mic.chunk_size == CHUNK_BALANCED

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_custom_parameters_alsa(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test ALSA microphone with custom parameters."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device=0, sample_rate=48000, channels=2, format="S32_LE", chunk_size=2048)

        assert mic.sample_rate == 48000
        assert mic.channels == 2
        assert mic.format == "S32_LE"
        assert mic.chunk_size == 2048

    def test_custom_parameters_websocket(self):
        """Test WebSocket microphone with custom parameters."""
        mic = Microphone(device="ws://127.0.0.1:0", sample_rate=44100, channels=2, format="FLOAT_LE", chunk_size=512, timeout=5, secret="yolo")

        assert mic.sample_rate == 44100
        assert mic.channels == 2
        assert mic.format == "FLOAT_LE"
        assert mic.chunk_size == 512
        assert mic.timeout == 5
        assert mic.secret == "yolo"

    def test_unsupported_format_raises_error(self):
        """Test that unsupported format raises error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(device="hw:0,0", format="INVALID_FORMAT")

        with pytest.raises(MicrophoneConfigError):
            WebSocketMicrophone(port=0, format="INVALID_FORMAT")

    def test_invalid_port_raises_error(self):
        """Test that invalid port raises error."""
        with pytest.raises(MicrophoneConfigError):
            WebSocketMicrophone(port=-1)

        with pytest.raises(MicrophoneConfigError):
            WebSocketMicrophone(port=70000)

    def test_invalid_timeout_raises_error(self):
        """Test that invalid timeout raises error."""
        with pytest.raises(MicrophoneConfigError):
            WebSocketMicrophone(port=0, timeout=-5)

        with pytest.raises(MicrophoneConfigError):
            WebSocketMicrophone(port=0, timeout=0)


class TestALSAMicrophoneDeviceResolution:
    """Test ALSA device resolution."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_resolve_by_shorthand(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test resolving device by integer index."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        with patch("arduino.app_peripherals.microphone.alsa_microphone.Path.exists", return_value=True):
            with patch("arduino.app_peripherals.microphone.alsa_microphone.Path.resolve", return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c"):
                mic = ALSAMicrophone()
                assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

                mic = ALSAMicrophone(device=Microphone.USB_MIC_1)
                assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

                mic = ALSAMicrophone(device=Microphone.USB_MIC_2)
                assert mic.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_resolve_by_integer_index(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test resolving device by integer index."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = ALSAMicrophone(device=0)
        assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device=1)
        assert mic.device_stable_ref == "plughw:CARD=AnotherCard,DEV=0"

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=[])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=[])
    def test_resolve_no_usb_devices_raises_error(self, mock_pcms, mock_card_indexes, mock_cards):
        """Test that error is raised when no USB devices found."""
        with pytest.raises(MicrophoneConfigError) as exc_info:
            ALSAMicrophone(device=0)

        assert "No ALSA microphones found" in str(exc_info.value)

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_resolve_out_of_range_raises_error(self, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that out of range index raises error."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        with pytest.raises(MicrophoneConfigError) as exc_info:
            ALSAMicrophone(device=5)

        assert "out of range" in str(exc_info.value)

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_resolve_explicit_device_name(self, mock_pcms, mock_card_name):
        """Test that explicit device names are passed through."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = ALSAMicrophone(device="CARD=SomeCard,DEV=0")
        assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device="plughw:CARD=SomeCard,DEV=0")
        assert mic.device_stable_ref == "plughw:CARD=SomeCard,DEV=0"

        mic = ALSAMicrophone(device="hw:1,0")
        assert mic.device_stable_ref == "hw:CARD=AnotherCard,DEV=0"


class TestMicrophoneStartStop:
    """Test start and stop lifecycle."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_start_opens_device(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that start() opens ALSA device."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]
        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)

        assert not mic.is_started()

        mic.start()

        assert mic.is_started()
        assert mock_pcm.called
        assert pcm_instance.setchannels.called
        assert pcm_instance.setrate.called

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_stop_closes_device(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that stop() closes ALSA device."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]
        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)
        mic.start()
        mic.stop()

        assert not mic.is_started()
        assert pcm_instance.close.called

    @pytest.mark.asyncio
    async def test_websocket_start_creates_server(self):
        """Test that start() creates WebSocket server."""
        mic = Microphone(device="ws://127.0.0.1:0")

        try:
            mic.start()

            # Wait for server to start (with shorter intervals)
            for _ in range(100):
                if mic._server is not None:
                    break
                await asyncio.sleep(0.01)  # 10ms intervals

            assert mic.is_started()
            assert mic._server is not None
        finally:
            mic.stop()

    @pytest.mark.asyncio
    async def test_websocket_stop_closes_server(self):
        """Test that stop() closes WebSocket server."""
        mic = Microphone(device="ws://127.0.0.1:0")

        mic.start()
        # Wait for server to start (with shorter intervals)
        for _ in range(100):
            if mic._server is not None:
                break
            await asyncio.sleep(0.01)  # 10ms intervals

        mic.stop()
        await asyncio.sleep(0.05)  # Shorter wait for cleanup

        assert not mic.is_started()

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_double_start_is_idempotent(self, mock_pcms, mock_card_name):
        """Test that starting twice is safe."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="hw:0,0")

        mic.start = MagicMock()
        mic._is_started = False
        mic._mic_lock = threading.Lock()

        # Simulate idempotent behavior
        def start_impl():
            with mic._mic_lock:
                if mic._is_started:
                    return
                mic._is_started = True

        mic.start.side_effect = start_impl

        mic.start()
        first_state = mic._is_started
        mic.start()

        assert mic._is_started == first_state

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_double_stop_is_idempotent(self, mock_pcms, mock_card_name):
        """Test that stopping twice is safe."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="hw:0,0")

        mic._is_started = True
        mic._mic_lock = threading.Lock()
        mic.stop = MagicMock()

        def stop_impl():
            with mic._mic_lock:
                if not mic._is_started:
                    return
                mic._is_started = False

        mic.stop.side_effect = stop_impl

        mic.stop()
        mic.stop()  # Should not raise

        assert not mic._is_started


class TestMicrophoneContextManager:
    """Test context manager behavior."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_context_manager_starts_and_stops(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that context manager starts and stops microphone."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]
        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)

        assert not mic.is_started()

        with mic:
            assert mic.is_started()

        assert not mic.is_started()

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0, 1])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_context_manager_stops_on_exception(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that context manager stops even on exception."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]
        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        mic = Microphone(device=0)

        try:
            with mic:
                assert mic.is_started()
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        assert not mic.is_started()


class TestMicrophoneThreadSafety:
    """Test thread safety of initialization and state management."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_concurrent_start_stop(self, mock_pcms, mock_card_name):
        """Test concurrent start/stop operations are thread-safe."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="hw:0,0")
        errors = []

        # Mock the internal methods to avoid actual hardware access
        mic._open_microphone = MagicMock()
        mic._close_microphone = MagicMock()
        mic._read_audio = MagicMock(return_value=np.zeros(1024, dtype=np.int16))

        def start_stop_loop():
            try:
                for _ in range(10):
                    mic.start()
                    time.sleep(0.001)
                    mic.stop()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=start_stop_loop) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestALSAFormatMapping:
    """Test ALSA format mapping."""

    @pytest.mark.parametrize(
        "format_str,expected_dtype",
        [
            ("S8", np.int8),
            ("S16_LE", np.int16),
            ("S32_LE", np.int32),
            ("FLOAT_LE", np.float32),
        ],
    )
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_format_to_dtype_mapping(self, mock_pcms, mock_card_name, format_str, expected_dtype):
        """Test that formats are correctly mapped to numpy dtypes."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = ALSAMicrophone(format=format_str)

        assert mic._dtype == expected_dtype

    def test_unsupported_format_with_none_dtype(self):
        """Test that formats with no numpy dtype raise error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(format="MU_LAW")


class TestWebSocketURLParsing:
    """Test WebSocket URL parsing."""

    def test_ignore_host(self):
        """Test parsing hosts."""
        mic = Microphone(device="ws://0.0.0.0")
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

        mic = Microphone(device=f"ws://192.168.1.1")  # Overwritten
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

        mic = Microphone(device="ws://127.0.0.1")  # Overwritten
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

        mic = Microphone(device="ws://localhost")  # Overwritten
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

        mic = Microphone(device="ws://example.com")  # Overwritten
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

    def test_parse_port(self):
        """Test parsing ports."""
        mic = Microphone(device="ws://0.0.0.0")
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.port == 8080  # Default port

        mic = Microphone(device="ws://0.0.0.0:9876")
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.port == 9876

        mic = Microphone(device="ws://0.0.0.0:0")
        assert isinstance(mic, WebSocketMicrophone)
        mic.start()  # Bind to any available port
        assert mic.port is not 0
        mic.stop()


class TestBaseMicrophoneAbstraction:
    """Test base microphone abstract class requirements."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseMicrophone cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMicrophone()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""

        # Missing _read_audio
        class IncompleteMic1(BaseMicrophone):
            def _open_microphone(self):
                pass

            def _close_microphone(self):
                pass

        with pytest.raises(TypeError):
            IncompleteMic1()

        # Missing _close_microphone
        class IncompleteMic2(BaseMicrophone):
            def _open_microphone(self):
                pass

            def _read_audio(self):
                pass

        with pytest.raises(TypeError):
            IncompleteMic2()

        # Missing _open_microphone
        class IncompleteMic3(BaseMicrophone):
            def _close_microphone(self):
                pass

            def _read_audio(self):
                pass

        with pytest.raises(TypeError):
            IncompleteMic3()


class TestMicrophoneInitialState:
    """Test initial state of microphones."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    def test_microphone_starts_in_stopped_state(self, mock_pcms, mock_card_name):
        """Test that microphones start in stopped state."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        mic = Microphone(device="hw:0,0")

        assert not mic.is_started()

    def test_websocket_microphone_initial_state(self):
        """Test WebSocket microphone initial state."""
        mic = Microphone(device="ws://localhost:0")

        assert isinstance(mic, WebSocketMicrophone)
        assert mic.port == 0
        assert not mic.is_started()
        assert mic._server is None
        assert mic._client is None
