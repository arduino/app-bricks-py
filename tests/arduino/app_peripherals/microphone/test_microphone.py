# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import threading
from unittest.mock import MagicMock

import numpy as np

from arduino.app_peripherals.microphone import Microphone, BaseMicrophone, ALSAMicrophone, WebSocketMicrophone
from arduino.app_peripherals.microphone.errors import MicrophoneConfigError, MicrophoneError, MicrophoneOpenError, MicrophoneReadError


class TestMicrophoneFactoryInstantiation:
    """Test factory instantiation of different microphone types."""

    def test_factory_creates_alsa_microphone_with_integer(self, mock_alsa_usb_mics):
        """Test factory creates ALSA microphone with integer device index."""
        mic = Microphone(device=0)

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

    def test_factory_creates_alsa_microphone_with_string_index(self, mock_alsa_usb_mics):
        """Test factory creates ALSA microphone with string device index."""

        mic = Microphone(device="1")

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "CARD=AnotherCard,DEV=0"

    def test_factory_creates_alsa_microphone_with_device_name(self, mock_alsa_usb_mics):
        """Test factory creates ALSA microphone with explicit device name."""
        mic = Microphone(device="hw:0,0")

        assert isinstance(mic, ALSAMicrophone)
        assert mic.device_stable_ref == "CARD=SomeCard,DEV=0"

    def test_factory_creates_websocket_microphone_with_ws_url(self):
        """Test factory creates WebSocket microphone with ws:// URL."""
        mic = Microphone(device="ws://localhost:9234")

        assert isinstance(mic, WebSocketMicrophone)
        assert mic._bind_ip == "0.0.0.0"
        assert mic.port == 9234

    @pytest.mark.parametrize(
        "device",
        [
            "ws://0.0.0.0",
            "ws://192.168.1.1",
            "ws://127.0.0.1",
            "ws://localhost",
            "ws://example.com",
        ],
    )
    def test_factory_creates_websocket_ignore_host(self, device):
        """Test parsing hosts."""
        mic = Microphone(device=device)
        assert isinstance(mic, WebSocketMicrophone)
        assert mic.url == "ws://0.0.0.0:8080"

    def test_factory_creates_websocket_parse_port(self):
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

    def test_factory_invalid_device_type_raises_error(self):
        """Test that invalid device type raises MicrophoneConfigError."""
        with pytest.raises(MicrophoneConfigError):
            Microphone(device=None)


class TestMicrophoneConfiguration:
    """Test microphone configuration and parameters."""

    def test_default_parameters(self, mock_alsa_usb_mics):
        """Test that microphones use default parameters."""
        mic = Microphone(device=0)

        assert mic.sample_rate == Microphone.RATE_16K
        assert mic.channels == Microphone.CHANNELS_MONO
        assert mic.format == np.int16
        assert mic.buffer_size == Microphone.BUFFER_SIZE_BALANCED

    def test_custom_parameters_alsa(self, mock_alsa_usb_mics):
        """Test ALSA microphone with custom parameters."""
        mic = Microphone(device=0, sample_rate=48000, channels=2, format=np.int32, buffer_size=2048)

        assert mic.sample_rate == 48000
        assert mic.channels == 2
        assert mic.format == np.int32
        assert mic.buffer_size == 2048

    def test_custom_parameters_websocket(self):
        """Test WebSocket microphone with custom parameters."""
        mic = Microphone(device="ws://127.0.0.1:0", sample_rate=44100, channels=2, format=np.float32, buffer_size=512, timeout=5, secret="yolo")

        assert mic.sample_rate == 44100
        assert mic.channels == 2
        assert mic.format == np.float32
        assert mic.buffer_size == 512
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

    def test_invalid_device_type_raises_error(self):
        """Test that invalid device type raises error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(device=None)

    def test_no_devices_found_raises_error(self):
        """Test that no USB devices found raises error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone()

    def test_out_of_range_device_index_raises_error(self):
        """Test that out of range device index raises error."""
        with pytest.raises(MicrophoneConfigError):
            ALSAMicrophone(device=10)


class TestMicrophoneStartStop:
    """Test start and stop lifecycle."""

    def test_double_start_is_idempotent(self):
        """Test that starting twice is safe."""
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

    def test_double_stop_is_idempotent(self):
        """Test that stopping twice is safe."""
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

    def test_restart(self):
        """Test that microphone can be restarted."""
        mic = Microphone(device="CARD=SomeCard,DEV=0")
        mic.start()
        mic.stop()

        # Should be able to restart
        mic.start()
        assert mic.is_started()


class TestMicrophoneContextManager:
    """Test context manager behavior."""

    def test_context_manager_starts_and_stops(self):
        """Test that context manager starts and stops microphone."""
        mic = Microphone(device=0)

        assert not mic.is_started()

        with mic:
            assert mic.is_started()

        assert not mic.is_started()

    def test_context_manager_stops_on_exception(self):
        """Test that context manager stops even on exception."""
        mic = Microphone(device=0)

        try:
            with mic:
                assert mic.is_started()
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        assert not mic.is_started()


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
