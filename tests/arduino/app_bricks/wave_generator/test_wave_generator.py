# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch
import pytest
import time
import threading

import numpy as np

from arduino.app_bricks.wave_generator import WaveGenerator
from arduino.app_utils import AppController
from arduino.app_peripherals.speaker import BaseSpeaker
import arduino.app_utils.app as app


# Mock ALSA for factory tests
MOCK_CARDS = ["SomeCard"]
MOCK_CARD_INDEXES = [i for i in range(len(MOCK_CARDS))]
MOCK_PCMS = ["plughw:CARD=SomeCard,DEV=0"]


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


@pytest.fixture(autouse=True)
def mock_speaker(monkeypatch):
    """Mock Speaker to avoid hardware dependencies."""

    class FakeSpeaker(BaseSpeaker):
        def __init__(self):
            super().__init__(sample_rate=16000, channels=1, format=np.float32, buffer_size=2048, auto_reconnect=False)
            self.device = "fake_device"
            self.running = threading.Event()
            self.written_audio = []

        def _open_speaker(self):
            self.running.set()

        def _close_speaker(self):
            self.running.clear()

        def _write_audio(self, audio_chunk: np.ndarray):
            if self.running.is_set():
                self.written_audio.append(audio_chunk)

    # # Patch Speaker in the wave_generator module
    # monkeypatch.setattr("arduino.app_peripherals.speaker.ALSASpeaker", FakeSpeaker)

    return FakeSpeaker()


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_PCMS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM")
@patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists", return_value=True)
@patch(
    "arduino.app_peripherals.speaker.alsa_speaker.Path.resolve",
    return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c",
)
class TestWaveGeneratorInit:
    """Test suite for WaveGenerator brick."""

    def test_default_init(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test WaveGenerator initializes with default parameters."""
        wave_gen = WaveGenerator()

        assert wave_gen.wave_type == "sine"
        assert wave_gen.attack == 0.01
        assert wave_gen.release == 0.03
        assert wave_gen.glide == 0.02
        assert wave_gen._speaker is not None
        assert wave_gen._speaker.sample_rate == 48000

    def test_custom_init(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test WaveGenerator initializes with custom parameters."""
        wave_gen = WaveGenerator(
            wave_type="square",
            attack=0.02,
            release=0.05,
            glide=0.03,
        )

        assert wave_gen.wave_type == "square"
        assert wave_gen.attack == 0.02
        assert wave_gen.release == 0.05
        assert wave_gen.glide == 0.03
        assert wave_gen._speaker.sample_rate == 48000

    def test_init_with_custom_speaker(
        self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards, mock_speaker
    ):
        """Test WaveGenerator with externally provided Speaker."""
        speaker = mock_speaker
        wave_gen = WaveGenerator(speaker=speaker)

        assert wave_gen._speaker is speaker

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM_FORMAT_FLOAT_LE", new=1)
    def test_start_stop(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test WaveGenerator start and stop methods."""
        pcm_instance = MagicMock()
        pcm_instance.info.return_value = {
            "rate": 48000,
            "channels": 1,
            "format": 1,
            "format_name": "FLOAT_LE",
            "period_size": 256,
        }
        mock_pcm.return_value = pcm_instance

        wave_gen = WaveGenerator()

        assert not wave_gen._running.is_set()

        wave_gen.start()
        assert wave_gen._running.is_set()
        assert wave_gen._speaker.is_started()

        time.sleep(0.1)

        wave_gen.stop()
        assert not wave_gen._running.is_set()
        assert not wave_gen._speaker.is_started()

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM_FORMAT_FLOAT_LE", new=1)
    def test_multiple_start_stop(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test starting and stopping multiple times."""
        pcm_instance = MagicMock()
        pcm_instance.info.return_value = {
            "rate": 48000,
            "channels": 1,
            "format": 1,
            "format_name": "FLOAT_LE",
            "period_size": 256,
        }
        mock_pcm.return_value = pcm_instance

        wave_gen = WaveGenerator()

        for _ in range(3):
            wave_gen.start()
            assert wave_gen._running.is_set()
            time.sleep(0.05)

            wave_gen.stop()
            assert not wave_gen._running.is_set()
            time.sleep(0.05)

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM_FORMAT_FLOAT_LE", new=1)
    def test_double_start(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that starting an already running generator logs a warning."""
        pcm_instance = MagicMock()
        pcm_instance.info.return_value = {
            "rate": 48000,
            "channels": 1,
            "format": 1,
            "format_name": "FLOAT_LE",
            "period_size": 256,
        }
        mock_pcm.return_value = pcm_instance

        wave_gen = WaveGenerator()

        wave_gen.start()
        assert wave_gen._running.is_set()

        # Should not crash when starting a second time
        wave_gen.start()
        assert wave_gen._running.is_set()

        wave_gen.stop()

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM_FORMAT_FLOAT_LE", new=1)
    def test_double_stop(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test that stopping a non-running generator logs a warning."""
        pcm_instance = MagicMock()
        pcm_instance.info.return_value = {
            "rate": 48000,
            "channels": 1,
            "format": 1,
            "format_name": "FLOAT_LE",
            "period_size": 256,
        }
        mock_pcm.return_value = pcm_instance

        wave_gen = WaveGenerator()

        # Should not crash when stopping before starting
        wave_gen.stop()
        assert not wave_gen._running.is_set()

    @patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM_FORMAT_FLOAT_LE", new=1)
    def test_app_integration(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards, app_instance):
        """Test integration with AppController (start/stop via App)."""
        pcm_instance = MagicMock()
        pcm_instance.info.return_value = {
            "rate": 48000,
            "channels": 1,
            "format": 1,
            "format_name": "FLOAT_LE",
            "period_size": 256,
        }
        mock_pcm.return_value = pcm_instance

        wave_gen = WaveGenerator()
        pcm_instance.write.return_value = wave_gen._speaker.buffer_size

        # Register manually to avoid auto-registration
        app_instance.start_brick(wave_gen)

        assert wave_gen._running.is_set()
        assert wave_gen._speaker.is_started()

        time.sleep(0.1)

        app_instance.stop_brick(wave_gen)

        assert not wave_gen._running.is_set()
        assert not wave_gen._speaker.is_started()


@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.cards", return_value=MOCK_CARDS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_indexes", return_value=MOCK_CARD_INDEXES)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.card_name", side_effect=lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"])
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.pcms", return_value=MOCK_PCMS)
@patch("arduino.app_peripherals.speaker.alsa_speaker.alsaaudio.PCM")
@patch("arduino.app_peripherals.speaker.alsa_speaker.Path.exists", return_value=True)
@patch(
    "arduino.app_peripherals.speaker.alsa_speaker.Path.resolve",
    return_value="/sys/devices/platform/soc@0/4ef8800.usb/4e00000.usb/xhci-hcd.2.auto/usb1/1-1/1-1.3/1-1.3:1.0/sound/card0/pcmC0D0c",
)
class TestWaveGeneratorGetterSetters:
    def test_get_set_frequency(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting frequency."""
        wave_gen = WaveGenerator()

        assert wave_gen.frequency == 440.0  # Default frequency

        wave_gen.frequency = 880.0
        assert wave_gen.frequency == 880.0

        # Test range
        with pytest.raises(ValueError):
            wave_gen.frequency = -100.0
        assert wave_gen.frequency == 880.0

    def test_get_set_amplitude(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting amplitude."""
        wave_gen = WaveGenerator()

        assert wave_gen.amplitude == 0.0  # Default amplitude

        wave_gen.amplitude = 0.5
        assert wave_gen.amplitude == 0.5

        wave_gen.amplitude = 1.0
        assert wave_gen.amplitude == 1.0

        # Test range
        with pytest.raises(ValueError):
            wave_gen.amplitude = 1.5
        assert wave_gen.amplitude == 1.0

        with pytest.raises(ValueError):
            wave_gen.amplitude = -0.5
        assert wave_gen.amplitude == 1.0

    def test_get_set_wave_type(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting wave type."""
        wave_gen = WaveGenerator()

        wave_gen.wave_type = "sine"
        assert wave_gen.wave_type == "sine"

        wave_gen.wave_type = "square"
        assert wave_gen.wave_type == "square"

        wave_gen.wave_type = "sawtooth"
        assert wave_gen.wave_type == "sawtooth"

        wave_gen.wave_type = "triangle"
        assert wave_gen.wave_type == "triangle"

        # Test invalid wave type
        with pytest.raises(ValueError):
            wave_gen.wave_type = "invalid"  # type: ignore

    def test_get_set_volume(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting hardware volume."""
        wave_gen = WaveGenerator()
        assert wave_gen.volume == 100  # Default volume

        wave_gen.volume = 70
        assert wave_gen.volume == 70
        assert wave_gen._speaker.volume == 70

        wave_gen.volume = 100
        assert wave_gen.volume == 100
        assert wave_gen._speaker.volume == 100

    def test_get_set_envelope_params(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test setting envelope parameters."""
        wave_gen = WaveGenerator()

        wave_gen.attack = 0.05
        assert wave_gen.attack == 0.05

        wave_gen.release = 0.1
        assert wave_gen.release == 0.1

        wave_gen.glide = 0.04
        assert wave_gen.glide == 0.04

        # Test range
        with pytest.raises(ValueError):
            wave_gen.attack = -0.01
        assert wave_gen.attack == 0.05

        with pytest.raises(ValueError):
            wave_gen.release = -0.02
        assert wave_gen.release == 0.1

        with pytest.raises(ValueError):
            wave_gen.glide = -0.03
        assert wave_gen.glide == 0.04

    def test_get_state(self, mock_resolve, mock_exists, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test getting current generator state."""
        wave_gen = WaveGenerator()

        wave_gen.wave_type = "square"
        wave_gen.frequency = 440.0
        wave_gen.amplitude = 0.8
        wave_gen.volume = 90

        state = wave_gen.state

        assert "amplitude" in state
        assert "frequency" in state
        assert "wave_type" in state
        assert state["wave_type"] == "square"
        assert "attack" in state
        assert "release" in state
        assert "glide" in state
        assert "volume" in state
        assert state["volume"] == 90


class TestWaveGeneratorAudioGeneration:
    """Test suite for WaveGenerator's audio generation features."""

    @pytest.mark.parametrize("wave_type", ["sine", "square", "sawtooth", "triangle"])
    def test_generate_shape(self, mock_speaker, wave_type):
        """Test generating a sine wave block."""
        wave_gen = WaveGenerator(speaker=mock_speaker)
        wave_gen.amplitude = 0.5
        wave_gen.wave_type = wave_type

        block = wave_gen._generate_audio_block()

        assert wave_gen.wave_type == wave_type
        assert isinstance(block, np.ndarray)
        assert block.dtype == np.float32
        assert len(block) == wave_gen._block_frame_count
        assert np.max(np.abs(block)) <= 0.5

    def test_phase_tracking(self, mock_speaker):
        """Test that phase is tracked across blocks."""
        wave_gen = WaveGenerator(speaker=mock_speaker)
        wave_gen.amplitude = 1.0

        initial_phase = wave_gen._prev_phase

        # Generate multiple blocks
        for _ in range(10):
            wave_gen._generate_audio_block()

        # Phase should have advanced
        assert wave_gen._prev_phase != initial_phase
        # Phase should be wrapped to [0, 2π]
        assert 0.0 <= wave_gen._prev_phase < 2 * np.pi

    def test_zero_amplitude_produces_silence(self, mock_speaker):
        """Test that zero amplitude produces silent output."""
        wave_gen = WaveGenerator(speaker=mock_speaker)

        block = wave_gen._generate_audio_block()

        # All samples should be zero or very close to zero
        assert np.allclose(block, 0.0, atol=1e-6)

    def test_frequency_glide(self, mock_speaker):
        """Test frequency glide behavior"""
        # Each block generated will have a duration of 32ms @ 16kHz
        wave_gen = WaveGenerator(speaker=mock_speaker, glide=0.064)
        wave_gen.amplitude = 0.5

        wave_gen.frequency = 220.0  # Start at 220 Hz
        wave_gen._generate_audio_block()
        assert wave_gen._prev_frequency == 330.0
        wave_gen._generate_audio_block()
        assert wave_gen._prev_frequency == 220.0

        wave_gen.frequency = 880.0  # Jump to 880 Hz
        wave_gen._generate_audio_block()
        assert wave_gen._prev_frequency == 550.0
        wave_gen._generate_audio_block()
        assert wave_gen._prev_frequency == 880.0

    def test_amplitude_attack(self, mock_speaker):
        """Test amplitude envelope attack."""
        # Each block generated will have a duration of 32ms @ 16kHz
        wave_gen = WaveGenerator(speaker=mock_speaker, attack=0.064, release=0.0)

        wave_gen.amplitude = 0.0  # Initial amplitude
        block = wave_gen._generate_audio_block()
        assert np.allclose(block, 0.0, atol=1e-6)

        wave_gen.amplitude = 1.0  # Target amplitude, reached after 2 blocks
        block = wave_gen._generate_audio_block()  # First block: 0.0 -> 0.5
        assert np.all(-0.5 <= block) and np.all(block <= 0.5)

        block = wave_gen._generate_audio_block()  # Second block: 0.5 -> 1.0
        assert np.all(-1 <= block) and np.all(block <= 1)
        block = wave_gen._generate_audio_block()  # Third block: 1.0
        assert np.all(-1 <= block) and np.all(block <= 1)

    def test_amplitude_release(self, mock_speaker):
        """Test amplitude envelope release."""
        # Each block generated will have a duration of 32ms @ 16kHz
        wave_gen = WaveGenerator(speaker=mock_speaker, attack=0.0, release=0.064)

        wave_gen.amplitude = 1.0  # Initial amplitude, reached instantaneously
        block = wave_gen._generate_audio_block()
        assert np.all(-1 <= block) and np.all(block <= 1)

        wave_gen.amplitude = 0  # Target amplitude, reached after 2 blocks
        block = wave_gen._generate_audio_block()  # First block: 1.0 -> 0.5
        assert np.all(-1 <= block) and np.all(block <= 1)
        block = wave_gen._generate_audio_block()  # Second block: 0.5 -> 0.0
        assert np.all(-0.5 <= block) and np.all(block <= 0.5)

        block = wave_gen._generate_audio_block()  # Third block: 0.0
        assert np.allclose(block, 0.0, atol=1e-6)

    def test_buffer_preallocation(self, mock_speaker):
        """Test that buffers are pre-allocated and reused."""
        wave_gen = WaveGenerator(speaker=mock_speaker)

        # Generate first block
        wave_gen._generate_audio_block()

        # Check buffers are allocated
        assert wave_gen._ramp_vec is not None
        assert wave_gen._buf_phases is not None
        assert wave_gen._buf_envelope is not None
        assert wave_gen._buf_samples is not None

        # Store references
        ramp_vec_ref = wave_gen._ramp_vec
        buf_phases_ref = wave_gen._buf_phases
        buf_envelope_ref = wave_gen._buf_envelope
        buf_samples_ref = wave_gen._buf_samples

        # Generate second block
        wave_gen._generate_audio_block()

        # Verify buffers are the same objects (not reallocated)
        assert wave_gen._ramp_vec is ramp_vec_ref
        assert wave_gen._buf_phases is buf_phases_ref
        assert wave_gen._buf_envelope is buf_envelope_ref
        assert wave_gen._buf_samples is buf_samples_ref

    def test_producer_loop_generates_audio(self, app_instance, mock_speaker):
        """Test that producer loop generates and plays audio."""
        speaker = mock_speaker

        wave_gen = WaveGenerator(speaker=speaker)
        wave_gen.amplitude = 0.5

        app_instance.start_brick(wave_gen)

        # Asynchronously stop after a short delay
        stop_done = threading.Event()
        threading.Timer(0.2, lambda: (app_instance.stop_brick(wave_gen), stop_done.set())).start()
        stop_done.wait()

        # Check that audio was played
        assert len(speaker.written_audio) > 0
