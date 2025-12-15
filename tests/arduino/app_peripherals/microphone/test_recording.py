# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import io
import numpy as np
import wave
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.microphone import Microphone, BaseMicrophone
from arduino.app_peripherals.microphone.errors import MicrophoneReadError


class MockMicrophone(BaseMicrophone):
    """Mock microphone for testing recording functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk_counter = 0

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        self._chunk_counter += 1
        # Return incrementing values for verification
        return np.full(self.chunk_size, self._chunk_counter, dtype=np.int16)


# Mock ALSA for factory tests
MOCK_CARDS = ["SomeCard"]
MOCK_PCMS = ["plughw:CARD=SomeCard,DEV=0"]


class TestRecordDuration:
    """Test recording for specific durations."""

    def test_record_returns_numpy_array(self):
        """Test that record returns a numpy array (raw PCM)."""
        mic = MockMicrophone(sample_rate=16000, chunk_size=1024)
        mic.start()

        recording = mic.record(duration=0.1)

        assert isinstance(recording, np.ndarray)
        assert len(recording) > 0
        # record() returns raw PCM data with original dtype
        assert recording.dtype == np.int16

    def test_record_wav_returns_numpy_array(self):
        """Test that record_wav returns a numpy array (WAV format)."""
        mic = MockMicrophone(sample_rate=16000, chunk_size=1024)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        assert isinstance(wav_data, np.ndarray)
        assert len(wav_data) > 0
        # WAV data should be uint8 bytes
        assert wav_data.dtype == np.uint8

    def test_record_approximate_duration(self):
        """Test that record returns approximately correct number of samples."""
        sample_rate = 16000
        chunk_size = 1024
        duration = 0.2

        mic = MockMicrophone(sample_rate=sample_rate, chunk_size=chunk_size)
        mic.start()

        recording = mic.record(duration=duration)

        expected_samples = int(duration * sample_rate)

        assert len(recording) == expected_samples

    def test_record_returns_exact_sample_count(self):
        """Test that record returns exactly the requested number of samples."""
        mic = MockMicrophone(sample_rate=16000)
        mic.start()

        duration = 0.15
        recording = mic.record(duration=duration)

        # Should return exactly the requested number of samples
        expected_samples = int(duration * 16000)
        assert len(recording) == expected_samples

    def test_record_with_different_sample_rates(self):
        """Test recording with different sample rates."""
        test_cases = [
            (8000, 0.1),
            (16000, 0.1),
            (44100, 0.1),
            (48000, 0.1),
        ]

        for sample_rate, duration in test_cases:
            mic = MockMicrophone(sample_rate=sample_rate, chunk_size=512)
            mic.start()

            recording = mic.record(duration=duration)

            expected_samples = int(duration * sample_rate)

            assert len(recording) == expected_samples

    def test_record_short_duration(self):
        """Test recording very short duration."""
        cs = 1024
        mic = MockMicrophone(sample_rate=16000, chunk_size=cs)
        mic.start()

        recording = mic.record(duration=0.05)

        assert len(recording) > 0
        assert len(recording) < cs * 2  # Should be relatively short

    def test_record_validates_positive_duration(self):
        """Test that record validates positive duration."""
        mic = MockMicrophone()
        mic.start()

        with pytest.raises(ValueError) as exc_info:
            mic.record(duration=0)

        assert "positive" in str(exc_info.value).lower()

    def test_record_negative_duration_raises_error(self):
        """Test that negative duration raises error."""
        mic = MockMicrophone()
        mic.start()

        with pytest.raises(ValueError):
            mic.record(duration=-1)

    def test_record_requires_started_microphone(self):
        """Test that record requires microphone to be started."""
        mic = MockMicrophone()

        with pytest.raises(MicrophoneReadError) as exc_info:
            mic.record(duration=1.0)

        assert "start" in str(exc_info.value)

    def test_record_concatenates_chunks_correctly(self):
        """Test that record correctly concatenates audio chunks."""
        mic = MockMicrophone(chunk_size=100)
        mic.start()

        recording = mic.record(duration=0.05)

        # Each chunk has incrementing fill values (1, 2, 3, ...)
        # Verify we got multiple chunks
        assert len(recording) >= 100


class TestWAVFileValidation:
    """Test WAV format validation using wave module."""

    def test_wav_has_correct_sample_rate(self):
        """Test that WAV data has correct sample rate."""
        sample_rate = 16000
        mic = MockMicrophone(sample_rate=sample_rate, channels=1)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        # Parse WAV data using wave module
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getframerate() == sample_rate

    def test_wav_has_correct_channels(self):
        """Test that WAV data has correct number of channels."""
        mic = MockMicrophone(sample_rate=16000, channels=1)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getnchannels() == 1

    def test_wav_has_correct_sample_width(self):
        """Test that WAV data has correct sample width."""
        mic = MockMicrophone(sample_rate=16000)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            # int16 = 2 bytes sample width
            assert wav.getsampwidth() == 2

    def test_wav_has_correct_number_of_frames(self):
        """Test that WAV data has approximately correct number of frames."""
        sample_rate = 16000
        duration = 0.2
        mic = MockMicrophone(sample_rate=sample_rate)
        mic.start()

        wav_data = mic.record_wav(duration=duration)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            nframes = wav.getnframes()
            expected_frames = int(sample_rate * duration)

            assert nframes == expected_frames

    def test_wav_is_readable(self):
        """Test that WAV data can be read and decoded."""
        mic = MockMicrophone(sample_rate=16000)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)

            assert len(audio_data) > 0
            assert audio_data.dtype == np.int16

    def test_wav_contains_valid_audio_data(self):
        """Test that WAV data contains valid audio data."""
        mic = MockMicrophone(sample_rate=16000, chunk_size=100)
        mic.start()

        wav_data = mic.record_wav(duration=0.05)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # Mock mic produces incrementing values (1, 2, 3, ...)
            # Check that we have non-zero data
            assert np.any(audio_data != 0)

    def test_wav_can_be_saved_to_file(self):
        """Test that WAV data can be saved to a file."""
        mic = MockMicrophone(sample_rate=16000)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            output_path.write_bytes(wav_data.tobytes())

            assert output_path.exists()

            # Verify it's a valid WAV file
            with wave.open(str(output_path), "rb") as wav:
                assert wav.getframerate() == 16000
                assert wav.getnchannels() == 1


class TestWAVFileFormats:
    """Test WAV format creation with different audio formats."""

    def test_audio_to_wav_int16_audio(self):
        """Test converting int16 audio data to WAV."""
        mic = MockMicrophone()

        audio = np.arange(1000, dtype=np.int16)
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

        # Validate with wave module
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 2
            frames = wav.readframes(wav.getnframes())
            loaded = np.frombuffer(frames, dtype=np.int16)
            np.testing.assert_array_equal(audio, loaded)

    def test_audio_to_wav_int32_audio(self):
        """Test converting int32 audio data to WAV."""
        mic = MockMicrophone()

        audio = np.arange(1000, dtype=np.int32)
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 4

    def test_audio_to_wav_float32_converts_to_int16(self):
        """Test that float32 audio is converted to int16 in WAV."""
        mic = MockMicrophone()

        # Normalized float audio [-1, 1]
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_data = mic._audio_to_wav(audio)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            # Should be converted to int16
            assert wav.getsampwidth() == 2

    def test_audio_to_wav_int8_audio(self):
        """Test converting int8 audio data to WAV."""
        mic = MockMicrophone()

        audio = np.arange(-128, 128, dtype=np.int8)
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

    def test_audio_to_wav_uint8_audio(self):
        """Test converting uint8 audio data to WAV."""
        mic = MockMicrophone()

        audio = np.arange(0, 256, dtype=np.uint8)
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

    def test_audio_to_wav_s24_le_format(self):
        """Test converting S24_LE (24-bit little-endian) audio data to WAV."""
        mic = MockMicrophone(format="S24_LE")

        # Create 24-bit audio packed in 32-bit containers (LSB padding)
        # Values: 0x00123456, 0x00789ABC, 0x00FEDCBA (with LSB padding byte 0)
        audio = np.array([0x00123456, 0x00789ABC, 0x00FEDCBA], dtype="<i4")
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

        # Validate with wave module - should be 3-byte sample width
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 3
            # Read back the raw bytes
            frames = wav.readframes(wav.getnframes())
            # Each sample should be 3 bytes
            assert len(frames) == 9  # 3 samples * 3 bytes

    def test_audio_to_wav_s24_be_format(self):
        """Test converting S24_BE (24-bit big-endian) audio data to WAV."""
        mic = MockMicrophone(format="S24_BE")

        # Create 24-bit audio packed in 32-bit big-endian containers (LSB padding)
        # Using valid signed int32 values
        audio = np.array([0x12345600, 0x0789AB00, -0x01234600], dtype=">i4")
        wav_data = mic._audio_to_wav(audio)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

        # Validate with wave module - should be 3-byte sample width
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 3
            # Read back the raw bytes
            frames = wav.readframes(wav.getnframes())
            # Each sample should be 3 bytes
            assert len(frames) == 9  # 3 samples * 3 bytes

    def test_audio_to_wav_s24_le_preserves_values(self):
        """Test that S24_LE conversion preserves 24-bit audio values."""
        mic = MockMicrophone(format="S24_LE")

        # S24_LE format: 24-bit audio in the 3 MSB of 32-bit LE container, padding in LSB
        audio = np.array(
            [
                0x11000000,  # Stored as: [0x00, 0x00, 0x00, 0x11] -> extract significant bytes 1-3: [0x00, 0x00, 0x11]
                0x00110000,  # Stored as: [0x00, 0x00, 0x11, 0x00] -> extract significant bytes 1-3: [0x00, 0x11, 0x00]
                0x00001100,  # Stored as: [0x00, 0x11, 0x00, 0x00] -> extract significant bytes 1-3: [0x11, 0x00, 0x00]
            ],
            dtype="<i4",
        )

        wav_data = mic._audio_to_wav(audio)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 3
            frames = wav.readframes(wav.getnframes())

            # WAV is little-endian, bytes are extracted from positions 1-3 and not swapped
            # Sample 0: extract significant bytes => [0x00, 0x00, 0x11]
            assert frames[0:3] == b"\x00\x00\x11"
            # Sample 1: extract significant bytes => [0x00, 0x11, 0x00]
            assert frames[3:6] == b"\x00\x11\x00"
            # Sample 2: extract significant bytes => [0x11, 0x00, 0x00]
            assert frames[6:9] == b"\x11\x00\x00"

    def test_audio_to_wav_s24_be_preserves_values(self):
        """Test that S24_BE conversion preserves 24-bit audio values."""
        mic = MockMicrophone(format="S24_BE")

        # S24_BE format: 24-bit audio in the 3 MSB of 32-bit BE container, padding in LSB
        audio = np.array(
            [
                0x11000000,  # Stored in BE as: [0x11, 0x00, 0x00, 0x00] -> extract significant bytes 0-2: [0x11, 0x00, 0x00]
                0x00110000,  # Stored in BE as: [0x00, 0x11, 0x00, 0x00] -> extract significant bytes 0-2: [0x00, 0x11, 0x00]
                0x00001100,  # Stored in BE as: [0x00, 0x00, 0x11, 0x00] -> extract significant bytes 0-2: [0x00, 0x00, 0x11]
            ],
            dtype=">i4",
        )

        wav_data = mic._audio_to_wav(audio)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getsampwidth() == 3
            frames = wav.readframes(wav.getnframes())

            # WAV is little-endian, bytes are extracted from positions 0-2 and swapped
            # Sample 0: extract significant bytes and swap => [0x00, 0x00, 0x11]
            assert frames[0:3] == b"\x00\x00\x11"
            # Sample 1: extract significant bytes and swap => [0x00, 0x11, 0x00]
            assert frames[3:6] == b"\x00\x11\x00"
            # Sample 2: extract significant bytes and swap => [0x11, 0x00, 0x00]
            assert frames[6:9] == b"\x11\x00\x00"


class TestWAVFileFloatConversion:
    """Test float audio conversion in WAV format."""

    def test_float_values_are_scaled_correctly(self):
        """Test that float values are scaled to int16 range."""
        mic = MockMicrophone()

        # Test known values
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_data = mic._audio_to_wav(audio)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            loaded = np.frombuffer(frames, dtype=np.int16)

            # Check scaling (0.5 * 32767 ≈ 16383)
            assert abs(loaded[1] - 16383) < 10
            assert abs(loaded[2] - (-16383)) < 10

    def test_float_clipping(self):
        """Test that float values outside [-1, 1] are clipped."""
        mic = MockMicrophone()

        # Values outside valid range
        audio = np.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=np.float32)
        wav_data = mic._audio_to_wav(audio)

        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            loaded = np.frombuffer(frames, dtype=np.int16)

            # First and last should be clipped to max/min
            assert loaded[0] == -32767  # Clipped from -2.0
            assert loaded[-1] == 32767  # Clipped from 2.0


class TestRecordingWithRealMicrophone:
    """Test recording with real microphone implementations (mocked hardware)."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_microphone_record(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test recording with ALSA microphone (raw PCM)."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Mock audio data
        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        mic = Microphone(device=0)
        mic.start()

        recording = mic.record(duration=0.1)

        assert isinstance(recording, np.ndarray)
        assert len(recording) > 0
        assert recording.dtype == np.int16

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_PCMS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_microphone_record_wav(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test recording WAV format with ALSA microphone."""
        mock_card_name.side_effect = lambda idx: [MOCK_CARDS[idx], f"USB Audio Device {idx}"]

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        mic = Microphone(device=0)
        mic.start()

        wav_data = mic.record_wav(duration=0.1)

        assert isinstance(wav_data, np.ndarray)
        assert wav_data.dtype == np.uint8

        # Validate WAV format
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1


class TestRecordingBufferManagement:
    """Test buffer management during recording."""

    def test_record_preallocates_buffer(self):
        """Test that record preallocates buffer for efficiency."""
        mic = MockMicrophone(sample_rate=16000, chunk_size=1024)
        mic.start()

        recording = mic.record(duration=0.1)

        # Should complete without errors
        assert isinstance(recording, np.ndarray)

    def test_record_timeout_when_insufficient_data(self):
        """Test that record times out when microphone stops producing data."""

        class LimitedMockMic(MockMicrophone):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_chunks = 2

            def _read_audio(self):
                if self._chunk_counter >= self.max_chunks:
                    return None
                return super()._read_audio()

        mic = LimitedMockMic(sample_rate=16000, chunk_size=1024)
        mic.start()

        # Request longer duration than available - should timeout
        with pytest.raises(TimeoutError) as exc_info:
            mic.record(duration=1.0, timeout_factor=0.5)

        # Error should mention the timeout
        assert "timeout" in str(exc_info.value).lower()

    def test_record_handles_partial_chunks(self):
        """Test that record handles partial last chunk correctly."""
        mic = MockMicrophone(sample_rate=16000, chunk_size=1000)
        mic.start()

        # Very short duration that won't align with chunk boundaries
        recording = mic.record(duration=0.01)

        assert len(recording) > 0


class TestRecordingDataIntegrity:
    """Test data integrity during recording."""

    def test_recording_preserves_data_order(self):
        """Test that recording preserves chunk order."""
        sample_rate = 16000
        chunk_size = 100
        duration = 0.05  # 0.05s * 16000 Hz = 800 samples = 8 chunks of 100

        mic = MockMicrophone(sample_rate=sample_rate, chunk_size=chunk_size)
        mic.start()

        recording = mic.record(duration=duration)

        # Should have exactly 800 samples
        expected_samples = int(duration * sample_rate)
        assert len(recording) == expected_samples

        # Mock returns incrementing chunks (all 1s, all 2s, all 3s, ...)
        # Note: record() reads one chunk first for dtype detection,
        # so actual recording starts with chunk 2
        # Verify we got sequential chunks
        # First chunk in recording should be all 2s
        assert np.all(recording[:100] == 2)
        # Second chunk should be all 3s
        assert np.all(recording[100:200] == 3)
        # Third chunk should be all 4s
        assert np.all(recording[200:300] == 4)

    def test_wav_roundtrip_preserves_int16_data(self):
        """Test that WAV conversion roundtrip preserves int16 data."""
        mic = MockMicrophone()

        # Create test pattern
        original = np.array([0, 1000, -1000, 32000, -32000], dtype=np.int16)
        wav_data = mic._audio_to_wav(original)

        # Load back
        with wave.open(io.BytesIO(wav_data.tobytes()), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            loaded = np.frombuffer(frames, dtype=np.int16)

        np.testing.assert_array_equal(original, loaded)


class SparseMicrophone(BaseMicrophone):
    """Mock microphone that returns None chunks intermittently to simulate sparse audio."""

    def __init__(self, none_ratio=0.5, *args, **kwargs):
        """
        Args:
            none_ratio: Ratio of None chunks to return (0.0 to 1.0).
                0.5 means roughly 50% of capture() calls return None.
            args: Passed to BaseMicrophone.
            kwargs: Passed to BaseMicrophone.
        """
        super().__init__(*args, **kwargs)
        self._chunk_counter = 0
        self._none_ratio = none_ratio

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        self._chunk_counter += 1
        # Return None roughly none_ratio% of the time
        if self._none_ratio > 0:
            freq = max(2, int(1 / self._none_ratio))
            if (self._chunk_counter % freq) == 0:
                return None
        return np.full(self.chunk_size, 1, dtype=np.int16)


class NeverProducingMicrophone(BaseMicrophone):
    """Mock microphone that never produces audio."""

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        return None


class TestRecordingRobustness:
    """Test recording robustness with sparse or missing audio."""

    def test_record_with_sparse_audio_returns_correct_duration(self):
        """Test that record returns correct audio duration even with sparse chunks."""
        sample_rate = 16000
        chunk_size = 1024
        duration = 0.2

        # 50% of chunks are None
        mic = SparseMicrophone(none_ratio=0.5, sample_rate=sample_rate, chunk_size=chunk_size)
        mic.start()

        recording = mic.record(duration=duration)

        # Should still get the correct number of samples
        expected_samples = int(duration * sample_rate)
        assert len(recording) == expected_samples

    def test_record_with_sparse_audio_takes_longer_wall_clock_time(self):
        """Test that sparse audio takes longer wall-clock time but returns correct duration."""
        sample_rate = 16000
        duration = 0.1

        # 80% of chunks are None
        mic = SparseMicrophone(none_ratio=0.8, sample_rate=sample_rate, chunk_size=512)
        mic.start()

        recording = mic.record(duration=duration, timeout_factor=5.0)

        # Recording should still have correct number of samples
        expected_samples = int(duration * sample_rate)
        assert len(recording) == expected_samples

    def test_record_timeout_when_no_initial_audio(self):
        """Test that record times out when no initial audio is produced."""
        mic = NeverProducingMicrophone(sample_rate=16000)
        mic.start()

        with pytest.raises(TimeoutError) as exc_info:
            mic.record(duration=0.1, timeout_factor=0.5)

        assert "No audio data received" in str(exc_info.value)

    def test_record_timeout_when_audio_becomes_sparse(self):
        """Test that record times out when audio stream becomes too sparse."""

        class BecomesSparseMicrophone(BaseMicrophone):
            """Produces audio initially, then stops."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._chunk_counter = 0

            def _open_microphone(self):
                pass

            def _close_microphone(self):
                pass

            def _read_audio(self):
                self._chunk_counter += 1
                # Return audio for first 5 chunks, then None forever
                if self._chunk_counter <= 5:
                    return np.full(self.chunk_size, 1, dtype=np.int16)
                return None

        mic = BecomesSparseMicrophone(sample_rate=16000, chunk_size=1024)
        mic.start()

        # Request long duration with tight timeout
        with pytest.raises(TimeoutError) as exc_info:
            mic.record(duration=1.0, timeout_factor=0.2)

        error_msg = str(exc_info.value)
        assert "Recording timeout" in error_msg
        assert "collected" in error_msg
        assert "target:" in error_msg

    def test_record_timeout_includes_diagnostic_info(self):
        """Test that timeout error includes useful diagnostic information."""
        mic = NeverProducingMicrophone(sample_rate=16000)
        mic.start()

        duration = 0.5
        timeout_factor = 0.2

        with pytest.raises(TimeoutError) as exc_info:
            mic.record(duration=duration, timeout_factor=timeout_factor)

        error_msg = str(exc_info.value)
        # Should mention the timeout value
        assert str(duration * timeout_factor) in error_msg or f"{duration * timeout_factor:.2f}" in error_msg

    def test_record_with_valid_timeout_factor(self):
        """Test that record works with custom timeout_factor."""
        # Very sparse audio (90% None)
        mic = SparseMicrophone(none_ratio=0.9, sample_rate=16000, chunk_size=1024)
        mic.start()

        # With generous timeout, should succeed
        recording = mic.record(duration=0.1, timeout_factor=5.0)

        expected_samples = int(0.1 * 16000)
        assert len(recording) == expected_samples

    def test_record_no_busy_wait_on_none_chunks(self):
        """Test that record doesn't busy-wait when receiving None chunks."""

        class SlowSparseMicrophone(BaseMicrophone):
            """Tracks how many times _read_audio is called."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._read_count = 0

            def _open_microphone(self):
                pass

            def _close_microphone(self):
                pass

            def _read_audio(self):
                self._read_count += 1
                # Return None 80% of the time
                if self._read_count % 5 == 0:
                    return np.full(self.chunk_size, 1, dtype=np.int16)
                return None

        mic = SlowSparseMicrophone(sample_rate=16000, chunk_size=1024)
        mic.start()

        mic.record(duration=0.05, timeout_factor=3.0)

        # If there was busy-waiting, _read_audio would be called millions of times
        # With sleep(0.001), it should be reasonable
        # For 0.05s of audio at 16000Hz with chunk_size=1024, we need ~1 chunk
        # With 80% None rate, we need ~5 calls per chunk, so maybe 10-50 calls total
        assert mic._read_count < 10000, f"Too many read attempts: {mic._read_count} (possible busy-wait)"

    def test_record_exact_sample_count_with_sparse_audio(self):
        """Test that record returns exactly the requested number of samples."""
        sample_rate = 22050
        duration = 0.123
        expected_samples = int(duration * sample_rate)

        # Sparse audio
        mic = SparseMicrophone(none_ratio=0.6, sample_rate=sample_rate, chunk_size=512)
        mic.start()

        recording = mic.record(duration=duration, timeout_factor=3.0)

        # Should be exact
        assert len(recording) == expected_samples

    def test_record_default_timeout_factor(self):
        """Test that default timeout_factor of 2.0 is reasonable."""
        # Moderately sparse (40% None)
        mic = SparseMicrophone(none_ratio=0.4, sample_rate=16000, chunk_size=1024)
        mic.start()

        # Should work with default timeout_factor
        recording = mic.record(duration=0.1)  # Uses default timeout_factor=2.0

        expected_samples = int(0.1 * 16000)
        assert len(recording) == expected_samples
