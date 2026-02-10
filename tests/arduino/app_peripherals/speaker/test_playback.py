# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import io

import numpy as np
import wave

from arduino.app_peripherals.speaker import Speaker, BaseSpeaker, FormatPlain, FormatPacked
from arduino.app_peripherals.speaker.errors import SpeakerWriteError


class MockSpeaker(BaseSpeaker):
    """Mock speaker for testing playback functionality."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = 1024,
        auto_reconnect: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, channels=channels, format=format, buffer_size=buffer_size, auto_reconnect=auto_reconnect)
        self._chunks_written = []

    def _open_speaker(self):
        pass

    def _close_speaker(self):
        pass

    def _write_audio(self, audio_chunk: np.ndarray):
        # Store written chunks for verification
        self._chunks_written.append(audio_chunk.copy())


class TestPlayPCM:
    """Test playing PCM audio data."""

    def test_play_pcm_with_single_chunk(self):
        """Test play_pcm with audio that fits in a single buffer."""
        spkr = MockSpeaker(sample_rate=16000, buffer_size=1024)
        spkr.start()

        audio_data = np.arange(512, dtype=np.int16)
        spkr.play_pcm(audio_data)

        assert len(spkr._chunks_written) == 1
        assert len(spkr._chunks_written[0]) == 512
        np.testing.assert_array_equal(spkr._chunks_written[0], audio_data)

    def test_play_pcm_with_multiple_chunks(self):
        """Test play_pcm with audio that requires multiple buffers."""
        buffer_size = 1024
        spkr = MockSpeaker(sample_rate=16000, buffer_size=buffer_size)
        spkr.start()

        audio_data = np.arange(2500, dtype=np.int16)
        spkr.play_pcm(audio_data)

        # Should be split into 3 chunks: 1024, 1024, 452
        assert len(spkr._chunks_written) == 3
        assert len(spkr._chunks_written[0]) == buffer_size
        assert len(spkr._chunks_written[1]) == buffer_size
        assert len(spkr._chunks_written[2]) == 452

        # Verify data integrity
        reconstructed = np.concatenate(spkr._chunks_written)
        np.testing.assert_array_equal(reconstructed, audio_data)

    def test_play_pcm_validates_empty_data(self):
        """Test that play_pcm validates empty data."""
        spkr = MockSpeaker()
        spkr.start()

        with pytest.raises(ValueError) as exc_info:
            spkr.play_pcm(np.array([], dtype=np.int16))

        assert "empty" in str(exc_info.value).lower()

    def test_play_pcm_validates_none_data(self):
        """Test that play_pcm validates None data."""
        spkr = MockSpeaker()
        spkr.start()

        with pytest.raises(ValueError):
            spkr.play_pcm(None)

    def test_play_pcm_requires_started_speaker(self):
        """Test that play_pcm requires speaker to be started."""
        spkr = MockSpeaker()

        with pytest.raises(SpeakerWriteError) as exc_info:
            audio_data = np.zeros(1024, dtype=np.int16)
            spkr.play_pcm(audio_data)

        assert "start" in str(exc_info.value).lower()

    def test_play_pcm_validates_dtype(self):
        """Test that play_pcm validates data type matches speaker format."""
        spkr = MockSpeaker(format=np.int16)
        spkr.start()

        # Try to play float32 data when int16 is expected
        audio_data = np.zeros(1024, dtype=np.float32)
        with pytest.raises(ValueError) as exc_info:
            spkr.play_pcm(audio_data)

        assert "dtype" in str(exc_info.value).lower()

    def test_play_pcm_with_stereo(self):
        """Test play_pcm with stereo audio."""
        spkr = MockSpeaker(sample_rate=16000, channels=2, buffer_size=1024)
        spkr.start()

        # Stereo data: interleaved [L, R, L, R, ...]
        audio_data = np.arange(2048, dtype=np.int16)  # 1024 stereo frames
        spkr.play_pcm(audio_data)

        assert len(spkr._chunks_written) == 1
        np.testing.assert_array_equal(spkr._chunks_written[0], audio_data)

    @pytest.mark.parametrize(
        "format",
        [np.uint8, np.int16, np.int32, np.float32, np.float64],
    )
    def test_play_pcm_with_different_formats(self, format):
        """Test play_pcm with different audio formats."""
        spkr = MockSpeaker(format=format, buffer_size=512)
        spkr.start()

        audio_data = np.zeros(512, dtype=format)
        spkr.play_pcm(audio_data)

        assert len(spkr._chunks_written) == 1
        assert spkr._chunks_written[0].dtype == format


class TestPlayWAV:
    """Test playing WAV audio data."""

    def test_play_wav_with_valid_data(self):
        """Test play_wav with valid WAV data."""
        spkr = MockSpeaker(sample_rate=16000, channels=1)
        spkr.start()

        # Create WAV data
        audio_samples = np.arange(1000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        spkr.play_wav(wav_data)

        # Verify data was written
        assert len(spkr._chunks_written) > 0
        reconstructed = np.concatenate(spkr._chunks_written)
        np.testing.assert_array_equal(reconstructed, audio_samples)

    def test_play_wav_validates_empty_data(self):
        """Test that play_wav validates empty data."""
        spkr = MockSpeaker()
        spkr.start()

        with pytest.raises(ValueError):
            spkr.play_wav(np.array([], dtype=np.uint8))

    def test_play_wav_validates_invalid_format(self):
        """Test that play_wav validates WAV format."""
        spkr = MockSpeaker()
        spkr.start()

        # Invalid WAV data (just random bytes)
        invalid_data = np.random.randint(0, 255, size=100, dtype=np.uint8)
        with pytest.raises(ValueError):
            spkr.play_wav(invalid_data)

    def test_play_wav_validates_sample_rate_mismatch(self):
        """Test that play_wav validates sample rate matches."""
        spkr = MockSpeaker(sample_rate=16000)
        spkr.start()

        # Create WAV with different sample rate
        audio_samples = np.arange(1000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)  # Different from speaker's 16000
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            spkr.play_wav(wav_data)

        assert "sample rate" in str(exc_info.value).lower()

    def test_play_wav_validates_channels_mismatch(self):
        """Test that play_wav validates channels match."""
        spkr = MockSpeaker(sample_rate=16000, channels=1)
        spkr.start()

        # Create stereo WAV
        audio_samples = np.arange(2000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            spkr.play_wav(wav_data)

        assert "channel" in str(exc_info.value).lower()

    def test_play_wav_with_stereo(self):
        """Test play_wav with stereo WAV data."""
        spkr = MockSpeaker(sample_rate=16000, channels=2)
        spkr.start()

        # Create stereo WAV
        audio_samples = np.arange(2000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        spkr.play_wav(wav_data)

        # Verify data was written
        assert len(spkr._chunks_written) > 0


class TestPlayMethod:
    """Test the basic play method."""

    def test_play_writes_single_chunk(self):
        """Test that play writes a single chunk."""
        spkr = MockSpeaker(buffer_size=1024)
        spkr.start()

        audio_data = np.zeros(1024, dtype=np.int16)
        spkr.play(audio_data)

        assert len(spkr._chunks_written) == 1
        np.testing.assert_array_equal(spkr._chunks_written[0], audio_data)

    def test_play_validates_empty_data(self):
        """Test that play validates empty data."""
        spkr = MockSpeaker()
        spkr.start()

        with pytest.raises(ValueError):
            spkr.play(np.array([], dtype=np.int16))

    def test_play_validates_dtype(self):
        """Test that play validates dtype."""
        spkr = MockSpeaker(format=np.int16)
        spkr.start()

        with pytest.raises(ValueError):
            spkr.play(np.zeros(1024, dtype=np.float32))

    def test_play_requires_started(self):
        """Test that play requires speaker to be started."""
        spkr = MockSpeaker()

        with pytest.raises(SpeakerWriteError):
            spkr.play(np.zeros(1024, dtype=np.int16))


class TestVolumeControl:
    """Test volume control functionality."""

    def test_volume_scaling_int16(self):
        """Test volume scaling with int16 format."""
        spkr = MockSpeaker(format=np.int16)
        spkr.start()
        spkr.volume = 50

        audio_data = np.full(1024, 1000, dtype=np.int16)
        spkr.play(audio_data)

        # Volume should scale the audio (approximately halved at 50%)
        written = spkr._chunks_written[0]
        assert np.all(np.abs(written) < np.abs(audio_data))

    def test_volume_zero_silences_audio(self):
        """Test that volume=0 produces silence."""
        spkr = MockSpeaker(format=np.int16)
        spkr.start()
        spkr.volume = 0

        audio_data = np.full(1024, 1000, dtype=np.int16)
        spkr.play(audio_data)

        # Volume=0 should produce all zeros
        written = spkr._chunks_written[0]
        np.testing.assert_array_equal(written, np.zeros_like(audio_data))

    def test_volume_max_no_change(self):
        """Test that volume=100 doesn't change audio."""
        spkr = MockSpeaker(format=np.int16)
        spkr.start()
        spkr.volume = 100

        audio_data = np.full(1024, 1000, dtype=np.int16)
        spkr.play(audio_data)

        # Volume=100 should not change audio
        written = spkr._chunks_written[0]
        np.testing.assert_array_equal(written, audio_data)

    def test_volume_scaling_float32(self):
        """Test volume scaling with float32 format."""
        spkr = MockSpeaker(format=np.float32)
        spkr.start()
        spkr.volume = 50

        audio_data = np.full(1024, 0.5, dtype=np.float32)
        spkr.play(audio_data)

        # Volume should scale the audio
        written = spkr._chunks_written[0]
        np.testing.assert_array_almost_equal(written, audio_data * 0.5, decimal=5)


class TestStaticMethods:
    """Test static convenience methods."""

    def test_play_pcm_static_method(self, mock_alsa_usb_speakers):
        """Test Speaker.play_pcm static method."""
        audio_data = np.zeros(1024, dtype=np.int16)

        # Should not raise
        Speaker.play_pcm(pcm_audio=audio_data, sample_rate=16000, channels=1, format=np.int16, device=0)

    def test_play_wav_static_method(self, mock_alsa_usb_speakers):
        """Test Speaker.play_wav static method."""
        # Create WAV data
        audio_samples = np.arange(1000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        # Should not raise
        Speaker.play_wav(wav_audio=wav_data, device=0)


class TestWAVConversion:
    """Test WAV to PCM conversion."""

    def test_wav_to_pcm_int16(self):
        """Test converting 16-bit WAV to PCM."""
        spkr = MockSpeaker(sample_rate=16000, channels=1, format=np.int16)

        # Create WAV data
        audio_samples = np.arange(1000, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        pcm_data = spkr._wav_to_pcm(wav_data)

        np.testing.assert_array_equal(pcm_data, audio_samples)

    def test_wav_to_pcm_uint8(self):
        """Test converting 8-bit WAV to PCM."""
        spkr = MockSpeaker(sample_rate=16000, channels=1, format=np.uint8)

        # Create 8-bit WAV data
        audio_samples = np.arange(100, dtype=np.uint8)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(16000)
            wav_file.writeframesraw(audio_samples.tobytes())

        wav_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        pcm_data = spkr._wav_to_pcm(wav_data)

        # 8-bit WAV is unsigned, but conversion happens
        assert len(pcm_data) == len(audio_samples)
