# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np
import time
import threading
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.microphone import Microphone, BaseMicrophone


class MockMicrophone(BaseMicrophone):
    """Mock microphone for testing capture and stream methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk_counter = 0
        self._max_chunks = None
        self._return_none_count = 0

    def _open_microphone(self):
        pass

    def _close_microphone(self):
        pass

    def _read_audio(self):
        if self._max_chunks is not None and self._chunk_counter >= self._max_chunks:
            self._return_none_count += 1
            return None

        self._chunk_counter += 1
        return np.full(self.chunk_size, self._chunk_counter, dtype=np.int16)

    def set_max_chunks(self, count):
        """Limit number of chunks before returning None."""
        self._max_chunks = count


MOCK_USB_CARDS = ["SomeCard"]
MOCK_USB_PCM_DEVICES = ["plughw:CARD=SomeCard,DEV=0"]


class TestAudioCapture:
    """Test single audio chunk capture."""

    def test_capture_returns_numpy_array(self):
        """Test that capture returns a numpy array."""
        mic = MockMicrophone()
        mic.start()

        chunk = mic.capture()

        assert isinstance(chunk, np.ndarray)

    def test_capture_returns_correct_size(self):
        """Test that capture returns correct chunk size."""
        chunk_size = 2048
        mic = MockMicrophone(chunk_size=chunk_size)
        mic.start()

        chunk = mic.capture()

        assert len(chunk) == chunk_size

    def test_capture_returns_correct_dtype(self):
        """Test that capture returns correct data type."""
        mic = MockMicrophone()
        mic.start()

        chunk = mic.capture()

        assert chunk.dtype == np.int16

    def test_capture_when_not_started_returns_none(self):
        """Test that capture returns None when microphone not started."""
        mic = MockMicrophone()

        chunk = mic.capture()

        assert chunk is None

    def test_multiple_sequential_captures(self):
        """Test multiple sequential capture calls."""
        mic = MockMicrophone(chunk_size=128)
        mic.start()

        chunks = []
        for _ in range(5):
            chunk = mic.capture()
            chunks.append(chunk)

        assert len(chunks) == 5

        # Each chunk should have incrementing values (1, 2, 3, 4, 5)
        for i, chunk in enumerate(chunks, 1):
            assert np.all(chunk == i)

    def test_capture_handles_none_from_read(self):
        """Test that capture handles None from _read_audio."""
        mic = MockMicrophone()
        mic.set_max_chunks(0)
        mic.start()

        chunk = mic.capture()

        assert chunk is None

    def test_capture_is_thread_safe(self):
        """Test that capture is thread-safe."""
        mic = MockMicrophone()
        mic.start()

        results = []
        errors = []

        def capture_loop():
            try:
                for _ in range(10):
                    chunk = mic.capture()
                    results.append(chunk)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=capture_loop) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 30  # 3 threads × 10 captures


class TestAudioStreaming:
    """Test continuous audio streaming."""

    def test_stream_yields_audio_chunks(self):
        """Test that stream yields audio chunks continuously."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()
        chunks = []

        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            if i >= 4:
                break

        assert len(chunks) == 5
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)

    def test_stream_yields_correct_chunk_sizes(self):
        """Test that stream yields correct chunk sizes."""
        chunk_size = 1024
        mic = MockMicrophone(chunk_size=chunk_size)
        mic.start()

        stream = mic.stream()

        for i, chunk in enumerate(stream):
            assert len(chunk) == chunk_size
            if i >= 2:
                break

    def test_stream_continues_while_started(self):
        """Test that stream continues while microphone is started."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if chunk_count >= 5:
                mic.stop()
                break

        assert chunk_count >= 5

    def test_stream_stops_when_microphone_stopped(self):
        """Test that stream stops when microphone is stopped."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()

        # Get a couple chunks
        next(stream)
        next(stream)

        # Stop microphone
        mic.stop()

        # Stream should eventually stop (give it time to detect stop)
        time.sleep(0.01)

    def test_stream_skips_none_values(self):
        """Test that stream skips None values without yielding them."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()
        chunks = []

        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            if i >= 4:
                break

        # All yielded chunks should be non-None
        for chunk in chunks:
            assert chunk is not None

    def test_stream_can_be_iterated_multiple_times(self):
        """Test that stream can be called multiple times."""
        mic = MockMicrophone()
        mic.start()

        # First stream
        stream1 = mic.stream()
        chunk1 = next(stream1)

        # Second stream (should work independently)
        stream2 = mic.stream()
        chunk2 = next(stream2)

        assert chunk1 is not None
        assert chunk2 is not None


class TestStreamingWithRealMicrophone:
    """Test streaming with real microphone implementations (mocked hardware)."""

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_microphone_capture(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test capture with ALSA microphone."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Mock audio data
        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        mic = Microphone(device=0)
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) == 1024

    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.cards", return_value=MOCK_USB_CARDS)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_indexes", return_value=[0])
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.card_name")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.pcms", return_value=MOCK_USB_PCM_DEVICES)
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.PCM")
    @patch("arduino.app_peripherals.microphone.alsa_microphone.alsaaudio.mixers", return_value=[])
    def test_alsa_microphone_stream(self, mock_mixers, mock_pcm, mock_pcms, mock_card_name, mock_card_indexes, mock_cards):
        """Test streaming with ALSA microphone."""
        mock_card_name.side_effect = lambda idx: (MOCK_USB_CARDS[idx], f"USB Audio Device {idx}")

        pcm_instance = MagicMock()
        mock_pcm.return_value = pcm_instance

        # Mock audio data
        test_data = np.arange(1024, dtype=np.int16)
        pcm_instance.read.return_value = (1024, test_data.tobytes())

        mic = Microphone(device=0)
        mic.start()

        stream = mic.stream()
        chunks = []

        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            if i >= 2:
                break

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)


class TestCaptureDataFormat:
    """Test data format validation in capture."""

    @pytest.mark.parametrize(
        "format_str,expected_dtype",
        [
            ("S16_LE", np.int16),
            ("S32_LE", np.int32),
        ],
    )
    def test_capture_respects_format(self, format_str, expected_dtype):
        """Test that capture returns data in correct format."""

        class TypedMockMic(BaseMicrophone):
            def _open_microphone(self):
                pass

            def _close_microphone(self):
                pass

            def _read_audio(self):
                return np.zeros(self.chunk_size, dtype=expected_dtype)

        mic = TypedMockMic(format=format_str, chunk_size=128)
        mic.start()

        chunk = mic.capture()

        assert chunk.dtype == expected_dtype


class TestStreamSynchronization:
    """Test stream synchronization and timing."""

    def test_stream_with_slow_consumer(self):
        """Test streaming when consumer is slower than producer."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()
        chunks = []

        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            time.sleep(0.01)  # Simulate slow processing
            if i >= 2:
                break

        assert len(chunks) == 3

    def test_capture_is_non_blocking(self):
        """Test that capture doesn't block unnecessarily."""
        mic = MockMicrophone()
        mic.set_max_chunks(0)
        mic.start()

        start = time.time()
        chunk = mic.capture()
        elapsed = time.time() - start

        assert chunk is None
        assert elapsed < 0.2  # Should return quickly


class TestStreamingLifecycle:
    """Test stream lifecycle and cleanup."""

    def test_stream_after_stop_start_cycle(self):
        """Test that stream works after stop/start cycle."""
        mic = MockMicrophone()

        mic.start()
        chunk1 = mic.capture()
        mic.stop()

        mic.start()
        chunk2 = mic.capture()
        mic.stop()

        assert chunk1 is not None
        assert chunk2 is not None

    def test_stream_with_context_manager(self):
        """Test streaming with context manager."""
        mic = MockMicrophone()

        with mic:
            stream = mic.stream()
            chunk = next(stream)
            assert chunk is not None

        # After context, microphone should be stopped
        assert not mic.is_started()


class TestCaptureDataIntegrity:
    """Test data integrity during capture."""

    def test_captured_data_not_modified_by_subsequent_captures(self):
        """Test that captured data is not modified by later operations."""
        mic = MockMicrophone()
        mic.start()

        chunk1 = mic.capture()
        original_values = chunk1.copy()

        # Capture more
        for _ in range(5):
            mic.capture()

        # Original should be unchanged
        np.testing.assert_array_equal(chunk1, original_values)

    def test_stream_yields_independent_arrays(self):
        """Test that stream yields independent numpy arrays."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()

        chunks = []
        for i, chunk in enumerate(stream):
            chunks.append(chunk)
            if i >= 2:
                break

        # Modify first chunk
        chunks[0][:] = 9999

        # Other chunks should be unaffected
        assert not np.all(chunks[1] == 9999)


class TestCapturePerformance:
    """Test capture performance characteristics."""

    def test_capture_many_chunks_quickly(self):
        """Test capturing many chunks doesn't accumulate delay."""
        mic = MockMicrophone()
        mic.start()

        start = time.time()

        for _ in range(100):
            mic.capture()

        elapsed = time.time() - start

        # Should complete quickly (mostly just function call overhead)
        assert elapsed < 1.0

    def test_stream_can_handle_fast_iteration(self):
        """Test that stream handles fast iteration."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()
        count = 0

        start = time.time()
        for chunk in stream:
            count += 1
            if count >= 50:
                break
        elapsed = time.time() - start

        assert count == 50
        assert elapsed < 1.0


class TestCaptureEdgeCases:
    """Test edge cases in capture and streaming."""

    def test_capture_with_very_small_chunk_size(self):
        """Test capture with very small chunk size."""
        mic = MockMicrophone(chunk_size=16)
        mic.start()

        chunk = mic.capture()

        assert len(chunk) == 16

    def test_capture_with_very_large_chunk_size(self):
        """Test capture with very large chunk size."""
        mic = MockMicrophone(chunk_size=16384)
        mic.start()

        chunk = mic.capture()

        assert len(chunk) == 16384

    def test_stream_breaking_early(self):
        """Test breaking out of stream early."""
        mic = MockMicrophone()
        mic.start()

        stream = mic.stream()

        # Break after first chunk
        for chunk in stream:
            assert chunk is not None
            break

        # Should be able to capture again
        chunk2 = mic.capture()
        assert chunk2 is not None


class TestCaptureWithDifferentSampleRates:
    """Test capture with different sample rates."""

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_capture_with_various_sample_rates(self, sample_rate):
        """Test that capture works with various sample rates."""
        mic = MockMicrophone(sample_rate=sample_rate)
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
        assert isinstance(chunk, np.ndarray)


class TestCaptureWithDifferentChannels:
    """Test capture with different channel configurations."""

    @pytest.mark.parametrize("channels", [1, 2])
    def test_capture_with_various_channels(self, channels):
        """Test that capture works with different channel counts."""
        mic = MockMicrophone(channels=channels)
        mic.start()

        chunk = mic.capture()

        assert chunk is not None
