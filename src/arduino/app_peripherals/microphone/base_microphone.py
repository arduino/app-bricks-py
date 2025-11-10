# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneOpenError
from arduino.app_utils import Logger

logger = Logger("Microphone")


class BaseMicrophone(ABC):
    """
    Abstract base class for microphone implementations.

    This class defines the common interface that all microphone implementations must follow,
    providing a unified API regardless of the underlying audio capture protocol or type.

    The output is always a numpy array with the ALSA PCM format.
    """

    def __init__(
        self,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = CHUNK_BALANCED,
    ):
        """
        Initialize the microphone base.

        Args:
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format in ALSA PCM notation (default: "S16_LE").
            chunk_size (int): Number of frames per chunk (default: 1024).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.chunk_size = chunk_size
        self.logger = logger  # This will be overridden by subclasses if needed

        self._mic_lock = threading.Lock()
        self._is_started = False

    def start(self) -> None:
        """Start the microphone capture."""
        with self._mic_lock:
            if self._is_started:
                return

            try:
                self._open_microphone()
                self._is_started = True
                self.logger.info(f"Successfully started {self.__class__.__name__}")
            except Exception as e:
                raise MicrophoneOpenError(f"Failed to start microphone: {e}")

    def stop(self) -> None:
        """Stop the microphone and release resources."""
        with self._mic_lock:
            if not self._is_started:
                return

            try:
                self._close_microphone()
                self._is_started = False
                self.logger.info(f"Stopped {self.__class__.__name__}")
            except Exception as e:
                self.logger.warning(f"Error stopping microphone: {e}")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture an audio chunk from the microphone.

        Returns:
            Numpy array in ALSA PCM format or None if no audio is available.
        """
        with self._mic_lock:
            if not self._is_started:
                return None

            return self._read_audio()

    def stream(self):
        """
        Continuously capture audio chunks from the microphone.

        This is a generator that yields audio chunks continuously while the microphone is started.

        Yields:
            np.ndarray: Audio chunks as numpy arrays.
        """
        while self._is_started:
            chunk = self.capture()
            if chunk is not None:
                yield chunk
            else:
                # Avoid busy-waiting if no audio available
                time.sleep(0.001)

    def record(self, duration: float, timeout_factor: float = 2.0) -> np.ndarray:
        """
        Record audio for a specified duration and return as raw PCM format.

        Args:
            duration (float): Recording duration in seconds.
            timeout_factor (float): Maximum wall-clock time as multiple of duration (default: 2.0).
                                   This prevents indefinite blocking when audio source is sparse.

        Returns:
            np.ndarray: Raw audio data in raw ALSA PCM format.

        Raises:
            MicrophoneOpenError: If microphone is not started.
            ValueError: If duration is not positive.
            TimeoutError: If recording takes longer than duration * timeout_factor.
        """
        return self._record_pcm(duration, timeout_factor)

    def record_wav(self, duration: float, timeout_factor: float = 2.0) -> np.ndarray:
        """
        Record audio for a specified duration and return as WAV format.

        Args:
            duration (float): Recording duration in seconds.
            timeout_factor (float): Maximum wall-clock time as multiple of duration (default: 2.0).
                                   This prevents indefinite blocking when audio source is sparse.

        Returns:
            np.ndarray: Raw audio data in WAV format as numpy array.

        Raises:
            MicrophoneOpenError: If microphone is not started.
            ValueError: If duration is not positive.
            TimeoutError: If recording takes longer than duration * timeout_factor.
        """
        audio_data = self._record_pcm(duration, timeout_factor)
        return self._audio_to_wav(audio_data)

    def _record_pcm(self, duration: float, timeout_factor: float = 2.0) -> np.ndarray:
        """
        Record raw audio data for a specified duration.

        Args:
            duration (float): Recording duration in seconds.
            timeout_factor (float): Maximum wall-clock time as multiple of duration (default: 2.0).
                                   This prevents indefinite blocking when audio source is sparse.

        Returns:
            np.ndarray: Raw audio data in raw ALSA PCM format.

        Raises:
            MicrophoneOpenError: If microphone is not started.
            ValueError: If duration is not positive.
            TimeoutError: If recording takes longer than duration * timeout_factor.
        """
        if not self._is_started:
            raise MicrophoneOpenError("Microphone must be started before recording")

        if duration <= 0:
            raise ValueError("Duration must be positive")

        # Calculate timeout to prevent indefinite blocking
        timeout = duration * timeout_factor
        total_samples = int(duration * self.sample_rate * self.channels)

        # Get dtype from first chunk with timeout protection
        first_chunk = None
        start_wait = time.time()
        while first_chunk is None:
            if time.time() - start_wait > timeout:
                raise TimeoutError(f"No audio data received after {timeout:.2f}s")
            first_chunk = self.capture()
            if first_chunk is None:
                time.sleep(0.01)

        # Allocate the full recording buffer
        recording = np.zeros(total_samples, dtype=first_chunk.dtype)

        offset = 0
        start_time = time.time()
        while offset < total_samples:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                audio_duration = offset / (self.sample_rate * self.channels)
                raise TimeoutError(f"Recording timeout: collected {audio_duration:.2f}s of audio in {elapsed:.2f}s (target: {duration}s)")

            chunk = self.capture()
            if chunk is not None:
                chunk_len = len(chunk)
                # Ensure we don't overflow the buffer
                if offset + chunk_len > total_samples:
                    chunk_len = total_samples - offset
                    recording[offset : offset + chunk_len] = chunk[:chunk_len]
                    break
                recording[offset : offset + chunk_len] = chunk
                offset += chunk_len
            else:
                time.sleep(0.001)

        return recording

    def _audio_to_wav(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert raw PCM audio data to WAV format.

        Args:
            audio (np.ndarray): Raw PCM audio data to convert.

        Returns:
            np.ndarray: WAV data as uint8 numpy array (including header).
        """
        import wave
        import io

        # Get base dtype kind and size
        dtype_kind = audio.dtype.kind
        dtype_size = audio.dtype.itemsize

        # Convert to native byte order since the wave module handle byte ordering for the WAV format
        if audio.dtype.byteorder not in ("=", "|"):
            audio = audio.astype(audio.dtype.newbyteorder("="))

        if dtype_kind == "i":  # Signed integer
            if dtype_size == 1:  # int8
                # WAV uses unsigned 8-bit - must convert
                write_data = (audio.astype(np.int16) + 128).astype(np.uint8)
                sampwidth = 1
            elif dtype_size == 2:  # int16
                write_data = audio
                sampwidth = 2
            elif dtype_size == 4:  # int32
                # Check if this is 24-bit audio packed in 32-bit containers
                is_24bit = self.format in ("S24_LE", "S24_BE")
                if is_24bit:
                    # Extract 24-bit samples from 32-bit containers (padding is in LSB per ALSA)
                    import sys

                    bytes_view = audio.view("u1").reshape(-1, 4)  # Reshape to rows of 4 bytes
                    if sys.byteorder == "little":
                        # On LE system: LSB padding is at byte 0, take bytes 1-3
                        write_data = bytes_view[:, 1:4].flatten()
                    else:
                        # On BE system: LSB padding is at byte 3, take bytes 0-2
                        write_data = bytes_view[:, :3].flatten()
                    sampwidth = 3
                else:
                    # True 32-bit audio
                    write_data = audio
                    sampwidth = 4
            else:
                raise ValueError(f"Unsupported signed integer size: {dtype_size} bytes. Supported: 1, 2, 4.")

        elif dtype_kind == "u":  # Unsigned integer
            if dtype_size == 1:  # uint8
                # Already in correct format for WAV
                write_data = audio
                sampwidth = 1
            elif dtype_size == 2:  # uint16
                # Convert to signed int16
                write_data = (audio.astype(np.int32) - 32768).astype(np.int16)
                sampwidth = 2
            elif dtype_size == 4:  # uint32
                # Convert to signed int32
                write_data = (audio.astype(np.int64) - 2147483648).astype(np.int32)
                sampwidth = 4
            else:
                raise ValueError(f"Unsupported unsigned integer size: {dtype_size} bytes. Supported: 1, 2, 4.")

        elif dtype_kind == "f":  # Float
            # ALSA float formats are normalized [-1.0, 1.0] => scale and convert to int16
            write_data = np.clip(audio, -1.0, 1.0)
            write_data = (write_data * 32767).astype(np.int16)
            sampwidth = 2

        else:
            raise ValueError(f"Unsupported audio data type: {audio.dtype}. Supported: int8/16/32, uint8/16/32, float32/64.")

        # Write to in-memory buffer
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(write_data.tobytes())

        # Convert to numpy uint8 array
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    def is_started(self) -> bool:
        """Check if the microphone is started."""
        return self._is_started

    @abstractmethod
    def _open_microphone(self) -> None:
        """Open the microphone connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _close_microphone(self) -> None:
        """Close the microphone connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _read_audio(self) -> Optional[np.ndarray]:
        """Read a single audio chunk from the microphone. Must be implemented by subclasses."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
