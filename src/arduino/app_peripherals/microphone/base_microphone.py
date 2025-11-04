# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import RATE_16K, MONO, FORMAT_S16_LE, BALANCED_CHUNK
from .errors import MicrophoneOpenError
from arduino.app_utils import Logger

logger = Logger("Microphone")


class BaseMicrophone(ABC):
    """
    Abstract base class for microphone implementations.

    This class defines the common interface that all microphone implementations must follow,
    providing a unified API regardless of the underlying audio capture protocol or type.
    """

    def __init__(
        self,
        sample_rate: int = RATE_16K,
        channels: int = MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = BALANCED_CHUNK,
    ):
        """
        Initialize the microphone base.

        Args:
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format (default: "S16_LE").
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
            Numpy array or None if no audio is available.
        """
        with self._mic_lock:
            if not self._is_started:
                return None

            return self._read_audio()

    def is_started(self) -> bool:
        """Check if the microphone is started."""
        return self._is_started

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

    def record(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration.

        Args:
            duration (float): Recording duration in seconds.

        Returns:
            np.ndarray: Complete recording as a single numpy array.

        Raises:
            MicrophoneOpenError: If microphone is not started.
            ValueError: If duration is not positive.
        """
        if not self._is_started:
            raise MicrophoneOpenError("Microphone must be started before recording")

        if duration <= 0:
            raise ValueError("Duration must be positive")

        # Pre-allocate array for the entire recording
        # Estimate total samples needed
        total_samples = int(duration * self.sample_rate * self.channels)

        # Get dtype from first chunk to know how to allocate
        first_chunk = None
        while first_chunk is None:
            first_chunk = self.capture()

        # Allocate the full recording buffer
        recording = np.zeros(total_samples, dtype=first_chunk.dtype)

        # Start recording fresh from this point
        offset = 0
        start_time = time.time()
        elapsed = 0

        while elapsed < duration:
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
            elapsed = time.time() - start_time

        # Trim to actual recorded length if we recorded less than expected
        if offset < total_samples:
            recording = recording[:offset]

        return recording

    def record_file(self, duration: float, output_file: Union[str, Path]) -> None:
        """
        Record audio for a specified duration and save to a WAV file.

        Args:
            duration (float): Recording duration in seconds.
            output_file (Union[str, Path]): Path to save the recording as a WAV file.

        Raises:
            MicrophoneOpenError: If microphone is not started.
            ValueError: If duration is not positive.
        """
        recording = self.record(duration)
        self._save_wav(recording, output_file)
        self.logger.info(f"Recording saved to {output_file}")

    def _save_wav(self, audio: np.ndarray, filepath: Union[str, Path]) -> None:
        """
        Save audio data to a WAV file.

        Args:
            audio (np.ndarray): Audio data to save.
            filepath (Union[str, Path]): Output file path.
        """
        import wave

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get base dtype kind and size
        dtype_kind = audio.dtype.kind
        dtype_size = audio.dtype.itemsize
        is_big_endian = audio.dtype.byteorder == ">"

        # Determine sample width and prepare data for writing
        # We'll write directly with minimal copies
        if dtype_kind == "i":  # Signed integer
            if dtype_size == 1:  # int8
                # WAV uses unsigned 8-bit - must convert
                write_data = (audio.astype(np.int16) + 128).astype(np.uint8)
                sampwidth = 1
            elif dtype_size == 2:  # int16
                write_data = audio.byteswap() if is_big_endian else audio
                sampwidth = 2
            elif dtype_size == 4:  # int32
                write_data = audio.byteswap() if is_big_endian else audio
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
            # Assume normalized [-1.0, 1.0], convert to int16
            write_data = np.clip(audio, -1.0, 1.0)
            write_data = (write_data * 32767).astype(np.int16)
            sampwidth = 2

        else:
            raise ValueError(f"Unsupported audio data type: {audio.dtype}. Supported: int8/16/32, uint8/16/32, float32/64.")

        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(write_data.tobytes())

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
