# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import threading
from abc import ABC, abstractmethod
from typing import Literal
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneOpenError, MicrophoneReadError
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
        auto_reconnect: bool = True,
    ):
        """
        Initialize the microphone base.

        Args:
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format in ALSA PCM notation (default: "S16_LE").
            chunk_size (int): Number of frames per chunk (default: 512).
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        self.sample_rate = sample_rate
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        self.channels = channels
        if format == "":
            raise ValueError("Format must be a non-empty string")
        self.format = format
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self.chunk_size = chunk_size
        self.logger = logger  # This will be overridden by subclasses if needed
        self.name = self.__class__.__name__  # This will be overridden by subclasses if needed

        self._mic_lock = threading.Lock()
        self._is_started = False

        # Auto-reconnection parameters
        self.auto_reconnect = auto_reconnect
        self.auto_reconnect_delay = 1.0
        self.first_connection_max_retries = 10

        # Stream interruption detection
        self._consecutive_none_chunks = 0

        # Status handling
        self._status: Literal["disconnected", "connected", "streaming", "paused"] = "disconnected"
        self._on_status_changed_cb: Callable[[str, dict], None] | None = None
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MicrophoneCallbacksRunner")

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Read-only property for camera status."""
        return self._status

    @property
    def _none_chunk_threshold(self) -> int:
        """Heuristic: 750ms of empty chunk based on current sample rate. Always at least 10."""
        return max(10, int(0.75 * self.sample_rate) // self.chunk_size)

    def start(self) -> None:
        """Start the microphone capture."""
        with self._mic_lock:
            self.logger.info("Starting microphone...")

            attempt = 0
            while not self.is_started():
                try:
                    self._open_microphone()
                    self._is_started = True
                    self.logger.info(f"Successfully started {self.name}")
                except MicrophoneOpenError as e:  # We consider this a fatal error so we don't retry
                    self.logger.error(f"Fatal error while starting {self.name}: {e}")
                    raise
                except Exception as e:
                    if not self.auto_reconnect:
                        raise
                    attempt += 1
                    if attempt >= self.first_connection_max_retries:
                        raise MicrophoneOpenError(
                            f"Failed to start microphone {self.name} after {self.first_connection_max_retries} attempts, last error is: {e}"
                        )

                    delay = min(self.auto_reconnect_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff
                    self.logger.warning(
                        f"Failed attempt {attempt}/{self.first_connection_max_retries} at starting microphone {self.name}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

    def stop(self) -> None:
        """Stop the microphone and release resources."""
        with self._mic_lock:
            if not self.is_started():
                return

            self.logger.info("Stopping microphone...")

            try:
                self._close_microphone()
                self._event_executor.shutdown()
                self._is_started = False
                self.logger.info(f"Successfully stopped {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to stop microphone: {e}")

    def capture(self) -> np.ndarray | None:
        """
        Capture an audio chunk from the microphone.

        Returns:
            Numpy array in ALSA PCM format or None if no audio is available.

        Raises:
            MicrophoneReadError: If the microphone is not started.
            Exception: If the underlying implementation fails to read a frame.
        """
        with self._mic_lock:
            if not self.is_started():
                raise MicrophoneReadError(f"Attempted to read from {self.name} before starting it.")

            chunk = self._read_audio()
            if chunk is None:
                self._consecutive_none_chunks += 1
                if self._consecutive_none_chunks >= self._none_chunk_threshold:
                    self._set_status("paused")
                return None

            self._set_status("streaming")

            self._consecutive_none_chunks = 0

            return chunk

    def stream(self):
        """
        Continuously capture audio chunks from the microphone.

        This is a generator that yields audio chunks continuously while the microphone is started.

        Yields:
            np.ndarray: Audio chunks as numpy arrays.
        """
        while self.is_started():
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
        pcm_data = self._record_pcm(duration, timeout_factor)
        return self._audio_to_wav(pcm_data)

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

    def on_status_changed(self, callback: Callable[[str, dict], None] | None):
        """Registers or removes a callback to be triggered on microphone lifecycle events.

        When a microphone status changes, the provided callback function will be invoked.
        If None is provided, the callback will be removed.

        Args:
            callback (Callable[[str, dict], None]): A callback that will be called every time the
                microphone status changes with the new status and any associated data. The status
                names depend on the actual microphone implementation being used. Some common events
                are:
                - 'connected': The microphone has been reconnected.
                - 'disconnected': The microphone has been disconnected.
                - 'streaming': The stream is streaming.
                - 'paused': The stream has been paused and is temporarily unavailable.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_status(status: str, data: dict):
                print(f"Microphone is now: {status}")
                print(f"Data: {data}")
                # Here you can add your code to react to the event

            microphone.on_status_changed(on_status)
        """
        if callback is None:
            self._on_status_changed_cb = None
        else:

            def _callback_wrapper(new_status: str, data: dict):
                try:
                    callback(new_status, data)
                except Exception as e:
                    self.logger.error(f"Callback for '{new_status}' status failed with error: {e}")

            self._on_status_changed_cb = _callback_wrapper

    @abstractmethod
    def _open_microphone(self) -> None:
        """Open the microphone connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _close_microphone(self) -> None:
        """Close the microphone connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _read_audio(self) -> np.ndarray | None:
        """Read a single audio chunk from the microphone. Must be implemented by subclasses."""
        pass

    def _set_status(self, new_status: Literal["disconnected", "connected", "streaming", "paused"], data: dict | None = None) -> None:
        """
        Updates the current status of the microphone and invokes the registered status
        changed callback in the background, if any.

        Only allowed states and transitions are considered, other states are ignored.
        Allowed states are:
            - disconnected
            - connected
            - streaming
            - paused

        Args:
            new_status (str): The name of the new status.
            data (dict): Additional data associated with the status change.
        """

        if self.status == new_status:
            return

        allowed_transitions = {
            "disconnected": ["connected"],
            "connected": ["disconnected", "streaming"],
            "streaming": ["paused", "disconnected"],
            "paused": ["streaming", "disconnected"],
        }

        # If new status is not in the state machine, ignore it
        if new_status not in allowed_transitions:
            return

        # Check if new_status is an allowed transition for the current status
        if new_status in allowed_transitions[self._status]:
            self._status = new_status
            if self._on_status_changed_cb is not None:
                self._event_executor.submit(self._on_status_changed_cb, new_status, data if data is not None else {})

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
