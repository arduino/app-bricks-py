# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import threading
from typing import Literal
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .errors import MicrophoneConfigError, MicrophoneOpenError, MicrophoneReadError
from arduino.app_utils import Logger

logger = Logger("Microphone")

type FormatPlain = type | np.dtype | str
type FormatPacked = tuple[FormatPlain, bool]


class BaseMicrophone(ABC):
    """
    Abstract base class for microphone implementations.

    This class defines the common interface that all microphone implementations must follow,
    providing a unified API regardless of the underlying audio capture protocol or type.

    The output is always a NumPy array with the ALSA PCM format.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        format: FormatPlain | FormatPacked,
        buffer_size: int,
        auto_reconnect: bool,
    ):
        """
        Initialize the microphone base.

        Args:
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
            buffer_size (int): Size of the audio buffer.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.
        """
        if sample_rate <= 0:
            raise MicrophoneConfigError("Sample rate must be positive")
        self.sample_rate = sample_rate

        if channels <= 0:
            raise MicrophoneConfigError("Number of channels must be positive")
        self.channels = channels

        if format is None:
            raise MicrophoneConfigError("Format must be specified")
        if isinstance(format, tuple):
            if len(format) != 2:
                raise MicrophoneConfigError("Format tuple must be of the form (format: FormatPlain, is_packed: bool)")
            format, self.format_is_packed = format
        else:
            self.format_is_packed = False
        if isinstance(format, str) and format.strip() == "":
            raise MicrophoneConfigError("Format must be a non-empty string or a valid numpy dtype/type or a tuple")
        try:
            self.format: np.dtype = np.dtype(format)
        except TypeError as e:
            raise MicrophoneConfigError(f"Invalid format: {format}") from e

        if buffer_size <= 0:
            raise MicrophoneConfigError("Buffer size must be positive")
        self.buffer_size = buffer_size

        self.logger = logger  # This will be overridden by subclasses if needed
        self.name = self.__class__.__name__  # This will be overridden by subclasses if needed

        self._volume: float = 1.0  # Software volume control (0.0 to 1.0)
        self._apply_volume_func = _create_volume_func(self.format)

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
    def volume(self) -> int:
        """
        Get or set the microphone volume level.

        This controls the software volume of the microphone device.

        Args:
            volume (int): Software volume level (0-100).

        Returns:
            int: Current volume level (0-100).

        Raises:
            ValueError: If the volume is not valid.
        """
        return int(self._volume * 100)

    @volume.setter
    def volume(self, volume: int):
        if not (0 <= volume <= 100):
            raise ValueError("Volume must be between 0 and 100.")

        self._volume = volume / 100.0

    @property
    def status(self) -> Literal["disconnected", "connected", "streaming", "paused"]:
        """Read-only property for camera status."""
        return self._status

    @property
    def _none_chunk_threshold(self) -> int:
        """Heuristic: 750ms of empty chunk based on current sample rate. Always at least 10."""
        return max(10, int(0.75 * self.sample_rate) // self.buffer_size)

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

            audio_chunk = self._read_audio()
            if audio_chunk is None:
                self._consecutive_none_chunks += 1
                if self._consecutive_none_chunks >= self._none_chunk_threshold:
                    self._set_status("paused")
                return None

            self._set_status("streaming")

            self._consecutive_none_chunks = 0

            if audio_chunk.dtype != self.format:
                raise MicrophoneReadError(f"Input audio chunk with dtype {audio_chunk.dtype} does not match expected {self.format}")

            # Apply software volume control
            if self._volume != 1.0:
                audio_chunk = self._apply_volume_func(audio_chunk, self._volume)

            return audio_chunk

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

    def record_pcm(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration and return as raw PCM format.

        Args:
            duration (float): Recording duration in seconds.

        Returns:
            np.ndarray: Raw audio data in raw ALSA PCM format.

        Raises:
            MicrophoneOpenError: If microphone can't be opened or reopened.
            MicrophoneReadError: If no audio is available after multiple attempts.
            ValueError: If duration is not > 0.
            Exception: If the underlying implementation fails to read a frame.
        """
        if duration <= 0:
            raise ValueError("Duration must be > 0")

        # Get dtype from first chunk
        first_chunk = None
        attempts = 0
        max_attempts = 10
        while first_chunk is None and attempts < max_attempts:
            first_chunk = self.capture()
            if first_chunk is None:
                attempts += 1
                time.sleep(0.01)
        if first_chunk is None:
            raise MicrophoneReadError(f"Failed to read first audio chunk after {max_attempts} attempts.")

        offset = 0
        # Allocate the full recording buffer
        total_samples = int(duration * self.sample_rate * self.channels)
        recording = np.zeros(total_samples, dtype=first_chunk.dtype)
        while offset < total_samples:
            chunk = self.capture()
            if chunk is None:
                # Produce a synthetic "silence" chunk of audio
                chunk = np.zeros(self.buffer_size, dtype=first_chunk.dtype)

            chunk_len = len(chunk)
            if offset + chunk_len > total_samples:
                chunk_len = total_samples - offset
                recording[offset : offset + chunk_len] = chunk[:chunk_len]
                break
            recording[offset : offset + chunk_len] = chunk
            offset += chunk_len

        return recording

    def record_wav(self, duration: float) -> np.ndarray:
        """
        Record audio for a specified duration and return as WAV format.
        Note: Only uncompressed PCM WAV recordings are supported.

        Args:
            duration (float): Recording duration in seconds.

        Returns:
            np.ndarray: Raw audio data in WAV format as numpy array.

        Raises:
            MicrophoneOpenError: If microphone can't be opened or reopened.
            MicrophoneReadError: If no audio is available after multiple attempts.
            ValueError: If duration is not > 0.
            Exception: If the underlying implementation fails to read a frame.
        """
        pcm_data = self.record_pcm(duration)
        return self._pcm_to_wav(pcm_data)

    def _pcm_to_wav(self, pcm_audio: np.ndarray) -> np.ndarray:
        """
        Convert raw PCM audio data to WAV format.

        Args:
            pcm_audio (np.ndarray): Raw PCM audio data to convert.

        Returns:
            np.ndarray: WAV data as uint8 numpy array (including header).
        """
        import io
        import wave

        # Get base dtype kind and size
        mic_dtype_kind = pcm_audio.dtype.kind
        mic_dtype_size = pcm_audio.dtype.itemsize

        # Convert to native byte order since wave automatically handles byte ordering
        if pcm_audio.dtype.byteorder not in ("=", "|"):
            pcm_audio = pcm_audio.astype(pcm_audio.dtype.newbyteorder("="))

        if mic_dtype_kind == "i":  # Signed integer
            if mic_dtype_size == 1:  # int8
                # WAV uses unsigned 8-bit - must convert
                wav_audio = (pcm_audio.astype(np.int16) + 128).astype(np.uint8)
                sampwidth = 1
            elif mic_dtype_size == 2:  # int16
                wav_audio = pcm_audio
                sampwidth = 2
            elif mic_dtype_size == 4:  # int32
                # Check if this is 24-bit audio packed in 32-bit containers
                is_24bit = self.format_is_packed
                if is_24bit:
                    # Extract 24-bit samples from 32-bit containers (padding is in LSB per ALSA)
                    import sys

                    bytes_view = pcm_audio.view("u1").reshape(-1, 4)  # Reshape to rows of 4 bytes
                    if sys.byteorder == "little":
                        # On LE system: LSB padding is at byte 0, take bytes 1-3
                        wav_audio = bytes_view[:, 1:4].flatten()
                    else:
                        # On BE system: LSB padding is at byte 3, take bytes 0-2
                        wav_audio = bytes_view[:, :3].flatten()
                    sampwidth = 3
                else:
                    # True 32-bit audio
                    wav_audio = pcm_audio
                    sampwidth = 4
            else:
                raise ValueError(f"Unsupported signed integer size: {mic_dtype_size} bytes. Supported: 1, 2, 4.")

        elif mic_dtype_kind == "u":  # Unsigned integer
            if mic_dtype_size == 1:  # uint8
                # Already in correct format for WAV
                wav_audio = pcm_audio
                sampwidth = 1
            elif mic_dtype_size == 2:  # uint16
                # Convert to signed int16
                wav_audio = (pcm_audio.astype(np.int32) - 32768).astype(np.int16)
                sampwidth = 2
            elif mic_dtype_size == 4:  # uint32
                # Convert to signed int32
                wav_audio = (pcm_audio.astype(np.int64) - 2147483648).astype(np.int32)
                sampwidth = 4
            else:
                raise ValueError(f"Unsupported unsigned integer size: {mic_dtype_size} bytes. Supported: 1, 2, 4.")

        elif mic_dtype_kind == "f":  # Float
            # ALSA float formats are normalized [-1.0, 1.0]
            # float32 -> int16 (standard CD quality, ~24 bits precision -> 16 bits)
            # float64 -> int32 (high quality, ~53 bits precision -> 32 bits)
            wav_audio = np.clip(pcm_audio, -1.0, 1.0)
            if mic_dtype_size == 4:  # float32
                wav_audio = (wav_audio * 32767).astype(np.int16)
                sampwidth = 2
            elif mic_dtype_size == 8:  # float64
                wav_audio = (wav_audio * 2147483647).astype(np.int32)
                sampwidth = 4
            else:
                raise ValueError(f"Unsupported float size: {mic_dtype_size} bytes. Supported: 4, 8.")

        else:
            raise ValueError(f"Unsupported audio data type: {pcm_audio.dtype}. Supported: int8/16/32, uint8/16/32, float32/64.")

        # Write to in-memory buffer
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframesraw(wav_audio.tobytes())

        # Convert to numpy uint8 array
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

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


def _create_volume_func(dtype: np.dtype) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Create a volume application function based on dtype that can be cached.

    Args:
        dtype (np.dtype): Numpy data type of the audio samples.

    Returns:
        Callable[[np.ndarray, float], np.ndarray]: a function takes audio_chunk and
            volume and returns volume-adjusted audio.
    """
    # For floats, just multiply
    if np.issubdtype(dtype, np.floating):

        def apply_volume_float(audio_chunk: np.ndarray, volume: float) -> np.ndarray:
            if volume == 0.0:
                return np.zeros_like(audio_chunk)
            return audio_chunk * volume

        return apply_volume_float

    # For integers, convert to float, apply volume, convert back with clipping
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        max_val = float(info.max)
        min_val = float(info.min)

        def apply_volume_signed(audio_chunk: np.ndarray, volume: float) -> np.ndarray:
            if volume == 0.0:
                return np.zeros_like(audio_chunk)
            audio_float = audio_chunk.astype(np.float64) * volume
            return np.clip(audio_float, min_val, max_val).astype(dtype)

        return apply_volume_signed

    # For unsigned integers, center around midpoint before applying volume
    if np.issubdtype(dtype, np.unsignedinteger):
        info = np.iinfo(dtype)
        max_val = float(info.max)
        midpoint = max_val / 2.0

        def apply_volume_unsigned(audio_chunk: np.ndarray, volume: float) -> np.ndarray:
            if volume == 0.0:
                return np.zeros_like(audio_chunk)
            audio_centered = audio_chunk.astype(np.float64) - midpoint
            audio_scaled = audio_centered * volume + midpoint
            return np.clip(audio_scaled, 0, max_val).astype(dtype)

        return apply_volume_unsigned

    # Fallback: no volume adjustment
    def apply_volume_passthrough(audio_chunk: np.ndarray, volume: float) -> np.ndarray:
        return audio_chunk

    return apply_volume_passthrough
