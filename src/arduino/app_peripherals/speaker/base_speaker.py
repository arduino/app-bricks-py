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

from .errors import SpeakerConfigError, SpeakerOpenError, SpeakerWriteError
from arduino.app_utils import Logger

logger = Logger("Speaker")

type FormatPlain = type | np.dtype | str
type FormatPacked = tuple[FormatPlain, bool]


class BaseSpeaker(ABC):
    """
    Abstract base class for speaker implementations.

    This class defines the common interface that all speaker implementations must follow,
    providing a unified API regardless of the underlying audio playback protocol or type.

    The input is always a NumPy array with the PCM format.
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
        Initialize the speaker base.

        Args:
            sample_rate (int): Sample rate in Hz.
            channels (int): Number of audio channels.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
            buffer_size (int): Size of the audio buffer.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.

        Raises:
            SpeakerConfigError: If the provided configuration is not valid.
        """
        if sample_rate <= 0:
            raise SpeakerConfigError("Sample rate must be positive")
        self.sample_rate = sample_rate

        if channels <= 0:
            raise SpeakerConfigError("Number of channels must be positive")
        self.channels = channels

        if format is None:
            raise SpeakerConfigError("Format must be specified")
        if isinstance(format, tuple):
            if len(format) != 2:
                raise SpeakerConfigError("Format tuple must be of the form (format: FormatPlain, is_packed: bool)")
            format, self.format_is_packed = format
        else:
            self.format_is_packed = False
        if isinstance(format, str) and format.strip() == "":
            raise SpeakerConfigError("Format must be a non-empty string or a valid numpy dtype/type or a tuple")
        try:
            self.format: np.dtype = np.dtype(format)
        except TypeError as e:
            raise SpeakerConfigError(f"Invalid format: {format}") from e

        if buffer_size <= 0:
            raise SpeakerConfigError("Buffer size must be positive")
        self.buffer_size = buffer_size

        self.logger = logger  # This will be overridden by subclasses if needed
        self.name = self.__class__.__name__  # This will be overridden by subclasses if needed

        self._volume: float = 1.0  # Software volume control (0.0 to 1.0)
        self._apply_volume_func = _create_volume_func(self.format)

        self._spkr_lock = threading.Lock()
        self._is_started = False

        # Auto-reconnection parameters
        self.auto_reconnect = auto_reconnect
        self.auto_reconnect_delay = 1.0
        self.first_connection_max_retries = 10

        # Status handling
        self._status: Literal["disconnected", "connected"] = "disconnected"
        self._on_status_changed_cb: Callable[[str, dict], None] | None = None
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SpeakerCallbacksRunner")

    @property
    def volume(self) -> int:
        """
        Get or set the speaker volume level.

        This controls the software volume of the speaker device.

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
    def status(self) -> Literal["disconnected", "connected"]:
        """Read-only property for camera status."""
        return self._status

    def start(self) -> None:
        """Start the speaker capture."""
        with self._spkr_lock:
            self.logger.info("Starting speaker...")

            attempt = 0
            while not self.is_started():
                try:
                    self._open_speaker()
                    self._is_started = True
                    self.logger.info(f"Successfully started {self.name}")
                except SpeakerOpenError as e:  # We consider this a fatal error so we don't retry
                    self.logger.error(f"Fatal error while starting {self.name}: {e}")
                    raise
                except Exception as e:
                    if not self.auto_reconnect:
                        raise
                    attempt += 1
                    if attempt >= self.first_connection_max_retries:
                        raise SpeakerOpenError(
                            f"Failed to start speaker {self.name} after {self.first_connection_max_retries} attempts, last error is: {e}"
                        )

                    delay = min(self.auto_reconnect_delay * (2 ** (attempt - 1)), 60)  # Exponential backoff
                    self.logger.warning(
                        f"Failed attempt {attempt}/{self.first_connection_max_retries} at starting speaker {self.name}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

    def stop(self) -> None:
        """Stop the speaker and release resources."""
        with self._spkr_lock:
            if not self.is_started():
                return

            self.logger.info("Stopping speaker...")

            try:
                self._close_speaker()
                self._event_executor.shutdown()
                self._is_started = False
                self.logger.info(f"Successfully stopped {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to stop speaker: {e}")

    def play(self, audio_chunk: np.ndarray):
        """
        Play an audio chunk on the speaker.

        Args:
            audio_chunk (np.ndarray): NumPy array in PCM format.

        Raises:
            SpeakerWriteError: If the speaker is not started.
            ValueError: If audio_chunk is empty or invalid.
            Exception: If the underlying implementation fails to write a frame.
        """
        with self._spkr_lock:
            if not self.is_started():
                raise SpeakerWriteError(f"Attempted to write to {self.name} before starting it.")

            if audio_chunk is None or len(audio_chunk) == 0:
                raise ValueError("Audio data must not be empty.")

            if audio_chunk.dtype != self.format:
                raise ValueError(f"Audio data with dtype {audio_chunk.dtype} does not match expected {self.format}.")

            # Apply software volume control
            if self._volume != 1.0:
                audio_chunk = self._apply_volume_func(audio_chunk, self._volume)

            self._write_audio(audio_chunk)

    def play_pcm(self, pcm_audio: np.ndarray) -> None:
        """
        Play raw PCM audio data.

        Args:
            pcm_audio (np.ndarray): Raw PCM audio data in PCM format.

        Raises:
            SpeakerOpenError: If speaker can't be opened or reopened.
            SpeakerWriteError: If speaker is not started.
            ValueError: If pcm_audio is empty or invalid.
            Exception: If the underlying implementation fails to write a frame.
        """
        if pcm_audio is None or len(pcm_audio) == 0:
            raise ValueError("Audio data cannot be empty")

        if pcm_audio.dtype != self.format:
            raise ValueError(f"Audio data with dtype {pcm_audio.dtype} does not match expected {self.format}")

        offset = 0
        total_samples = len(pcm_audio)
        while offset < total_samples:
            chunk_size = min(self.buffer_size * self.channels, total_samples - offset)
            chunk = pcm_audio[offset : offset + chunk_size]

            self.play(chunk)

            offset += chunk_size

    def play_wav(self, wav_audio: np.ndarray) -> None:
        """
        Play audio from WAV format data.
        Note: Only uncompressed PCM WAV files are supported.

        Args:
            wav_audio (np.ndarray): WAV format audio data (including header).

        Raises:
            SpeakerOpenError: If speaker can't be opened or reopened.
            SpeakerWriteError: If speaker is not started.
            ValueError: If wav_audio is empty or invalid.
            Exception: If the underlying implementation fails to write a frame.
        """
        pcm_audio = self._wav_to_pcm(wav_audio)
        self.play_pcm(pcm_audio)

    def _wav_to_pcm(self, wav_audio: np.ndarray) -> np.ndarray:
        """
        Convert WAV format data to raw PCM audio data.

        This is the inverse of the _audio_to_wav method in the microphone class.

        Args:
            wav_audio (np.ndarray): WAV format audio data (including header).

        Returns:
            np.ndarray: Raw PCM audio data in ALSA PCM format.

        Raises:
            ValueError: If WAV data is invalid or format is unsupported.
        """
        import io
        import wave

        if wav_audio is None or len(wav_audio) == 0:
            raise ValueError("WAV data cannot be empty")

        # Read WAV from numpy array
        buffer = io.BytesIO(wav_audio.tobytes())
        try:
            # By opening the wav file, wave will also validate the PCM format for us
            with wave.open(buffer, "rb") as wav_file:
                if wav_file.getcomptype() != "NONE":
                    raise ValueError(f"Unsupported WAV compression type: {wav_file.getcomptype()}. Only uncompressed PCM format is supported.")

                wav_channels = wav_file.getnchannels()
                wav_sampwidth = wav_file.getsampwidth()
                wav_framerate = wav_file.getframerate()
                wav_frames = wav_file.readframes(wav_file.getnframes())

                if wav_channels != self.channels:
                    raise ValueError(f"WAV channels ({wav_channels}) do not match speaker channels ({self.channels})")
                if wav_framerate != self.sample_rate:
                    raise ValueError(f"WAV sample rate ({wav_framerate}Hz) does not match speaker sample rate ({self.sample_rate}Hz)")

                spk_dtype_kind = self.format.kind
                spk_dtype_size = self.format.itemsize

                # Convert based on expected output format
                if spk_dtype_kind == "i":  # Signed integer
                    if spk_dtype_size == 1:  # int8
                        if wav_sampwidth != 1:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with int8 format")
                        # WAV stores 8-bit as unsigned - must convert
                        wav_array = np.frombuffer(wav_frames, dtype=np.uint8)
                        pcm_audio = (wav_array.astype(np.int16) - 128).astype(np.int8)
                    elif spk_dtype_size == 2:  # int16
                        if wav_sampwidth != 2:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with int16 format")
                        pcm_audio = np.frombuffer(wav_frames, dtype="<i2")
                    elif spk_dtype_size == 4:  # int32 or int24 in int32 container
                        # Check if this is 24-bit audio
                        if wav_sampwidth == 3:
                            # Need to pack 24-bit samples into 32-bit containers (LSB padding per ALSA)
                            import sys

                            wav_bytes = np.frombuffer(wav_frames, dtype=np.uint8)
                            num_samples = len(wav_bytes) // 3
                            audio_bytes = np.zeros(num_samples * 4, dtype=np.uint8)

                            if sys.byteorder == "little":
                                # On LE system: LSB padding goes at byte 0, audio bytes at 1-3
                                audio_bytes.reshape(-1, 4)[:, 1:4] = wav_bytes.reshape(-1, 3)
                            else:
                                # On BE system: LSB padding goes at byte 3, audio bytes at 0-2
                                audio_bytes.reshape(-1, 4)[:, :3] = wav_bytes.reshape(-1, 3)

                            pcm_audio = audio_bytes.view(np.int32)
                        elif wav_sampwidth == 4:
                            # True 32-bit audio
                            pcm_audio = np.frombuffer(wav_frames, dtype="<i4")
                        else:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with int32 format (expected 3 or 4)")
                    else:
                        raise ValueError(f"Unsupported signed integer size: {spk_dtype_size} bytes. Supported: 1, 2, 4.")

                elif spk_dtype_kind == "u":  # Unsigned integer
                    if spk_dtype_size == 1:  # uint8
                        if wav_sampwidth != 1:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with uint8 format")
                        pcm_audio = np.frombuffer(wav_frames, dtype=np.uint8)
                    elif spk_dtype_size == 2:  # uint16
                        if wav_sampwidth != 2:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with uint16 format")
                        # WAV stores 16-bit as signed - must convert
                        wav_array = np.frombuffer(wav_frames, dtype="<i2")
                        pcm_audio = (wav_array.astype(np.int32) + 32768).astype(np.uint16)
                    elif spk_dtype_size == 4:  # uint32
                        if wav_sampwidth != 4:
                            raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with uint32 format")
                        # WAV stores 32-bit as signed - must convert
                        wav_array = np.frombuffer(wav_frames, dtype="<i4")
                        pcm_audio = (wav_array.astype(np.int64) + 2147483648).astype(np.uint32)
                    else:
                        raise ValueError(f"Unsupported unsigned integer size: {spk_dtype_size} bytes. Supported: 1, 2, 4.")

                elif spk_dtype_kind == "f":  # Float
                    # WAV stores as int16 or int32, need to convert to normalized float [-1.0, 1.0]
                    if wav_sampwidth == 2:
                        wav_array = np.frombuffer(wav_frames, dtype="<i2")
                        pcm_audio = (wav_array.astype(self.format) / 32767.0).astype(np.float32)
                    elif wav_sampwidth == 4:
                        wav_array = np.frombuffer(wav_frames, dtype="<i4")
                        pcm_audio = (wav_array.astype(self.format) / 2147483647.0).astype(np.float64)
                    else:
                        raise ValueError(f"WAV sample width ({wav_sampwidth}) incompatible with float format (expected 2 or 4)")

                else:
                    raise ValueError(f"Unsupported audio data type: {self.format}. Supported: int8/16/32, uint8/16/32, float32/64.")

                # Handle output byte order if needed
                if self.format.byteorder not in ("=", "|"):
                    pcm_audio = pcm_audio.astype(self.format.newbyteorder(self.format.byteorder))

                return pcm_audio

        except wave.Error as e:
            raise ValueError(f"Invalid WAV data: {e}")

        except Exception as e:
            raise ValueError(f"Error converting WAV to audio: {e}")

    def is_started(self) -> bool:
        """Check if the speaker is started."""
        return self._is_started

    def on_status_changed(self, callback: Callable[[str, dict], None] | None):
        """Registers or removes a callback to be triggered on speaker lifecycle events.

        When a speaker status changes, the provided callback function will be invoked.
        If None is provided, the callback will be removed.

        Args:
            callback (Callable[[str, dict], None]): A callback that will be called every time the
                speaker status changes with the new status and any associated data. The status
                names depend on the actual speaker implementation being used. Some common events
                are:
                - 'connected': The speaker has been reconnected.
                - 'disconnected': The speaker has been disconnected.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_status(status: str, data: dict):
                print(f"Speaker is now: {status}")
                print(f"Data: {data}")
                # Here you can add your code to react to the event

            speaker.on_status_changed(on_status)
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
    def _open_speaker(self):
        """Open the speaker connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _close_speaker(self):
        """Close the speaker connection. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _write_audio(self, audio_chunk: np.ndarray):
        """Write a single audio chunk to the speaker. Must be implemented by subclasses."""
        pass

    def _set_status(self, new_status: Literal["disconnected", "connected"], data: dict | None = None) -> None:
        """
        Updates the current status of the speaker and invokes the registered status
        changed callback in the background, if any.

        Only allowed states and transitions are considered, other states are ignored.
        Allowed states are:
            - disconnected
            - connected

        Args:
            new_status (str): The name of the new status.
            data (dict): Additional data associated with the status change.
        """

        if self.status == new_status:
            return

        allowed_transitions = {
            "disconnected": ["connected"],
            "connected": ["disconnected"],
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
