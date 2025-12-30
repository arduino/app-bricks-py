# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Literal

import numpy as np

from .errors import SpeakerConfigError, SpeakerOpenError, SpeakerWriteError
from arduino.app_utils import Logger

logger = Logger("Speaker")


class BaseSpeaker(ABC):
    """
    Abstract base class for speaker implementations.

    This class defines the common interface that all speaker implementations must follow,
    providing a unified API regardless of the underlying audio playback protocol or type.

    The input is always a NumPy array with the ALSA PCM format.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        format: type | np.dtype | str,
        buffer_size: int,
        auto_reconnect: bool,
    ):
        """
        Initialize the speaker base.

        Args:
            sample_rate (int): Sample rate in Hz.
            channels (int): Number of audio channels.
            format (type | np.dtype | str): Audio format as numpy dtype, type, or string:
                - Type classes: np.int16, np.float32, np.uint8
                - Dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
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
        if format is None or (isinstance(format, str) and format.strip() == ""):
            raise SpeakerConfigError("Format must be a non-empty string")
        self.format: np.dtype = np.dtype(format)
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

    @property
    def volume(self) -> int:
        """
        Get or set the speaker volume level.

        This controls the hardware volume of the speaker device.

        Args:
            volume (int): Hardware volume level (0-100).

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

    def play(self, audio_chunk: np.ndarray):
        """
        Play an audio chunk on the speaker.

        Args:
            audio_chunk (np.ndarray): NumPy array in ALSA PCM format.

        Raises:
            SpeakerWriteError: If the speaker is not started.
            Exception: If the underlying implementation fails to write a frame.
        """
        with self._spkr_lock:
            if not self.is_started():
                raise SpeakerWriteError(f"Attempted to write to {self.name} before starting it.")

            if audio_chunk.dtype != self.format:
                raise SpeakerWriteError(f"Audio chunk with dtype {audio_chunk.dtype} does not match expected {self.format}")

            # Apply software volume control
            if self._volume != 1.0:
                audio_chunk = self._apply_volume_func(audio_chunk, self._volume)

            self._write_audio(audio_chunk)

    # TODO: add play_pcm method
    # TODO: add play_wav method

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
