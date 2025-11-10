# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Tuple

import alsaaudio
import numpy as np

from .base_microphone import BaseMicrophone
from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneOpenError, MicrophoneReadError, MicrophoneConfigError
from arduino.app_utils import Logger

logger = Logger("ALSAMicrophone")

# Reconnection delay in seconds when device is disconnected
RECONNECT_DELAY = 2.0


class ALSAMicrophone(BaseMicrophone):
    """
    ALSA (Advanced Linux Sound Architecture) microphone implementation.

    This class handles local audio capture devices on Linux systems using ALSA.
    It supports explicit ALSA device names (e.g., "plughw:CARD=USB,DEV=0").
    """

    def __init__(
        self,
        device: str | int = 0,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = CHUNK_BALANCED,
    ):
        """
        Initialize ALSA microphone.

        Args:
            device (Union[str, int]): ALSA device identifier. Can be:
                - int: Microphone index (e.g., 0, 1) - uses USB microphones list
                - str: Microphone index as string (e.g., "0", "1")
                - str: ALSA device name (e.g., "plughw:CARD=USB,DEV=0")
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format (default: "S16_LE").
            chunk_size (int): Number of frames per chunk (default: 1024).

        Raises:
            MicrophoneConfigError: If the format is not supported.
        """
        super().__init__(sample_rate, channels, format, chunk_size)

        # Determine ALSA format and numpy dtype based on format
        self._alsa_format, self._dtype = self._resolve_format_and_dtype(format)
        if self._alsa_format is None or self._dtype is None:
            raise MicrophoneConfigError(f"Unsupported ALSA format: {format}")

        self.device = self._resolve_device_name(device)
        self.logger = logger

        self._pcm: Optional[alsaaudio.PCM] = None
        self._mixer: Optional[alsaaudio.Mixer] = None
        self._native_rate = None
        self._reconnect_attempts = 0

    def _resolve_format_and_dtype(self, format: str) -> Tuple[str, np.dtype] | None:
        """Get numpy dtype for audio format."""
        # Mapping format string -> (ALSA PCM_FORMAT_*, numpy dtype)
        format_map = {
            "S8": ("PCM_FORMAT_S8", np.int8),
            "U8": ("PCM_FORMAT_U8", np.uint8),
            "S16_LE": ("PCM_FORMAT_S16_LE", "<i2"),
            "S16_BE": ("PCM_FORMAT_S16_BE", ">i2"),
            "U16_LE": ("PCM_FORMAT_U16_LE", "<u2"),
            "U16_BE": ("PCM_FORMAT_U16_BE", ">u2"),
            "S24_LE": ("PCM_FORMAT_S24_LE", "<i4"),  # 24-bit packed in 32-bit container
            "S24_BE": ("PCM_FORMAT_S24_BE", ">i4"),  # 24-bit packed in 32-bit container
            "S32_LE": ("PCM_FORMAT_S32_LE", "<i4"),
            "S32_BE": ("PCM_FORMAT_S32_BE", ">i4"),
            "U32_LE": ("PCM_FORMAT_U32_LE", "<u4"),
            "U32_BE": ("PCM_FORMAT_U32_BE", ">u4"),
            "FLOAT_LE": ("PCM_FORMAT_FLOAT_LE", "<f4"),
            "FLOAT_BE": ("PCM_FORMAT_FLOAT_BE", ">f4"),
            "FLOAT64_LE": ("PCM_FORMAT_FLOAT64_LE", "<f8"),
            "FLOAT64_BE": ("PCM_FORMAT_FLOAT64_BE", ">f8"),
        }
        af, nf = format_map.get(format, (None, None))
        return af, np.dtype(nf) if nf else None

    def _resolve_device_name(self, device: str | int) -> str:
        """
        Resolve microphone identifier to an ALSA device name.

        Args:
            device: Microphone identifier (int index or str device name)

        Returns:
            ALSA device name

        Raises:
            MicrophoneConfigError: If microphone cannot be resolved
        """
        if isinstance(device, int):
            # Get USB microphones list and use index
            usb_devices = self.list_usb_devices()
            if not usb_devices:
                raise MicrophoneConfigError("No USB microphones found")
            if device < 0 or device >= len(usb_devices):
                raise MicrophoneConfigError(f"Microphone index {device} out of range. Available: 0-{len(usb_devices) - 1}")
            return usb_devices[device]

        if isinstance(device, str):
            # If it's a numeric string, treat as index
            if device.isdigit():
                device_idx = int(device)
                usb_devices = self.list_usb_devices()
                if not usb_devices:
                    raise MicrophoneConfigError("No USB microphones found")
                if device_idx < 0 or device_idx >= len(usb_devices):
                    raise MicrophoneConfigError(f"Microphone index {device_idx} out of range. Available: 0-{len(usb_devices) - 1}")
                return usb_devices[device_idx]

            # Otherwise it's an explicit ALSA device name
            return device

        raise MicrophoneConfigError(f"Cannot resolve device identifier: {device}")

    def _open_microphone(self) -> None:
        """Open the ALSA PCM device."""
        logger.debug(f"Opening PCM device: {self.device}")

        try:
            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=self.device,
            )
            try:
                self._pcm.setchannels(self.channels)
                self._pcm.setrate(self.sample_rate)
                self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                self._pcm.setperiodsize(self.chunk_size)
                self._native_rate = self.sample_rate
                logger.debug(
                    "PCM opened with requested params: %s, %dHz, %dch, %s",
                    self.device,
                    self.sample_rate,
                    self.channels,
                    self.format,
                )
            except Exception as e:
                logger.warning(f"Requested params not supported on {self.device}: {e}")

                if not self.device.startswith("plughw"):
                    # Try fallback with plughw
                    plugdev = self.device.replace("hw", "plughw", 1) if self.device.startswith("hw") else f"plughw:{self.device}"
                    logger.debug(f"Trying fallback with plughw device: {plugdev}")
                    self._pcm = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NORMAL,
                        device=plugdev,
                    )
                    self._pcm.setchannels(self.channels)
                    self._pcm.setrate(self.sample_rate)
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    self._pcm.setperiodsize(self.chunk_size)
                    self.device = plugdev
                    self._native_rate = self.sample_rate

                    logger.debug(f"PCM opened with plughw fallback: {plugdev}")
                else:
                    logger.error(f"plughw fallback failed, using native device params for {self.device}")

                    self._pcm = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NORMAL,
                        device=self.device,
                    )
                    self._pcm.setchannels(self.channels)
                    self._native_rate = self._pcm.rate()
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    self._pcm.setperiodsize(self.chunk_size)

                    logger.debug("PCM opened with native params: %s, %dHz", self.device, self._native_rate)

        except alsaaudio.ALSAAudioError as e:
            logger.error(f"ALSAAudioError opening PCM device {self.device}: {e}")
            if "Device or resource busy" in str(e):
                raise MicrophoneOpenError(f"Microphone is busy. Close other audio applications and try again. ({self.device})")
            else:
                raise MicrophoneOpenError(f"ALSA error opening microphone: {e}")

        except Exception as e:
            logger.error(f"Unexpected error opening PCM device {self.device}: {e}")
            raise MicrophoneOpenError(f"Unexpected error opening microphone: {e}")

        self._mixer = self._load_mixer()  # Load mixer for volume control
        self._reconnect_attempts = 0

    def _load_mixer(self) -> Optional[alsaaudio.Mixer]:
        """Load the ALSA mixer for volume control."""
        try:
            cards = alsaaudio.cards()
            card_indexes = alsaaudio.card_indexes()
            for card_name, card_index in zip(cards, card_indexes):
                logger.debug(f"Checking Mic {card_name} (index {card_index}, device {self.device})")
                if f"CARD={card_name}," in self.device:
                    try:
                        mixer = alsaaudio.mixers(cardindex=card_index)
                        if len(mixer) == 0:
                            logger.warning(f"No mixers found for mic {card_name}.")
                            continue
                        mx = alsaaudio.Mixer(mixer[0], cardindex=card_index)
                        logger.debug(f"Loaded mixer: {mixer[0]} for mic {card_name}")
                        return mx
                    except alsaaudio.ALSAAudioError as e:
                        logger.debug(f"Failed to load mixer for mic {card_name}: {e}")

            # No suitable mixer found
            return None
        except alsaaudio.ALSAAudioError as e:
            logger.warning(f"Error loading mixer {self.device}: {e}")
            return None

    def _close_microphone(self) -> None:
        """Close the ALSA PCM device."""
        if self._pcm is not None:
            try:
                self._pcm.close()
            except Exception as e:
                logger.warning(f"Error closing PCM device: {e}")
            finally:
                self._pcm = None

    def _read_audio(self) -> Optional[np.ndarray]:
        """Read a single audio chunk from the ALSA microphone.

        Automatically attempts to reconnect if the device is disconnected until the device is
        available again.
        """
        if self._pcm is None:
            self._attempt_reconnection()
            if self._pcm is None:
                return None

        try:
            length, data = self._pcm.read()
            if length > 0:
                try:
                    arr = np.frombuffer(data, dtype=self._dtype)
                    self._reconnect_attempts = 0  # Reset on successful read
                    return arr
                except Exception as e:
                    logger.error(f"Error converting PCM data to numpy array: {e}")
                    return None
            else:
                logger.debug("No audio data read from PCM device.")
                return None

        except alsaaudio.ALSAAudioError as e:
            logger.warning(f"ALSA error reading audio: {e}")

            if self._is_device_disconnected():
                logger.error("Microphone disconnected")

                if self._pcm is not None:
                    try:
                        self._pcm.close()
                    except Exception:
                        pass
                    self._pcm = None

                return None

            raise MicrophoneReadError(f"Failed to read from ALSA microphone: {e}")

        except Exception as e:
            logger.error(f"Unexpected error reading audio: {e}")
            raise MicrophoneReadError(f"Unexpected error reading audio: {e}")

    def _is_device_disconnected(self) -> bool:
        """Check if the device is still in the USB devices list."""
        try:
            usb_devices = self.list_usb_devices()
            return self.device not in usb_devices
        except Exception as e:
            logger.debug(f"Error checking device status: {e}")
            return True  # Assume disconnected if we can't check

    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to the microphone device.

        Blocks and retries indefinitely with RECONNECT_DELAY between attempts.
        """
        import time

        self._reconnect_attempts = 0
        while True:
            self._reconnect_attempts += 1
            logger.debug(f"Waiting for microphone to be reconnected... (attempt {self._reconnect_attempts})")

            time.sleep(RECONNECT_DELAY)

            try:
                self._open_microphone()
                logger.info("Microphone reconnected successfully.")
                self._reconnect_attempts = 0
                return
            except (MicrophoneOpenError, MicrophoneConfigError) as e:
                logger.debug(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
                continue

    def get_volume(self) -> int:
        """Get the current volume level of the microphone.

        Returns:
            int: Volume level (0-100). If no mixer is available, returns None.
        """
        if self._mixer is None:
            logger.warning("No mixer available for volume control")
            return None

        try:
            volume = self._mixer.getvolume()[0]
            return volume
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error getting volume: {e}")
            return None

    def set_volume(self, volume: int):
        """Set the volume level of the microphone.

        Args:
            volume (int): Volume level (0-100).

        Raises:
            ValueError: If the volume is not between 0 and 100.
        """
        if self._mixer is None:
            logger.warning("No mixer available for volume control")
            return

        if not (0 <= volume <= 100):
            raise ValueError("Volume must be between 0 and 100.")

        try:
            self._mixer.setvolume(volume)
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error setting volume: {e}")
            return

    @staticmethod
    def list_devices() -> list:
        """Return a list of available ALSA capture devices.

        Returns:
            list: List of ALSA device names.
        """
        try:
            return alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
        except Exception as e:
            logger.error(f"Error retrieving ALSA devices: {e}")
            return []

    @staticmethod
    def list_usb_devices() -> list:
        """Return an ordered list of ALSA device names for available USB microphones (plughw only).

        Returns:
            list: List of USB microphone device names.
        """
        usb_devices = []
        try:
            cards = alsaaudio.cards()
            card_indexes = alsaaudio.card_indexes()
            card_map = {name: idx for idx, name in zip(card_indexes, cards)}
            for card_name, card_index in card_map.items():
                try:
                    desc = alsaaudio.card_name(card_index)
                    desc_str = desc[1] if isinstance(desc, tuple) else str(desc)
                    if "usb" in card_name.lower() or "usb" in desc_str.lower():
                        # Find all plughw devices for this card
                        for dev in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                            if dev.startswith("plughw:CARD=") and f"CARD={card_name}" in dev:
                                usb_devices.append(dev)
                except Exception as e:
                    logger.error(f"Error parsing card info for {card_name}: {e}")
        except Exception as e:
            logger.error(f"Error listing USB microphones: {e}")

        return usb_devices
