# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import alsaaudio
import numpy as np

from .base_microphone import BaseMicrophone
from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneOpenError, MicrophoneReadError, MicrophoneConfigError
from arduino.app_utils import Logger

logger = Logger("ALSAMicrophone")


class ALSAMicrophone(BaseMicrophone):
    """
    ALSA (Advanced Linux Sound Architecture) microphone implementation.

    This class handles local audio capture devices on Linux systems using ALSA.
    It supports explicit ALSA device names (e.g., "plughw:CARD=USB,DEV=0").
    """

    USB_MIC_1 = "usb:1"
    """Shorthand for the first USB microphone available."""
    USB_MIC_2 = "usb:2"
    """Shorthand for the second USB microphone available."""

    def __init__(
        self,
        device: str | int = 0,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = CHUNK_BALANCED,
        auto_reconnect: bool = True,
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
            auto_reconnect (bool, optional): Enable automatic reconnection on failure. Default: True.

        Raises:
            MicrophoneConfigError: If the format is not supported.
        """
        super().__init__(sample_rate, channels, format, chunk_size, auto_reconnect)

        # Determine ALSA format and numpy dtype based on format
        self._alsa_format, self._dtype = self._resolve_format_and_dtype(format)
        if self._alsa_format is None or self._dtype is None:
            raise MicrophoneConfigError(f"Unsupported ALSA format: {format}")

        try:
            self.device_stable_ref = self._resolve_stable_ref(device)  # e.g., "plughw:CARD=MyMic,DEV=0"
            self.name = self._resolve_name(self.device_stable_ref)  # Override parent name with a human-readable name
        except Exception as e:
            raise MicrophoneConfigError(f"Failed to look for microphone device '{device}': {e}")
        self.logger = logger

        self._pcm: Optional[alsaaudio.PCM] = None
        self._mixer: Optional[alsaaudio.Mixer] = None

        self._last_reconnection_attempt = 0.0  # Used for auto-reconnection when _read_frame is called

    def _resolve_format_and_dtype(self, format: str) -> Tuple[str | None, np.dtype | None]:
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
        return (af, np.dtype(nf)) if nf is not None else (None, None)

    def _resolve_stable_ref(self, identifier: str | int) -> str:
        """
        Resolve a microphone identifier to coordinates that are stable across
        reconnections and that don't depend on current running system state.

        Args:
            identifier: Microphone identifier

        Returns:
            str: stable reference to the microphone in ALSA device name format

        Raises:
            RuntimeError: If microphone can't be resolved
        """
        all_devices = self.list_devices()
        if not all_devices:
            raise RuntimeError("No ALSA microphones found")

        resolved_device = ""
        if isinstance(identifier, str) and not identifier.isdigit():
            from arduino.app_peripherals.microphone import Microphone  # Avoid circular import

            if identifier in (Microphone.USB_MIC_1, Microphone.USB_MIC_2):
                # Resolve USB microphone by ordinal index
                usb_index = int(identifier.replace("usb:", "")) - 1
                usb_devices = self.list_usb_devices()
                if not usb_devices:
                    raise RuntimeError("No USB microphones found")
                if usb_index < 0 or usb_index >= len(usb_devices):
                    raise RuntimeError(f"USB microphone index {usb_index + 1} out of range. Available: 1-{len(usb_devices)}")
                resolved_device = usb_devices[usb_index]

            elif identifier.startswith("/dev/snd/by-id"):
                # Already a stable link, resolve audio device following the symlink
                if not os.path.exists(identifier):
                    raise RuntimeError(f"{identifier} does not exist")
                device_path = os.path.realpath(identifier)  # Resolves to /dev/snd/controlCX
                base_name = os.path.basename(device_path)
                if base_name.startswith("controlC") and base_name[8:].isdigit():
                    card_idx = int(base_name[8:])
                    card_name = alsaaudio.card_name(card_idx)
                    if not isinstance(card_name, list) or len(card_name) == 0:
                        raise RuntimeError(f"Failed to resolve card name for card number {card_idx}")
                    resolved_device = f"plughw:CARD={card_name[0]},DEV=0"

            else:
                numeric_format_match = re.match(r"^(.+:)?(\d+),(\d+)$", identifier)
                if numeric_format_match:
                    try:
                        prefix = numeric_format_match.group(1)  # Returns None if no prefix
                        card_idx = int(numeric_format_match.group(2))
                        device_index = int(numeric_format_match.group(3))
                        card_name = alsaaudio.card_name(card_idx)
                        if not isinstance(card_name, list) or len(card_name) == 0:
                            raise RuntimeError(f"Failed to resolve card name for card number {card_idx}")
                        resolved_device = f"{prefix if prefix else ''}CARD={card_name[0]},DEV={device_index}"
                    except Exception as e:
                        raise RuntimeError(f"Failed to resolve card name for hw/plughw identifier {identifier}: {e}")

                card_name_format_match = re.match(r"^(.+:)?CARD=([^,]+),DEV=(\d+)$", identifier)
                if card_name_format_match:
                    # Already in stable name format
                    resolved_device = identifier if identifier.startswith("plughw:") or identifier.startswith("hw:") else f"plughw:{identifier}"

        elif isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            # Treat as /dev/controlC<card_idx>, resolve audio device by card number
            card_idx = int(identifier)
            card_name = alsaaudio.card_name(card_idx)
            if not isinstance(card_name, list) or len(card_name) == 0:
                raise RuntimeError(f"Failed to resolve card name for card number {card_idx}")
            resolved_device = f"plughw:CARD={card_name[0]},DEV=0"

        if resolved_device not in all_devices:
            raise RuntimeError(f"Resolved device '{resolved_device}' not found among available ALSA devices")

        if resolved_device:
            return resolved_device

        raise RuntimeError(f"Unsupported device identifier: {identifier}")

    def _resolve_runtime_ref(self, device_stable_ref: str) -> tuple[str | None, int, int]:
        """
        Resolve an ALSA device name to runtime prefix, card and device indexes
        that depend on current running system state.

        Args:
            device_stable_ref: ALSA device name

        Returns:
            tuple: (prefix, card_index, device_index)
                - prefix (str | None): Optional prefix (e.g., "plughw")
                - card_index (int): ALSA card index
                - device_index (int): ALSA device index

        Raises:
            RuntimeError: If microphone can't be resolved
        """
        card_indexes = alsaaudio.card_indexes()
        if len(card_indexes) == 0:
            raise RuntimeError("No ALSA sound cards found")

        match = re.match(r"^(.+:)?CARD=([^,]+),DEV=(\d+)$", device_stable_ref)
        if match:
            try:
                prefix = match.group(1)  # Returns None if no prefix
                card_name = match.group(2)
                device_index = int(match.group(3))
                for card_index in card_indexes:
                    names = alsaaudio.card_name(card_index)
                    if card_name in names:
                        return prefix.replace(":", "") if prefix else None, card_index, device_index

            except Exception as e:
                raise RuntimeError(f"Failed to resolve microphone runtime ref from stable ref {device_stable_ref}: {e}")

        raise RuntimeError(f"Invalid device reference for name resolution: {device_stable_ref}")

    def _resolve_name(self, device_stable_ref: str) -> str:
        """
        Resolve a human-readable name for the microphone whose stable path is provided
        by looking at ALSA card names. Falls back to the device stable ref
        (CARD=<card_name>,DEV=<device>) if no card name exists.

        Args:
            device_stable_ref: ALSA device name

        Returns:
            str: human readable name

        Raises:
            RuntimeError: If device name can't be resolved
        """
        # Match stable refs like "plughw:CARD=MyDevice,DEV=0" or "CARD=MyDevice,DEV=0"
        match = re.match(r"^(.+:)?CARD=([^,]+),DEV=(\d+)$", device_stable_ref)
        if match:
            try:
                card_name = match.group(2)
                return card_name
            except Exception as e:
                raise RuntimeError(f"Failed to resolve microphone name from stable ref {device_stable_ref}: {e}")

        raise RuntimeError(f"Invalid device reference for name resolution: {device_stable_ref} (type:{type(device_stable_ref)})")

    def _open_microphone(self) -> None:
        """Open the ALSA PCM device."""
        logger.debug(f"Opening PCM device: {self.device_stable_ref}")

        try:
            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=self.device_stable_ref,
            )
            self._pcm.setchannels(self.channels)
            self._pcm.setrate(self.sample_rate)
            self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
            self._pcm.setperiodsize(self.chunk_size)

            _, card_idx, device_idx = self._resolve_runtime_ref(self.device_stable_ref)
            if self._mixer is not None:
                self._mixer.close()
            self._mixer = alsaaudio.Mixer(f"card_{card_idx}_{device_idx}_mic_wr")  # Load mixer for volume control

        except alsaaudio.ALSAAudioError as e:
            if "busy" in str(e):
                raise MicrophoneOpenError(f"Microphone is busy. Close other audio applications and try again. ({self.device_stable_ref})")
            elif "mixer" in str(e):
                raise MicrophoneOpenError(f"Failed to open mixer for device ({self.device_stable_ref})")
            else:
                raise RuntimeError(f"ALSA error opening microphone: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error opening microphone: {e}")

        logger.debug(f"PCM opened with params: {self.device_stable_ref}, {self.sample_rate}Hz, {self.channels}ch, {self.format}")

    def _close_microphone(self) -> None:
        """Close the ALSA PCM device."""
        if self._pcm is not None:
            try:
                self._pcm.close()
            except Exception as e:
                logger.warning(f"Error closing PCM device: {e}")
            finally:
                self._pcm = None

    def _read_audio(self) -> np.ndarray | None:
        """Read a single audio chunk from the ALSA microphone.

        Automatically attempts to reconnect if the device is disconnected until the device is
        available again.
        """
        try:
            if self._pcm is None:
                if not self.auto_reconnect:
                    return None

                # Prevent spamming connection attempts
                current_time = time.monotonic()
                elapsed = current_time - self._last_reconnection_attempt
                if elapsed < self.auto_reconnect_delay:
                    time.sleep(self.auto_reconnect_delay - elapsed)
                self._last_reconnection_attempt = current_time

                self._open_microphone()
                self.logger.info(f"Successfully reopened microphone {self.name}")

            length, data = self._pcm.read()
            if length == 0:
                self.logger.debug("No audio data read from PCM device.")
                return None

            try:
                return np.frombuffer(data, dtype=self._dtype)
            except Exception as e:
                raise MicrophoneReadError(f"Error converting PCM data to numpy array: {e}")

        except (alsaaudio.ALSAAudioError, MicrophoneOpenError, MicrophoneReadError, Exception) as e:
            if self._is_device_disconnected():
                self.logger.error(
                    f"Failed to read from microphone {self.name}: {e}."
                    f"{' Retrying...' if self.auto_reconnect else ' Auto-reconnect is disabled, please restart the app.'}"
                )
                self._close_microphone()
                return None

            self.logger.error(f"Unexpected error reading audio: {e}")
            return None

    def _is_device_disconnected(self) -> bool:
        """Check if the device is still in the USB devices list."""
        try:
            usb_devices = self.list_devices()
            return self.device_stable_ref not in usb_devices
        except Exception as e:
            logger.debug(f"Error checking device status: {e}")
            return True  # Assume disconnected if we can't check

    def get_volume(self) -> int | None:
        """Get the current volume level of the microphone.

        Returns:
            int: Volume level (0-100). If no mixer is available, returns None.
        """
        if self._mixer is None:
            logger.warning("No mixer available for volume control")
            return None

        try:
            return self._mixer.getvolume(pcmtype=alsaaudio.PCM_CAPTURE)[0]
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
            self._mixer.setvolume(volume, pcmtype=alsaaudio.PCM_CAPTURE)
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error setting volume: {e}")
            return

    @staticmethod
    def list_devices() -> list:
        """Return a list of available ALSA microphones (plughw only).

        Returns:
            list: List of ALSA device names.
        """
        devices = []
        try:
            for dev in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                if dev.startswith("hw:CARD=") or dev.startswith("plughw:CARD="):
                    devices.append(dev)
        except Exception as e:
            logger.error(f"Error retrieving ALSA devices: {e}")
            return []

        return devices

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
                device_path = Path(f"/sys/class/sound/card{card_index}/device")
                if not device_path.exists():
                    continue

                try:
                    real_path = device_path.resolve()
                    if "usb" in str(real_path).lower():
                        # Find all plughw devices for this card
                        for dev in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                            if dev.startswith("plughw:CARD=") and f"CARD={card_name}" in dev:
                                usb_devices.append(dev)

                except Exception as e:
                    logger.error(f"Error parsing card info for {card_name}: {e}")

        except Exception as e:
            logger.error(f"Error listing USB microphones: {e}")

        return usb_devices
