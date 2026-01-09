# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
import time
from pathlib import Path
from typing import Optional

import alsaaudio
import numpy as np

from .base_microphone import BaseMicrophone, FormatPlain, FormatPacked
from .errors import MicrophoneOpenError, MicrophoneReadError, MicrophoneConfigError
from arduino.app_utils.logger import Logger

logger = Logger("ALSAMicrophone")


class ALSAMicrophone(BaseMicrophone):
    """
    ALSA (Advanced Linux Sound Architecture) microphone implementation.

    This class handles local audio capture devices on Linux systems using ALSA.
    """

    from .microphone import Microphone

    def __init__(
        self,
        device: str | int = Microphone.USB_MIC_1,
        sample_rate: int = Microphone.RATE_16K,
        channels: int = Microphone.CHANNELS_MONO,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = Microphone.BUFFER_SIZE_BALANCED,
        shared: bool = True,
        auto_reconnect: bool = True,
    ):
        """
        Initialize ALSA microphone.

        Args:
            device (Union[str, int]): ALSA device identifier. Can be:
                - int | str: device ordinal index (e.g., 0, 1, "0", "1", ...)
                - str: device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Microphone.USB_MIC_x macros
                Default: Microphone.USB_MIC_1 - First USB microphone.
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
                Default: np.int16 - 16-bit signed platform-endian.
            buffer_size (int): Size of the audio buffer (default: 1024).
            shared (bool): ALSA device sharing mode.
                - False: Opens the device in exclusive mode to provide lowest latency
                    but another application will fail when this instance is using the device.
                - True: Opens the device in shared mode to allow other applications to use
                    it at the same time but introduces higher latency. Will fail when another
                    instance is already using the device in exclusive mode.
                Default: True.
            auto_reconnect (bool, optional): Enable automatic reconnection on failure.
                Default: True.

            Note: When shared=True, only higher buffer size values are supported due to
                ALSA limitations (~2000).

        Raises:
            MicrophoneConfigError: If the format is not supported.
        """
        super().__init__(sample_rate, channels, format, buffer_size, auto_reconnect)

        try:
            self.device_stable_ref = self._resolve_stable_ref(device)  # e.g., "plughw:CARD=MyMic,DEV=0"
            self.name = self._resolve_name(self.device_stable_ref)  # Override parent name with a human-readable name
        except Exception as e:
            raise MicrophoneConfigError(f"Failed to look for microphone device '{device}': {e}")
        self.shared = shared
        self.logger = logger

        self._pcm: Optional[alsaaudio.PCM] = None

        self._last_reconnection_attempt = 0.0  # Used for auto-reconnection when _read_audio is called

    @property
    def alsa_format_idx(self) -> int:
        """Get the ALSA format index corresponding to the current numpy dtype format."""
        return getattr(alsaaudio, "PCM_FORMAT_" + self.alsa_format_name)

    @property
    def alsa_format_name(self) -> str:
        """Get the ALSA format string corresponding to the current numpy dtype format."""
        return _dtype_to_alsa_format_name(self.format, self.format_is_packed)

    @staticmethod
    def list_devices() -> list:
        """
        Return a list of available ALSA microphones (plughw only).

        Returns:
            list: List of speakers in ALSA device name format.
        """
        devices = []
        try:
            for dev in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                if dev.startswith("plughw:CARD="):
                    devices.append(dev.removeprefix("plughw:"))
        except Exception as e:
            logger.error(f"Error retrieving ALSA devices: {e}")
            return []

        return devices

    @staticmethod
    def list_usb_devices() -> list:
        """
        Return a list of available USB ALSA microphones (plughw only).

        Returns:
            list: List of USB microphones in ALSA device name format.
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
                        # Find all hw and plughw devices for this card
                        for dev in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                            if dev.startswith("plughw:CARD=") and f"CARD={card_name}," in dev:
                                usb_devices.append(dev.removeprefix("plughw:"))

                except Exception as e:
                    logger.error(f"Error parsing card info for {card_name}: {e}")

        except Exception as e:
            logger.error(f"Error listing USB microphones: {e}")

        return usb_devices

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
            if identifier.startswith("usb:"):
                # Resolve USB microphone by ordinal index
                usb_index = int(identifier.removeprefix("usb:")) - 1
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
                    card_name = self._resolve_name(card_idx)
                    resolved_device = f"CARD={card_name},DEV=0"

            else:
                numeric_format_match = re.match(r"^(.+:)?(\d+),(\d+)$", identifier)
                if numeric_format_match:
                    try:
                        card_idx = int(numeric_format_match.group(2))
                        device_index = int(numeric_format_match.group(3))
                        card_name = self._resolve_name(card_idx)
                        resolved_device = f"CARD={card_name},DEV={device_index}"
                    except Exception as e:
                        raise RuntimeError(f"Failed to resolve card name for hw/plughw identifier {identifier}: {e}")

                card_name_format_match = re.match(r"^(.+:)?CARD=([^,]+),DEV=(\d+)$", identifier)
                if card_name_format_match:
                    if card_name_format_match.group(1) is not None:
                        # Remove prefix like "plughw:" or "hw:"
                        resolved_device = identifier.split(":", 1)[-1]
                    else:
                        # Already in stable name format
                        resolved_device = identifier

        elif isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            # Treat as /dev/controlC<card_idx>, resolve audio device by card number
            card_idx = int(identifier)
            card_name = self._resolve_name(card_idx)
            resolved_device = f"CARD={card_name},DEV=0"

        if resolved_device:
            if resolved_device not in all_devices:
                raise RuntimeError(f"Resolved device '{resolved_device}' not found among available ALSA devices")
            return resolved_device

        raise RuntimeError(f"Unsupported device identifier: {identifier}")

    def _resolve_runtime_ref(self, device_stable_ref: str) -> tuple[int, int]:
        """
        Resolve an ALSA device name to runtime card and device indexes that
        depend on current running system state.

        Args:
            device_stable_ref: ALSA device name

        Returns:
            tuple: (card_index, device_index)
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
                card_name = match.group(2)
                device_index = int(match.group(3))
                for card_idx, curr_card_name in enumerate(alsaaudio.cards()):  # Look for the card index
                    if curr_card_name == card_name:
                        return card_idx, device_index

            except Exception as e:
                raise RuntimeError(f"Failed to resolve microphone runtime ref from stable ref {device_stable_ref}: {e}")

        raise RuntimeError(f"Invalid device reference for name resolution: {device_stable_ref}")

    def _resolve_name(self, device_ref: str | int) -> str:
        """
        Resolve a human-readable name for the microphone whose stable path or index
        is provided by looking at ALSA card names.

        Args:
            device_ref (str | int): ALSA device name (str) or card index (int)

        Returns:
            str: compact, human readable name

        Raises:
            RuntimeError: If device name can't be resolved
        """
        if isinstance(device_ref, str):
            # This is a card stable refs like "plughw:CARD=MyDevice,DEV=0" or "CARD=MyDevice,DEV=0"
            match = re.match(r"^(.+:)?CARD=([^,]+),DEV=(\d+)$", device_ref)
            if match:
                try:
                    card_name = match.group(2)
                    return card_name
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve microphone name from stable ref {device_ref}: {e}")

        elif isinstance(device_ref, int):
            # This is a card index like 0, 1, ...
            cards = alsaaudio.cards()
            if device_ref < 0 or device_ref >= len(cards):
                raise RuntimeError(f"Card index {device_ref} out of range. Available: 0-{len(cards) - 1}")
            card_name = cards[device_ref]
            return card_name

        raise RuntimeError(f"Invalid device reference for name resolution: {device_ref} (type:{type(device_ref)})")

    def _open_microphone(self) -> None:
        """Open the ALSA PCM device."""
        logger.debug(f"Opening PCM device: {self.device_stable_ref}")

        try:
            card_idx, device_idx = self._resolve_runtime_ref(self.device_stable_ref)
            if self.shared:
                device = f"plug_card_{card_idx}_dev_{device_idx}_mic"
            else:
                device = f"plughw:CARD={card_idx},DEV={device_idx}"

            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=device,
                rate=self.sample_rate,
                channels=self.channels,
                format=self.alsa_format_idx,
                periodsize=self.buffer_size,
            )

            info = self._pcm.info()

            actual_rate = info["rate"]
            if self.sample_rate != actual_rate:
                logger.warning(f"Requested sample rate {self.sample_rate}Hz not supported by {device}. Using {actual_rate}Hz instead.")
                self.sample_rate = actual_rate

            actual_channels = info["channels"]
            if self.channels != actual_channels:
                logger.warning(f"Requested channels {self.channels} not supported by {device}. Using {actual_channels} instead.")
                self.channels = actual_channels

            actual_format_name = info["format_name"]
            if self.alsa_format_idx != info["format"]:
                logger.warning(f"Requested format {self.alsa_format_name} not supported by {device}. Using {actual_format_name} instead.")
                self.format = _alsa_format_name_to_dtype(actual_format_name)

            actual_buffer_size = info["period_size"]
            if self.buffer_size != actual_buffer_size:
                logger.warning(f"Requested buffer_size {self.buffer_size} not supported by {device}. Using {actual_buffer_size} instead.")
                self.buffer_size = actual_buffer_size

        except MicrophoneOpenError:
            raise

        except alsaaudio.ALSAAudioError as e:
            if "busy" in str(e):
                raise MicrophoneOpenError(f"Microphone is busy. Close other audio applications and try again. ({self.device_stable_ref})")
            else:
                raise RuntimeError(f"ALSA error opening microphone: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error opening microphone: {e}")

        logger.debug(f"PCM opened with params: {device}, {self.sample_rate}Hz, {self.channels}ch, {self.format}, {self.buffer_size} frames/IO")

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
        """
        Read a single audio chunk from the ALSA microphone.

        Automatically attempts to reconnect if the device is disconnected
        until the device is available again.
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

            length, audio_chunk = self._pcm.read()
            if length == 0:
                self.logger.debug("No audio data read from PCM device.")
                return None

            try:
                return np.frombuffer(audio_chunk, dtype=self.format)
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

            self.logger.error(f"Unexpected error reading audio chunk: {e}")
            return None

    def _is_device_disconnected(self) -> bool:
        """Check if the device is still in the USB devices list."""
        try:
            usb_devices = self.list_devices()
            return self.device_stable_ref not in usb_devices
        except Exception as e:
            logger.debug(f"Error checking device status: {e}")
            return True  # Assume disconnected if we can't check


def _dtype_to_alsa_format_name(dtype: np.dtype, is_packed: bool = False) -> str:
    """
    Map numpy dtype to ALSA PCM format name.

    Args:
        dtype: Numpy dtype
        is_packed: Whether the format is packed (e.g., 24-bit audio)

    Returns:
        ALSA format name (e.g., 'S16_LE')

    Raises:
        MicrophoneConfigError: If dtype is unsupported
    """
    kind = dtype.kind
    size = dtype.itemsize
    byteorder = dtype.byteorder

    # Determine endianness: '<' = little, '>' = big, '=' = native, '|' = not applicable
    if byteorder == "=" or byteorder == "|":
        # Native byte order or not applicable (single byte)
        import sys

        byteorder = "<" if sys.byteorder == "little" else ">"

    # Signed integers
    if kind == "i":
        if size == 1:
            return "S8"
        elif size == 2:
            return "S16_LE" if byteorder == "<" else "S16_BE"
        elif size == 4:
            if is_packed:
                return "S24_LE" if byteorder == "<" else "S24_BE"
            return "S32_LE" if byteorder == "<" else "S32_BE"

    # Unsigned integers
    elif kind == "u":
        if size == 1:
            return "U8"
        elif size == 2:
            return "U16_LE" if byteorder == "<" else "U16_BE"
        elif size == 4:
            return "U32_LE" if byteorder == "<" else "U32_BE"

    # Floating point
    elif kind == "f":
        if size == 4:
            return "FLOAT_LE" if byteorder == "<" else "FLOAT_BE"
        elif size == 8:
            return "FLOAT64_LE" if byteorder == "<" else "FLOAT64_BE"

    raise MicrophoneConfigError(f"Unsupported numpy dtype for ALSA: {dtype}")


def _alsa_format_name_to_dtype(alsa_format: str) -> np.dtype:
    """
    Map ALSA PCM format name to numpy dtype.

    Args:
        alsa_format: ALSA format name (e.g., 'S16_LE')

    Returns:
        Numpy dtype object, or None if unsupported

    Raises:
        MicrophoneOpenError: If conversion is unsupported
    """
    # Direct mapping from ALSA format to numpy dtype string
    format_map = {
        "S8": "int8",
        "U8": "uint8",
        "S16_LE": "<i2",
        "S16_BE": ">i2",
        "U16_LE": "<u2",
        "U16_BE": ">u2",
        "S24_LE": "<i4",  # 24-bit packed in 32-bit container
        "S24_BE": ">i4",  # 24-bit packed in 32-bit container
        "S32_LE": "<i4",
        "S32_BE": ">i4",
        "U32_LE": "<u4",
        "U32_BE": ">u4",
        "FLOAT_LE": "<f4",
        "FLOAT_BE": ">f4",
        "FLOAT64_LE": "<f8",
        "FLOAT64_BE": ">f8",
    }

    dtype_str = format_map.get(alsa_format)
    if dtype_str is None:
        raise MicrophoneOpenError(f"Unsupported conversion for ALSA format to numpy dtype: {alsa_format}")

    return np.dtype(dtype_str)
