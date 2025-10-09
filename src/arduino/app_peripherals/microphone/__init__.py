# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import time
import alsaaudio
import numpy as np
import threading
import logging
import re
from arduino.app_utils import Logger

logger = Logger("Microphone")


class MicrophoneException(Exception):
    """Custom exception for Microphone errors."""

    pass


class MicrophoneDisconnectedException(MicrophoneException):
    """Raised when the microphone device is disconnected and max retries are exceeded."""

    pass


class Microphone:
    """Microphone class for capturing audio using ALSA PCM interface.

    Handles automatic reconnection on device disconnection.
    """

    USB_MIC_1 = "USB_MIC_1"
    USB_MIC_2 = "USB_MIC_2"

    # Mapping ALSA format string -> (PCM_FORMAT_*, numpy dtype)
    FORMAT_MAP = {
        "S8": ("PCM_FORMAT_S8", np.int8),
        "U8": ("PCM_FORMAT_U8", np.uint8),
        "S16_LE": ("PCM_FORMAT_S16_LE", np.int16),
        "S16_BE": ("PCM_FORMAT_S16_BE", ">i2"),
        "U16_LE": ("PCM_FORMAT_U16_LE", np.uint16),
        "U16_BE": ("PCM_FORMAT_U16_BE", ">u2"),
        "S24_LE": ("PCM_FORMAT_S24_LE", np.int32),  # 24bit packed in 32bit
        "S24_BE": ("PCM_FORMAT_S24_BE", ">i4"),
        "S24_3LE": ("PCM_FORMAT_S24_3LE", None),  # Not directly supported
        "S24_3BE": ("PCM_FORMAT_S24_3BE", None),  # Not directly supported
        "S32_LE": ("PCM_FORMAT_S32_LE", np.int32),
        "S32_BE": ("PCM_FORMAT_S32_BE", ">i4"),
        "U32_LE": ("PCM_FORMAT_U32_LE", np.uint32),
        "U32_BE": ("PCM_FORMAT_U32_BE", ">u4"),
        "FLOAT_LE": ("PCM_FORMAT_FLOAT_LE", np.float32),
        "FLOAT_BE": ("PCM_FORMAT_FLOAT_BE", ">f4"),
        "FLOAT64_LE": ("PCM_FORMAT_FLOAT64_LE", np.float64),
        "FLOAT64_BE": ("PCM_FORMAT_FLOAT64_BE", ">f8"),
        # Compressed/unsupported formats:
        "MU_LAW": ("PCM_FORMAT_MU_LAW", None),
        "A_LAW": ("PCM_FORMAT_A_LAW", None),
        "IMA_ADPCM": ("PCM_FORMAT_IMA_ADPCM", None),
        "MPEG": ("PCM_FORMAT_MPEG", None),
        "GSM": ("PCM_FORMAT_GSM", None),
    }

    def __init__(
        self,
        device: str = USB_MIC_1,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "S16_LE",
        periodsize: int = 1024,
        max_reconnect_attempts: int = 30,
        reconnect_delay: float = 2.0,
    ):
        """Initialize the Microphone object.

        Args:
            device (str): ALSA device name or USB_MIC_1/2 macro.
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format (default: "S16_LE").
            periodsize (int): Period size in frames (default: 1024).
            max_reconnect_attempts (int): Maximum attempts to reconnect on disconnection (default: 30).
            reconnect_delay (float): Delay in seconds between reconnection attempts (default: 2.0).

        Raises:
            MicrophoneException: If the microphone cannot be initialized or if the device is busy.
        """
        logger.info(
            "Init Microphone with device=%s, sample_rate=%d, channels=%d, format=%s, periodsize=%d",
            device,
            sample_rate,
            channels,
            format,
            periodsize,
        )
        self.device = self._resolve_device(device)
        self.sample_rate = sample_rate
        self.channels = channels
        if format not in self.FORMAT_MAP:
            raise MicrophoneException(f"Unsupported format: {format}")
        self.format = format
        self._alsa_format, self._dtype = self.FORMAT_MAP[format]
        if self._dtype is None:
            raise NotImplementedError(f"Format {self.format} is not supported for numpy conversion.")
        self.periodsize = periodsize
        self._pcm: alsaaudio.PCM = None
        self._pcm_lock = threading.Lock()
        self._native_rate = None
        self.is_recording = threading.Event()
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._mixer: alsaaudio.Mixer = self._load_mixer()

    def _resolve_device(self, device: str) -> str:
        """Resolve the ALSA device name, handling USB_MIC_1/2 macros and explicit device names.

        If device is None or USB_MIC_1/2, it will list available USB microphones and select the appropriate one.

        Args:
            device (str): The device name or USB_MIC_1/2 macro.

        Returns:
            str: The resolved ALSA device name.
        """
        logger.debug(f"Resolving device: {device}")
        if not device or device.startswith("USB_MIC_"):
            usb_devices = self._list_usb_microphones()
            if not usb_devices:
                logger.error("No USB microphones found for USB_MIC_1/2 macro.")
                raise MicrophoneException("No USB microphone found.")
            if device in (None, self.USB_MIC_1):
                logger.debug(f"Using USB_MIC_1: {usb_devices[0]}")
                return usb_devices[0]

            # Detect device via regex
            match = re.search(r"USB_MIC_(\d+)", device)
            if match:
                device_number = int(match.group(1))
                logger.debug(f"Detected USB_MIC_{device_number} from device string: {device}")
                if device_number < 2 or device_number > len(usb_devices):
                    logger.error(f"Invalid USB_MIC_{device_number} requested, only {len(usb_devices)} USB microphones found.")
                    raise MicrophoneException(f"Invalid USB_MIC_{device_number} requested, only {len(usb_devices)} USB microphones found.")

                logger.debug(f"Using USB_MIC_{device_number}: {usb_devices[device_number - 1]}")
                return usb_devices[device_number - 1]

        logger.debug(f"Using explicit device: {device}")
        return device

    def _load_mixer(self) -> alsaaudio.Mixer:
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
                        mx = alsaaudio.Mixer(mixer[0])
                        logger.debug(f"Loaded mixer: {mixer[0]} for mic {card_name}")
                        return mx
                    except alsaaudio.ALSAAudioError as e:
                        logger.debug(f"Failed to load mixer for mic {card_name}: {e}")

            # No suitable mixer found, return None
            return None
        except alsaaudio.ALSAAudioError as e:
            logger.warning(f"Error loading mixer {self.device}: {e}")
            return None

    def get_volume(self) -> int:
        """Get the current volume level of the microphone.

        Returns:
            int: Volume level (0-100). If no mixer is available, returns -1.

        Raises:
            MicrophoneException: If the mixer is not available or if volume cannot be retrieved.
        """
        if self._mixer is None:
            return -1  # No mixer available, return -1 to indicate no volume control
        try:
            volume = self._mixer.getvolume()[0]
            return volume
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error getting volume: {e}")
            raise MicrophoneException(f"Error getting volume: {e}")

    def set_volume(self, volume: int):
        """Set the volume level of the microphone.

        Args:
            volume (int): Volume level (0-100).

        Raises:
            MicrophoneException: If the mixer is not available or if volume cannot be set.
        """
        if self._mixer is None:
            return
        if not (0 <= volume <= 100):
            raise ValueError("Volume must be between 0 and 100.")
        try:
            self._mixer.setvolume(volume)
            logger.info(f"Volume set to {volume}%")
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error setting volume: {e}")
            raise MicrophoneException(f"Error setting volume: {e}")

    @staticmethod
    def _list_usb_microphones() -> list:
        """Return an ordered list of ALSA device names for available USB microphones (plughw only)."""
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
                    logger.debug(f"Error parsing card info for {card_name}: {e}")
        except Exception as e:
            logger.error(f"Error listing USB microphones: {e}")
        logger.info(f"USB microphones found: {usb_devices}")
        return usb_devices

    def _open_pcm(self):
        """Open the ALSA PCM device and set parameters, with fallback and error handling."""
        logger.debug(f"Opening PCM device: {self.device}")
        try:
            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                device=self.device,
            )
            try:
                self._pcm.setchannels(self.channels)
                self._pcm.setrate(self.sample_rate)
                self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                self._pcm.setperiodsize(self.periodsize)
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
                    plugdev = self.device.replace("hw", "plughw", 1) if self.device.startswith("hw") else f"plughw:{self.device}"
                    logger.debug(f"Trying fallback with plughw device: {plugdev}")
                    self._pcm = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                        device=plugdev,
                    )
                    self._pcm.setchannels(self.channels)
                    self._pcm.setrate(self.sample_rate)
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    self._pcm.setperiodsize(self.periodsize)
                    self.device = plugdev
                    self._native_rate = self.sample_rate
                    logger.debug(f"PCM opened with plughw fallback: {plugdev}")
                else:
                    logger.error(f"plughw fallback failed, using native device params for {self.device}")
                    self._pcm = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                        device=self.device,
                    )
                    self._pcm.setchannels(self.channels)
                    self._native_rate = self._pcm.rate()
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    self._pcm.setperiodsize(self.periodsize)
                    logger.debug("PCM opened with native params: %s, %dHz", self.device, self._native_rate)
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"ALSAAudioError opening PCM device {self.device}: {e}")
            if "Device or resource busy" in str(e):
                raise MicrophoneException("Selected microphone is busy. Close other audio applications and try again. (%s)" % self.device)
            else:
                raise MicrophoneException(f"ALSA error opening microphone: {e}")
        except Exception as e:
            logger.error(f"Unexpected error opening PCM device {self.device}: {e}")
            raise MicrophoneException(f"Unexpected error opening microphone: {e}")

    def start(self):
        """Start the microphone stream by opening the PCM device."""
        if self.is_recording.is_set():
            raise RuntimeError("Microphone is already recording, cannot start again.")
        self.connect()

    def connect(self):
        """Try to connect the microphone device."""
        with self._pcm_lock:
            self._open_pcm()
            self.is_recording.set()
            logger.info("Microphone connected successfully.")

    def stream(self):
        """Yield audio chunks from the microphone. Each chunk has periodsize samples.

        - Handles automatic reconnection if the device is unplugged and replugged.
        - Only one main loop, no nested loops.
        - Thread safe and clean state management.
        - When max reconnect attempts are reached, the generator returns (StopIteration for the caller).
        - All PCM operations are protected by lock.

        Yields:
            np.ndarray: Audio data as a numpy array of the correct dtype, depending on the format specified.
        """
        prev_time = None
        debug_mode = logger.isEnabledFor(logging.DEBUG)
        reconnect_attempts = 0
        while self.is_recording.is_set():
            if self._pcm is None:
                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached. Giving up.")
                    self.stop()
                    logger.warning("Microphone stream stopped, cleaning up resources")
                    logger.debug(f"stream stopped - is_recording: {self.is_recording.is_set()}")
                    # Instead of returning silently (which makes the caller re-invoke the loop and
                    # produce repeated "Microphone stream stopped" logs), raise a specific exception
                    # to signal permanent failure. The brick/worker can catch or let it propagate so
                    # the AppController handles the termination.
                    raise MicrophoneDisconnectedException("Max reconnect attempts reached while trying to reconnect microphone")
                logger.info(f"Waiting for microphone to be reconnected... (attempt {reconnect_attempts + 1})")
                time.sleep(self.reconnect_delay)
                try:
                    self.connect()
                    logger.info("Microphone reconnected successfully.")
                    reconnect_attempts = 0
                except MicrophoneException:
                    reconnect_attempts += 1
                continue
            try:
                if debug_mode:
                    t0 = time.perf_counter()
                with self._pcm_lock:
                    l, data = self._pcm.read()
                if debug_mode:
                    t1 = time.perf_counter()
                if l > 0:
                    try:
                        arr = np.frombuffer(data, dtype=self._dtype)
                    except Exception as e:
                        logger.error(f"Error converting PCM data to numpy array: {e}")
                        continue
                    if debug_mode:
                        elapsed = t1 - (prev_time if prev_time is not None else t0)
                        prev_time = t1
                        if elapsed > 0:
                            effective_rate = arr.size / elapsed
                            logger.debug(
                                "Chunk: %d samples, elapsed=%.4fs, effective_rate=%.1fHz, requested=%d",
                                arr.size,
                                elapsed,
                                effective_rate,
                                self.sample_rate,
                            )
                    yield arr
                    reconnect_attempts = 0  # reset on success
                else:
                    logger.debug("No audio data read from PCM device.")

            except alsaaudio.ALSAAudioError as e:
                if self.device not in self._list_usb_microphones():
                    logger.error(f"Microphone disconnected: {e}")
                    with self._pcm_lock:
                        if self._pcm is not None:
                            try:
                                self._pcm.close()
                            except Exception:
                                pass
                    self._pcm = None
                    continue
                else:
                    logger.error(f"Unexpected ALSA error: {e}")
                    self.stop()
                    logger.warning("Microphone stream stopped, cleaning up resources")
                    logger.debug(f"stream stopped - is_recording: {self.is_recording.is_set()}")
                    return
            except Exception as e:
                logger.error(f"Unexpected error in microphone stream: {e}")
                continue
        # Log stream stop only if recording is stopped externally
        logger.warning("Microphone stream stopped, cleaning up resources")
        logger.debug(f"stream stopped - is_recording: {self.is_recording.is_set()}")

    @staticmethod
    def list_usb_devices() -> list:
        """Return a list of available USB microphone ALSA device names (plughw only).

        Returns:
            list: List of USB microphone device names.
        """
        try:
            return Microphone._list_usb_microphones()
        except Exception as e:
            logger.error(f"Error retrieving USB devices: {e}")
            return []

    def stop(self):
        """Close the PCM device if open."""
        if not hasattr(self, "is_recording") or not self.is_recording.is_set():
            logger.warning("Microphone is not recording, nothing to stop.")
            return
        logger.warning("[stop] Closing PCM device.")
        try:
            with self._pcm_lock:
                if self._pcm is not None:
                    self._pcm.close()
                    self._pcm = None
        finally:
            self.is_recording.clear()
            logger.debug(f"Microphone stream Event is cleared: {self.is_recording}")
            logger.info(f"[stop] PCM device closed: {self.device}")

    def __del__(self):
        """Ensure PCM device is closed when the object is destroyed."""
        try:
            logger.warning("[__del__] Microphone __del__: cleaning up resources.")
            self.stop()
        except Exception as e:
            logger.warning(f"Microphone __del__: stop() failed or already closed: {e}")

    def __enter__(self):
        """Context manager entry method to start the microphone stream."""
        logger.debug("Entering Microphone context manager.")
        if self.is_recording.is_set():
            raise RuntimeError("Microphone is already recording, cannot enter context again.")
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Context manager exit method to stop the microphone stream."""
        logger.debug("Exiting Microphone context manager.")
        try:
            logger.warning("[__exit__] Microphone __del__: cleaning up resources.")
            self.stop()
        except Exception:
            logger.warning("Microphone is not recording, cannot exit context.")
        return False  # Do not suppress exceptions
