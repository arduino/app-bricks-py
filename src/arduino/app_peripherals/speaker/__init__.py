# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import alsaaudio
import numpy as np
import threading
import queue
import re
from arduino.app_utils import Logger

logger = Logger("Speaker")


class SpeakerException(Exception):
    """Custom exception for Speaker errors."""

    pass


class Speaker:
    """Speaker class for reproducing audio using ALSA PCM interface."""

    USB_SPEAKER_1 = "USB_SPEAKER_1"
    USB_SPEAKER_2 = "USB_SPEAKER_2"

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
        device: str = USB_SPEAKER_1,  # To change
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "S16_LE",
    ):
        """Initialize the Speaker object.

        Args:
            device (str): ALSA device name or USB_SPEAKER_1/2 macro.
            sample_rate (int): Sample rate in Hz (default: 16000).
            channels (int): Number of audio channels (default: 1).
            format (str): Audio format (default: "S16_LE").

        Raises:
            SpeakerException: If the speaker cannot be initialized or if the device is busy.
        """
        logger.info(
            "Init Speaker with device=%s, sample_rate=%d, channels=%d, format=%s",
            device,
            sample_rate,
            channels,
            format,
        )
        self.sample_rate = sample_rate
        self.channels = channels
        if format not in self.FORMAT_MAP:
            raise SpeakerException(f"Unsupported format: {format}")
        self.format = format
        self._alsa_format, self._dtype = self.FORMAT_MAP[format]
        if self._dtype is None:
            raise NotImplementedError(f"Format {self.format} is not supported for numpy conversion.")
        self._pcm: alsaaudio.PCM = None
        self._pcm_lock = threading.Lock()
        self._native_rate = None
        self._is_reproducing = threading.Event()
        self._playing_queue: bytes = queue.Queue(maxsize=100)  # Queue for audio data to play with limited capacity
        self.device = self._resolve_device(device)
        self._mixer: alsaaudio.Mixer = self._load_mixer()

    def _resolve_device(self, device: str) -> str:
        """Resolve the ALSA device name, handling USB_SPEAKER_1/2 macros and explicit device names.

        If device is None or USB_SPEAKER_1/2, it will list available USB speakers and select the appropriate one.

        Args:
            device (str): The device name or USB_SPEAKER_(1,2,...N) macro.

        Returns:
            str: The resolved ALSA device name.
        """
        logger.debug(f"Resolving device: {device}")
        if not device or device.startswith("USB_SPEAKER_"):
            usb_devices = self._list_usb_speakers()
            logger.info(f"Available USB speakers: {usb_devices}")
            if not usb_devices:
                logger.error("No USB speakers found for USB_SPEAKER_1/2 macro.")
                raise SpeakerException("No USB speaker found.")
            if device in (None, self.USB_SPEAKER_1):
                logger.debug(f"Using USB_SPEAKER_1: {usb_devices[0]}")
                return usb_devices[0]

            # Detect device via regex
            match = re.search(r"USB_SPEAKER_(\d+)", device)
            if match:
                device_number = int(match.group(1))
                logger.info(f"Detected USB_SPEAKER_{device_number} from device string: {device}")
                if device_number < 2 or device_number > len(usb_devices):
                    logger.error(f"Invalid USB_SPEAKER_{device_number} requested, only {len(usb_devices)} USB speakers found.")
                    raise SpeakerException(f"Invalid USB_SPEAKER_{device_number} requested, only {len(usb_devices)} USB speakers found.")

                logger.debug(f"Using USB_SPEAKER_{device_number}: {usb_devices[device_number - 1]}")
                return usb_devices[device_number - 1]

        logger.info(f"Using explicit device: {device}")
        return device

    @staticmethod
    def _list_usb_speakers() -> list:
        """Return an ordered list of ALSA device names for available USB speaker (plughw only)."""
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
            logger.error(f"Error listing USB speakers: {e}")
        logger.info(f"USB speakers found: {usb_devices}")
        return usb_devices

    @staticmethod
    def list_usb_devices() -> list:
        """Return a list of available USB speaker ALSA device names (plughw only).

        Returns:
            list: List of USB speaker device names.
        """
        try:
            return Speaker._list_usb_speakers()
        except Exception as e:
            logger.error(f"Error retrieving USB devices: {e}")
            return []

    def _open_pcm(self):
        """Open the ALSA PCM device and set parameters, with fallback and error handling."""
        logger.debug(f"Opening PCM device: {self.device}")
        try:
            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                device=self.device,
            )
            try:
                self._pcm.setchannels(self.channels)
                self._pcm.setrate(self.sample_rate)
                self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
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
                        type=alsaaudio.PCM_PLAYBACK,
                        mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                        device=plugdev,
                    )
                    self._pcm.setchannels(self.channels)
                    self._pcm.setrate(self.sample_rate)
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    self.device = plugdev
                    self._native_rate = self.sample_rate
                    logger.debug(f"PCM opened with plughw fallback: {plugdev}")
                else:
                    logger.error(f"plughw fallback failed, using native device params for {self.device}")
                    self._pcm = alsaaudio.PCM(
                        type=alsaaudio.PCM_PLAYBACK,
                        mode=alsaaudio.PCM_NORMAL,  # Hardcoded to blocking mode
                        device=self.device,
                    )
                    self._pcm.setchannels(self.channels)
                    self._native_rate = self._pcm.rate()
                    self._pcm.setformat(getattr(alsaaudio, self._alsa_format))
                    logger.debug("PCM opened with native params: %s, %dHz", self.device, self._native_rate)
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"ALSAAudioError opening PCM device {self.device}: {e}")
            if "Device or resource busy" in str(e):
                raise SpeakerException("Selected speaker is busy. Close other audio applications and try again. (%s)" % self.device)
            else:
                raise SpeakerException(f"ALSA error opening speaker: {e}")
        except Exception as e:
            logger.error(f"Unexpected error opening PCM device {self.device}: {e}")
            raise SpeakerException(f"Unexpected error opening spaker: {e}")

    def _load_mixer(self) -> alsaaudio.Mixer:
        try:
            cards = alsaaudio.cards()
            card_indexes = alsaaudio.card_indexes()
            for card_name, card_index in zip(cards, card_indexes):
                logger.debug(f"Checking Card {card_name} (index {card_index}, device {self.device})")
                if f"CARD={card_name}," in self.device:
                    try:
                        mixer = alsaaudio.mixers(cardindex=card_index)
                        if len(mixer) == 0:
                            logger.warning(f"No mixers found for card {card_name}.")
                            continue
                        mx = alsaaudio.Mixer(mixer[0])
                        logger.debug(f"Loaded mixer: {mixer[0]} for card {card_name}")
                        return mx
                    except alsaaudio.ALSAAudioError as e:
                        logger.debug(f"Failed to load mixer for card {card_name}: {e}")

            # No suitable mixer found, return None
            return None
        except alsaaudio.ALSAAudioError as e:
            logger.warning(f"Error loading mixer {self.device}: {e}")
            return None

    def get_volume(self) -> int:
        """Get the current volume level of the speaker.

        Returns:
            int: Volume level (0-100). If no mixer is available, returns -1.

        Raises:
            SpeakerException: If the mixer is not available or if volume cannot be retrieved.
        """
        if self._mixer is None:
            return -1  # No mixer available, return -1 to indicate no volume control
        try:
            volume = self._mixer.getvolume()[0]
            return volume
        except alsaaudio.ALSAAudioError as e:
            logger.error(f"Error getting volume: {e}")
            raise SpeakerException(f"Error getting volume: {e}")

    def set_volume(self, volume: int):
        """Set the volume level of the speaker.

        Args:
            volume (int): Volume level (0-100).

        Raises:
            SpeakerException: If the mixer is not available or if volume cannot be set.
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
            raise SpeakerException(f"Error setting volume: {e}")

    def _clear_queue(self):
        """Clear the playback queue."""
        with self._pcm_lock:
            while not self._playing_queue.empty():
                try:
                    self._playing_queue.get_nowait()
                except queue.Empty:
                    break
        logger.debug("Playback queue cleared.")

    def start(self):
        """Start the spaker stream by opening the PCM device."""
        if self._is_reproducing.is_set():
            raise RuntimeError("Spaker is already reproducing audio, cannot start again.")
        self._clear_queue()
        self._open_pcm()
        self._is_reproducing.set()
        logger.debug(f"Spaker stream event is set: {self._is_reproducing}, starting playback thread.")
        with self._pcm_lock:
            self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._playback_thread.start()

    def stop(self):
        """Close the PCM device if open."""
        if not self._is_reproducing.is_set():
            logger.warning("Spaker is not recording, nothing to stop.")
            return

        # Stop the playback thread
        logger.debug("Closing playback thread.")
        self._is_reproducing.clear()
        with self._pcm_lock:
            self._playback_thread.join(timeout=5)
            self._playback_thread = None

        logger.debug("Closing PCM device.")
        try:
            with self._pcm_lock:
                if self._pcm is not None:
                    self._pcm.close()
                    self._pcm = None

            self._clear_queue()
        finally:
            self._is_reproducing.clear()
            logger.debug(f"Speaker stream event is cleared: {self._is_reproducing}")
            logger.info(f"PCM device closed: {self.device}")

    def __del__(self):
        """Ensure PCM device is closed when the object is destroyed."""
        try:
            self.stop()
        except Exception as e:
            logger.warning(f"Spaker __del__: stop() failed or already closed: {e}")

    def __enter__(self):
        """Context manager entry method to start the speaker stream."""
        logger.debug("Entering Spaker context manager.")
        if self._is_reproducing.is_set():
            raise RuntimeError("Spaker is already reproducing, cannot enter context again.")
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Context manager exit method to stop the spaker stream."""
        logger.debug("Exiting Spaker context manager.")
        try:
            self.stop()
        except Exception:
            logger.warning("Spaker is not reproducing, cannot exit context.")
        return False  # Do not suppress exceptions

    def _playback_loop(self):
        """Thread function to handle audio playback."""
        logger.debug("Starting playback thread.")
        while self._is_reproducing.is_set():
            try:
                data = self._playing_queue.get(timeout=1)  # Wait for audio data
                if data is None:
                    continue  # Skip if no data is available

                with self._pcm_lock:
                    if self._pcm is not None:
                        self._pcm.write(data)

                self._playing_queue.task_done()  # Mark the task as done
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback thread error: {e}")
                break
        logger.debug("Playback thread stopped.")

    def play(self, data: bytes | np.ndarray, block_on_queue: bool = False):
        """Play audio data through the speaker.

        Args:
            data (bytes|np.ndarray): Audio data to play as bytes or np.ndarray.
            block_on_queue (bool): If True, block until the queue has space for the data.

        Raises:
            SpeakerException: If the speaker is not started or if playback fails.
        """
        if not self._is_reproducing.is_set():
            raise SpeakerException("Spaker is not started, cannot play audio.")

        try:
            if isinstance(data, bytes):
                self._playing_queue.put(data, block=block_on_queue)
            elif isinstance(data, np.ndarray):
                # Convert numpy array to bytes
                data_bytes = data.astype(self._dtype).tobytes()
                self._playing_queue.put(data_bytes, block=block_on_queue)
            else:
                raise TypeError("Audio data must be bytes or numpy array.")
        except queue.Full:
            # logger.warning("Playback queue is full, dropping oldest data.")
            self._playing_queue.get_nowait()
