# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from urllib.parse import urlparse

from .base_microphone import BaseMicrophone
from .alsa_microphone import ALSAMicrophone
from .websocket_microphone import WebSocketMicrophone
from .config import RATE_16K, CHANNELS_MONO, FORMAT_S16_LE, CHUNK_BALANCED
from .errors import MicrophoneConfigError


class Microphone:
    """
    Unified Microphone class that can be configured for different microphone types.

    This class serves as both a factory and a wrapper, automatically creating
    the appropriate microphone implementation based on the provided configuration.

    Supports:
        - ALSA Microphones (local microphones connected to the system via ALSA)
        - WebSocket Microphones (input audio streams via WebSocket client)

    Note: constructor arguments (except those in signature) must be provided in
    keyword format to forward them correctly to the specific microphone implementations.
    Refer to the documentation of each microphone type for available parameters.
    """

    USB_MIC_1 = ALSAMicrophone.USB_MIC_1
    """Shorthand for the first USB microphone available."""
    USB_MIC_2 = ALSAMicrophone.USB_MIC_2
    """Shorthand for the second USB microphone available."""

    def __new__(
        cls,
        device: str | int = USB_MIC_1,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = CHUNK_BALANCED,
        **kwargs,
    ) -> BaseMicrophone:
        """Create a microphone instance based on the device type.

        Args:
            device (Union[str, int]): Microphone device identifier. Supports:
                - int: ALSA microphone index (e.g., 0, 1) - uses USB microphones list
                - str: ALSA microphone index (e.g., "0", "1")
                - str: ALSA device name (e.g., "plughw:CARD=USB,DEV=0", "hw:0,0")
                - str: WebSocket URL for audio streams (e.g., "ws://0.0.0.0:8080")
            sample_rate (int, optional): Sample rate in Hz. Default: 16000
            channels (int, optional): Number of audio channels. Default: 1
            format (str, optional): Audio format. Default: "S16_LE"
            chunk_size (int, optional): Number of frames per chunk. Default: 1024
            **kwargs: Microphone-specific configuration parameters grouped by type:
                WebSocket Microphone Parameters:
                    port (int, optional): WebSocket server port. Default: 8080
                    timeout (float, optional): Connection timeout in seconds. Default: 3.0
                    audio_format (str): Expected audio format from clients ("binary", "base64",
                        "json") (default: "binary")
                    sample_rate (int): Sample rate in Hz (default: 16000)
                    channels (int): Number of audio channels (default: 1)
                    format (str): Audio format (default: "S16_LE")
                    chunk_size (int): Number of frames per chunk (default: 1024). This parameter
                        is advisory, it's sent to clients to suggest an optimal chunk size but
                        clients may ignore it.

        Returns:
            BaseMicrophone: Appropriate microphone implementation instance

        Raises:
            MicrophoneConfigError: If device type is not supported or parameters are invalid

        Examples:
            ALSA Microphone:

            ```python
            microphone = Microphone(0, sample_rate=16000, channels=1)  # First USB microphone
            microphone = Microphone(1)  # Second USB microphone
            microphone = Microphone("plughw:CARD=USB,DEV=0", format="S16_LE")
            microphone = Microphone("hw:0,0")
            ```

            WebSocket Microphone:

            ```python
            microphone = Microphone("ws://0.0.0.0:8080", audio_format="json")
            microphone = Microphone("ws://192.168.1.100:8080", sample_rate=48000)
            ```
        """
        if isinstance(device, str):
            parsed = urlparse(device)
            if parsed.scheme in ["ws", "wss"]:
                # WebSocket Microphone
                port = parsed.port if parsed.port is not None else 8080
                mic = WebSocketMicrophone(
                    port=port,
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    chunk_size=chunk_size,
                    **kwargs,
                )
                if parsed.hostname != "0.0.0.0":
                    mic.logger.warning(f"Ignoring bind addresses other than '0.0.0.0' ({parsed.hostname}).")
                return mic
            else:
                # ALSA Microphone
                return ALSAMicrophone(
                    device=device,
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    chunk_size=chunk_size,
                    **kwargs,
                )
        elif isinstance(device, int):
            # ALSA Microphone with index
            return ALSAMicrophone(
                device=device,
                sample_rate=sample_rate,
                channels=channels,
                format=format,
                chunk_size=chunk_size,
                **kwargs,
            )
        else:
            raise MicrophoneConfigError(f"Invalid device type: {type(device)}")
