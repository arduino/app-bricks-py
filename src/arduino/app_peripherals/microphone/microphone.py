# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from urllib.parse import urlparse

from .base_microphone import BaseMicrophone
from .alsa_microphone import ALSAMicrophone
from .websocket_microphone import WebSocketMicrophone
from .config import RATE_16K, MONO, FORMAT_S16_LE, BALANCED_CHUNK
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
    """

    def __new__(
        cls,
        device: str | int = 0,
        sample_rate: int = RATE_16K,
        channels: int = MONO,
        format: str = FORMAT_S16_LE,
        chunk_size: int = BALANCED_CHUNK,
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
                    host (str, optional): WebSocket server host. Default: "0.0.0.0"
                    port (int, optional): WebSocket server port. Default: 8080
                    timeout (float, optional): Connection timeout in seconds. Default: 10.0
                    audio_format (str, optional): Expected audio format ("binary", "base64",
                        "json"). Default: "binary"

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
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # ALSA Microphone with index
            return ALSAMicrophone(
                device=device,
                sample_rate=sample_rate,
                channels=channels,
                format=format,
                chunk_size=chunk_size,
                **kwargs,
            )
        elif isinstance(device, str):
            parsed = urlparse(device)
            if parsed.scheme in ["ws", "wss"]:
                # WebSocket Microphone
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                return WebSocketMicrophone(
                    host=host,
                    port=port,
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    chunk_size=chunk_size,
                    **kwargs,
                )
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
        else:
            raise MicrophoneConfigError(f"Invalid device type: {type(device)}")
