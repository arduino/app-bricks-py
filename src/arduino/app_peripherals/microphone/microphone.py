# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from .base_microphone import BaseMicrophone, FormatPlain, FormatPacked


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

    """
    Constants for microphone configuration.

    Provides commonly used values for devices, sample rates, channels, and buffer sizes.
    Select appropriate values based on application requirements and hardware capabilities.
    """

    # =============================================================================
    # Predefined devices
    # =============================================================================
    USB_MIC_1 = "usb:1"
    """Shorthand for the first USB microphone available."""
    USB_MIC_2 = "usb:2"
    """Shorthand for the second USB microphone available."""

    # =============================================================================
    # Sample Rate Constants
    # =============================================================================

    RATE_8K = 8000
    """8 kHz - Telephony bandwidth, VoIP applications"""

    RATE_16K = 16000
    """16 kHz - Common for voice processing, speech recognition, keyword spotting"""

    RATE_32K = 32000
    """32 kHz - Higher quality voice, some broadcast applications"""

    RATE_44K = 44100
    """44.1 kHz - CD quality audio standard"""

    RATE_48K = 48000
    """48 kHz - Professional audio, broadcast standard, high-quality recording"""

    # =============================================================================
    # Channel Constants
    # =============================================================================

    CHANNELS_MONO = 1
    """Mono - Single channel audio"""

    CHANNELS_STEREO = 2
    """Stereo - Two channel audio (left and right)"""

    # =============================================================================
    # Buffer Size Constants
    # =============================================================================

    BUFFER_SIZE_REALTIME = 256
    """
    Real-time/low-latency: ~16ms @ 16kHz, ~5ms @ 48kHz

    Use for: Live audio effects, real-time playback, low-latency voice chat
    Trade-off: Low latency but high CPU usage
    """

    BUFFER_SIZE_BALANCED = 1024
    """
    Balanced (default): ~64ms @ 16kHz, ~21ms @ 48kHz

    Use for: Voice commands, keyword spotting, general audio processing
    Trade-off: Good balance between latency and efficiency
    """

    BUFFER_SIZE_SAFE = 4096
    """
    Transcription/recording: ~256ms @ 16kHz, ~85ms @ 48kHz

    Use for: Speech recognition, transcription, recording, non-real-time processing
    Trade-off: High latency but low CPU usage
    """

    def __new__(
        cls,
        device: str | int = USB_MIC_1,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = BUFFER_SIZE_BALANCED,
        **kwargs,
    ) -> BaseMicrophone:
        """
        Create a microphone instance based on the device type.

        Args:
            device (Union[str, int]): Microphone device identifier. Supports:
                - int | str: ALSA device ordinal index (e.g., 0, 1, "0", "1", ...)
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Microphone.USB_MIC_x macros
                - str: WebSocket URL for audio streams (e.g., "ws://0.0.0.0:8080")
                Default: USB_MIC_1 - First USB microphone.
            sample_rate (int): Sample rate in Hz. Default: 16000
            channels (int): Number of audio channels. Default: 1
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
                Default: np.int16 - 16-bit signed platform-endian.
            buffer_size (int): Size of the audio buffer. Default: 1024.
            **kwargs: Microphone-specific configuration parameters grouped by type:
                ALSA Microphone Parameters:
                    shared (bool): Whether the microphone can be used by multiple applications
                        simultaneously. Default: True.
                    auto_reconnect (bool): Whether to automatically attempt to reconnect
                        if the microphone connection is lost. Default: True.
                WebSocket Microphone Parameters:
                    port (int): WebSocket server port. Default: 8080
                    timeout (float): Connection timeout in seconds. Default: 3.0
                    certs_dir_path (str): Path to the directory containing TLS certificates.
                    use_tls (bool): Enable TLS for secure connections. If True, 'encrypt' will
                        be ignored. Use this for transport-level security with clients that can
                        accept self-signed certificates or when supplying your own certificates.
                    secret (str): Secret key for authentication/encryption (empty = security disabled)
                    encrypt (bool): Enable encryption (only effective if secret is provided)
                    auto_reconnect (bool): Whether to automatically attempt to reconnect
                        if the microphone connection is lost. Default: True.

        Returns:
            BaseMicrophone: Appropriate microphone implementation instance

        Raises:
            MicrophoneConfigError: If device type is not supported or parameters are invalid

        Examples:
            ALSA Microphone:

            ```python
            microphone = Microphone(sample_rate=16000, channels=1)  # First USB microphone
            microphone = Microphone(USB_MIC_1, sample_rate=16000, channels=1)  # Equivalent to above
            microphone = Microphone(1)  # Second microphone
            microphone = Microphone("CARD=MyCard,DEV=0", format="S16_LE")
            microphone = Microphone("plughw:CARD=MyCard,DEV=0")
            microphone = Microphone("hw:0,0")
            microphone = Microphone("/dev/snd/by-id/usb-My-Device-00")  # Using device file path
            ```

            WebSocket Microphone:

            ```python
            microphone = Microphone("ws://0.0.0.0:8080", audio_format="json")
            microphone = Microphone("ws://192.168.1.100:8080", sample_rate=48000)
            ```
        """
        if isinstance(device, str):
            from urllib.parse import urlparse

            parsed = urlparse(device)
            if parsed.scheme in ["ws", "wss"]:
                from .websocket_microphone import WebSocketMicrophone  # Imported here to avoid circular dependency

                # WebSocket Microphone
                port = parsed.port if parsed.port is not None else 8080
                mic = WebSocketMicrophone(
                    port=port,
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    buffer_size=buffer_size,
                    **kwargs,
                )
                if parsed.hostname != "0.0.0.0":
                    mic.logger.warning(f"Ignoring bind addresses other than '0.0.0.0' ({parsed.hostname}).")
                return mic
            else:
                from .alsa_microphone import ALSAMicrophone  # Imported here to avoid circular dependency

                # ALSA Microphone
                return ALSAMicrophone(
                    device=device,
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    buffer_size=buffer_size,
                    **kwargs,
                )
        elif isinstance(device, int):
            from .alsa_microphone import ALSAMicrophone  # Imported here to avoid circular dependency

            # ALSA Microphone with index
            return ALSAMicrophone(
                device=device,
                sample_rate=sample_rate,
                channels=channels,
                format=format,
                buffer_size=buffer_size,
                **kwargs,
            )
        else:
            from .errors import MicrophoneConfigError

            raise MicrophoneConfigError(f"Invalid device type: {type(device)}")

    @staticmethod
    def record_pcm(duration: float, sample_rate: int, channels: int, format: FormatPlain | FormatPacked, device: str | int = USB_MIC_1) -> np.ndarray:
        """
        Record audio for a specified duration and return as raw PCM format.

        Args:
            duration (float): Recording duration in seconds.
            sample_rate (int): Sample rate in Hz.
            channels (int): Number of audio channels.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
            device (Union[str, int]): Microphone device identifier. Supports:
                - int | str: ALSA device ordinal index (e.g., 0, 1, "0", "1", ...)
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Microphone.USB_MIC_x macros
                - str: WebSocket URL for audio streams (e.g., "ws://0.0.0.0:8080")
                Default: USB_MIC_1 - First USB microphone.

        Returns:
            np.ndarray: Raw audio data in raw PCM format.

        Raises:
            MicrophoneOpenError: If microphone can't be opened.
            MicrophoneReadError: If no audio is available after multiple attempts.
            ValueError: If duration is not > 0.
            Exception: If the underlying implementation fails to read a frame.
        """
        with Microphone(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            format=format,
        ) as mic:
            return mic.record_pcm(duration=duration)

    @staticmethod
    def record_wav(duration: float, sample_rate: int, channels: int, format: FormatPlain | FormatPacked, device: str | int = USB_MIC_1) -> np.ndarray:
        """
        Record audio for a specified duration and return as WAV format.
        Note: Only uncompressed PCM WAV recordings are supported.

        Args:
            duration (float): Recording duration in seconds.
            sample_rate (int): Sample rate in Hz.
            channels (int): Number of audio channels.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
            device (Union[str, int], optional): Microphone device identifier. Supports:
                - int | str: ALSA device ordinal index (e.g., 0, 1, "0", "1", ...)
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Microphone.USB_MIC_x macros
                - str: WebSocket URL for audio streams (e.g., "ws://0.0.0.0:8080")
                Default: USB_MIC_1 - First USB microphone.

        Returns:
            np.ndarray: Raw audio data in WAV format as numpy array.

        Raises:
            MicrophoneOpenError: If microphone can't be opened.
            MicrophoneReadError: If no audio is available after multiple attempts.
            ValueError: If duration is not > 0.
            Exception: If the underlying implementation fails to read a frame.
        """
        with Microphone(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            format=format,
        ) as mic:
            return mic.record_wav(duration=duration)
