# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from .base_speaker import BaseSpeaker, FormatPlain, FormatPacked


class Speaker:
    """
    Unified Speaker class that can be configured for different speaker types.

    This class serves as both a factory and a wrapper, automatically creating
    the appropriate speaker implementation based on the provided configuration.

    Supports:
        - ALSA Speakers (local speakers connected to the system via ALSA)

    Note: constructor arguments (except those in signature) must be provided in
    keyword format to forward them correctly to the specific speaker implementations.
    Refer to the documentation of each speaker type for available parameters.
    """

    """
    Constants for speaker configuration.

    Provides commonly used values for devices, sample rates, channels, and buffer sizes.
    Select appropriate values based on application requirements and hardware capabilities.
    """

    # =============================================================================
    # Predefined devices
    # =============================================================================
    USB_SPEAKER_1 = "usb:1"
    """Shorthand for the first USB speaker available."""
    USB_SPEAKER_2 = "usb:2"
    """Shorthand for the second USB speaker available."""
    JACK_SPEAKER_1 = "jack:1"
    """Shorthand for the first built-in (jack) speaker available."""

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
        device: str | int = 0,
        sample_rate: int = RATE_16K,
        channels: int = CHANNELS_MONO,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = BUFFER_SIZE_BALANCED,
        **kwargs,
    ) -> BaseSpeaker:
        """
        Create a speaker instance based on the device type.

        Args:
            device (Union[str, int]): Speaker device identifier. Supports:
                - int | str: Auto-select the n-th available plugged speaker
                    (e.g., 0, 1, "0", "1", ...), giving priority to USB speakers,
                    then jack speakers if supported by the platform. Speakers
                    already auto-selected by other instances are skipped, so concurrent
                    instances get distinct speakers as long as enough are plugged
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Speaker.USB_SPEAKER_x / Speaker.JACK_SPEAKER_x macros
                Default: 0.
            sample_rate (int): Sample rate in Hz. Default: 16000.
            channels (int): Number of audio channels. Default: 1.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
                Default: np.int16 - 16-bit signed platform-endian.
            buffer_size (int): Size of the audio buffer. Default: 1024.
            **kwargs: Speaker-specific configuration parameters grouped by type:

                ALSA Speaker Parameters:
                    shared (bool): Whether the speaker can be used by multiple applications
                        simultaneously. Default: True.
                    auto_reconnect (bool): Whether to automatically attempt to reconnect
                        if the speaker connection is lost. Default: True.

        Returns:
            BaseSpeaker: Appropriate speaker implementation instance

        Raises:
            SpeakerConfigError: If speaker type is not supported or parameters are invalid
            SpeakerOpenError: If no speaker is available at the requested index

        Examples:
            ALSA Speaker:

            ```python
            speaker = Speaker(sample_rate=16000, channels=1)  # First plugged speaker
            speaker = Speaker(Speaker.USB_SPEAKER_1, sample_rate=16000, channels=1)  # First USB speaker
            speaker = Speaker(Speaker.JACK_SPEAKER_1)  # First built-in (jack) speaker
            speaker = Speaker(1)  # Second plugged speaker
            speaker = Speaker("CARD=USB,DEV=0", format="S16_LE")
            speaker = Speaker("plughw:CARD=USB,DEV=0")
            speaker = Speaker("hw:0,0", buffer_size=2048)
            speaker = Speaker("/dev/snd/by-id/usb-My-Device-00")  # Using device file path
            speaker = Speaker("pipewire")  # To use default PipeWire speaker (if available)
            speaker = Speaker("pipewire:NODE=MyPipewireNode")  # Using PipeWire node name
            ```
        """
        if not isinstance(device, (str, int)):
            from .errors import SpeakerConfigError

            raise SpeakerConfigError(f"Invalid device type: {type(device)}")

        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # Select and claim the n-th available speaker so other auto-selected instances skip it
            from .utils import claim_nth_available_speaker, speaker_registry

            device = claim_nth_available_speaker(int(device))
            try:
                speaker = _create_speaker(device, sample_rate, channels, format, buffer_size, **kwargs)
            except BaseException:
                speaker_registry.release(device)
                raise
            speaker_registry.bind(device, speaker)
            return speaker

        return _create_speaker(device, sample_rate, channels, format, buffer_size, **kwargs)

    @staticmethod
    def play_pcm(
        pcm_audio: np.ndarray,
        sample_rate: int,
        channels: int,
        format: FormatPlain | FormatPacked,
        device: str | int = 0,
    ):
        """
        Play raw PCM audio data.

        Args:
            pcm_audio (np.ndarray): Raw PCM audio data in ALSA PCM format.
            sample_rate (int): Sample rate in Hz.
            channels (int): Number of audio channels.
            format (FormatPlain | FormatPacked): Audio format as one of:
                - Type classes: np.int16, np.float32, np.uint8
                - dtype objects: np.dtype('<i2'), np.dtype('>f4')
                - Strings: 'int16', '<i2', '>f4', 'float32'
                - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
            device (Union[str, int], optional): Speaker device identifier. Supports:
                - int | str: Auto-select the n-th available plugged speaker
                    (e.g., 0, 1, "0", "1", ...), giving priority to USB speakers,
                    then jack speakers if supported by the platform
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Speaker.USB_SPEAKER_x / Speaker.JACK_SPEAKER_x macros
                Default: 0.

        Raises:
            SpeakerOpenError: If speaker can't be opened.
            SpeakerWriteError: If speaker is not started.
            ValueError: If pcm_audio is empty or invalid.
            Exception: If the underlying implementation fails to write a frame.
        """
        with Speaker(device=device, sample_rate=sample_rate, channels=channels, format=format) as speaker:
            speaker.play_pcm(pcm_audio)

    @staticmethod
    def play_wav(wav_audio: np.ndarray, device: str | int = 0):
        """
        Play audio from WAV format data.
        Note: Only uncompressed PCM WAV files are supported.

        Args:
            wav_audio (np.ndarray): WAV format audio data (including header).
            device (Union[str, int], optional): Speaker device identifier. Supports:
                - int | str: Auto-select the n-th available plugged speaker
                    (e.g., 0, 1, "0", "1", ...), giving priority to USB speakers,
                    then jack speakers if supported by the platform
                - str: ALSA device name (e.g., "plughw:CARD=MyCard,DEV=0", "hw:0,0", "CARD=MyCard,DEV=0")
                - str: ALSA device file path (e.g., "/dev/snd/by-id/usb-My-Device-00")
                - str: Speaker.USB_SPEAKER_x / Speaker.JACK_SPEAKER_x macros
                Default: 0.

        Raises:
            SpeakerOpenError: If speaker can't be opened.
            SpeakerWriteError: If speaker is not started.
            ValueError: If wav_audio is empty or invalid.
            Exception: If the underlying implementation fails to write a frame.
        """
        import io
        import wave

        # Read WAV from numpy array
        wav_channels = 1
        wav_sampwidth = 2
        wav_framerate = 16000
        buffer = io.BytesIO(wav_audio.tobytes())
        try:
            with wave.open(buffer, "rb") as wav_file:
                wav_channels = wav_file.getnchannels()
                wav_sampwidth = wav_file.getsampwidth()
                wav_framerate = wav_file.getframerate()
        except wave.Error as e:
            raise ValueError(f"Invalid WAV data: {e}")

        # We only force integer types due to wave module limitations that
        # only support PCM formats, which are int-based.
        match wav_sampwidth:
            case 1:
                format = "u1"  # 8-bit PCM is unsigned
            case 2:
                format = "<i2"  # 16-bit PCM is signed
            case 3:
                format = ("<i4", True)  # 24-bit PCM packed in 32-bit container
            case 4:
                format = "<i4"  # 32-bit PCM is signed
            case _:
                raise ValueError(f"Unsupported WAV sample width: {wav_sampwidth} bytes")

        # Initialize speaker with WAV file parameters
        with Speaker(device=device, sample_rate=wav_framerate, channels=wav_channels, format=format) as speaker:
            speaker.play_wav(wav_audio)


def _create_speaker(
    device: str,
    sample_rate: int,
    channels: int,
    format: FormatPlain | FormatPacked,
    buffer_size: int,
    **kwargs,
) -> BaseSpeaker:
    """Create the speaker implementation matching the given device identifier."""
    from .alsa_speaker import ALSASpeaker  # Imported here to avoid circular dependency

    return ALSASpeaker(
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        format=format,
        buffer_size=buffer_size,
        **kwargs,
    )
