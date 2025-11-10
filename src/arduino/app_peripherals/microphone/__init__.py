# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .microphone import Microphone
from .base_microphone import BaseMicrophone
from .alsa_microphone import ALSAMicrophone
from .websocket_microphone import WebSocketMicrophone
from .errors import *
from .config import (
    # Sample rate constants
    RATE_8K,
    RATE_16K,
    RATE_32K,
    RATE_44K,
    RATE_48K,
    # Channel constants
    CHANNELS_MONO,
    CHANNELS_STEREO,
    # Format constants
    FORMAT_S16_LE,
    FORMAT_S24_LE,
    FORMAT_S32_LE,
    # Chunk size constants
    CHUNK_LOW_LATENCY,
    CHUNK_BALANCED,
    CHUNK_HIGH_THROUGHPUT,
)

__all__ = [
    "Microphone",
    "BaseMicrophone",
    "ALSAMicrophone",
    "WebSocketMicrophone",
    "MicrophoneError",
    "MicrophoneConfigError",
    "MicrophoneOpenError",
    "MicrophoneReadError",
    # Sample rates
    "RATE_8K",
    "RATE_16K",
    "RATE_32K",
    "RATE_44K",
    "RATE_48K",
    # Channels
    "CHANNELS_MONO",
    "CHANNELS_STEREO",
    # Formats
    "FORMAT_S16_LE",
    "FORMAT_S24_LE",
    "FORMAT_S32_LE",
    # Chunk sizes
    "CHUNK_LOW_LATENCY",
    "CHUNK_BALANCED",
    "CHUNK_HIGH_THROUGHPUT",
]
