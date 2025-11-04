# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .microphone import Microphone
from .alsa_microphone import ALSAMicrophone
from .websocket_microphone import WebSocketMicrophone
from .errors import *
from .config import (
    # Sample rate constants
    RATE_8K,
    RATE_16K,
    RATE_22K,
    RATE_32K,
    RATE_44K,
    RATE_48K,
    RATE_96K,
    # Channel constants
    MONO,
    STEREO,
    # Format constants
    FORMAT_S16_LE,
    FORMAT_S24_LE,
    FORMAT_S32_LE,
    # Chunk size constants
    ULTRA_LOW_LATENCY_CHUNK,
    LOW_LATENCY_CHUNK,
    BALANCED_CHUNK,
    HIGH_THROUGHPUT_CHUNK,
    ULTRA_HIGH_THROUGHPUT_CHUNK,
)

__all__ = [
    "Microphone",
    "ALSAMicrophone",
    "WebSocketMicrophone",
    "MicrophoneError",
    "MicrophoneConfigError",
    "MicrophoneOpenError",
    "MicrophoneReadError",
    "MicrophoneDisconnectedError",
    # Sample rates
    "RATE_8K",
    "RATE_16K",
    "RATE_22K",
    "RATE_32K",
    "RATE_44K",
    "RATE_48K",
    "RATE_96K",
    # Channels
    "MONO",
    "STEREO",
    # Formats
    "FORMAT_S16_LE",
    "FORMAT_S24_LE",
    "FORMAT_S32_LE",
    # Chunk sizes
    "ULTRA_LOW_LATENCY_CHUNK",
    "LOW_LATENCY_CHUNK",
    "BALANCED_CHUNK",
    "HIGH_THROUGHPUT_CHUNK",
    "ULTRA_HIGH_THROUGHPUT_CHUNK",
]
