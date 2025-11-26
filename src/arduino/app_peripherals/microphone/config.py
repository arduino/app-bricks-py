# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""
Constants for microphone configuration.

Provides commonly used values for sample rates, channels, formats, and chunk sizes.
Select appropriate values based on hardware capabilities and application requirements.
"""

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
# Format Constants in ALSA PCM Notation
# =============================================================================

FORMAT_S8 = "S8"
"""Signed 8-bit"""

FORMAT_U8 = "U8"
"""Unsigned 8-bit"""

FORMAT_S16_LE = "S16_LE"
"""Signed 16-bit little-endian"""

FORMAT_S16_BE = "S16_BE"
"""Signed 16-bit big-endian"""

FORMAT_U16_LE = "U16_LE"
"""Unsigned 16-bit little-endian"""

FORMAT_U16_BE = "U16_BE"
"""Unsigned 16-bit big-endian"""

FORMAT_S24_LE = "S24_LE"
"""Signed 24-bit little-endian.
Note: this needs to be packed in a 32-bit container with the LSB padded."""

FORMAT_S24_BE = "S24_BE"
"""Signed 24-bit big-endian.
Note: this needs to be packed in a 32-bit container with the LSB padded."""

FORMAT_S32_LE = "S32_LE"
"""Signed 32-bit little-endian"""

FORMAT_S32_BE = "S32_BE"
"""Signed 32-bit big-endian"""

FORMAT_U32_LE = "U32_LE"
"""Unsigned 32-bit little-endian"""

FORMAT_U32_BE = "U32_BE"
"""Unsigned 32-bit big-endian"""

FORMAT_FLOAT_LE = "FLOAT_LE"
"""32-bit float little-endian"""

FORMAT_FLOAT_BE = "FLOAT_BE"
"""32-bit float big-endian"""

FORMAT_FLOAT64_LE = "FLOAT64_LE"
"""64-bit float little-endian
Note: this needs to be normalized to [-1.0, 1.0]."""

FORMAT_FLOAT64_BE = "FLOAT64_BE"
"""64-bit float big-endian.
Note: this needs to be normalized to [-1.0, 1.0]."""


# =============================================================================
# Chunk Size Constants
# =============================================================================
# These are hardware-independent and control latency vs. CPU usage trade-offs.
# Choose based on your application's real-time requirements.

CHUNK_LOW_LATENCY = 128
"""
Real-time/low-latency: ~8ms @ 16kHz, ~3ms @ 48kHz

Use for: Live audio effects, real-time playback, low-latency voice chat
Trade-off: Low latency but high CPU usage
"""

CHUNK_BALANCED = 512
"""
Balanced (default): ~32ms @ 16kHz, ~10ms @ 48kHz

Use for: Voice commands, keyword spotting, general audio processing
Trade-off: Good balance between latency and efficiency
"""

CHUNK_HIGH_THROUGHPUT = 2048
"""
Analysis/recording: ~128ms @ 16kHz, ~43ms @ 48kHz

Use for: Speech recognition, transcription, recording, non-real-time processing
Trade-off: High latency but low CPU usage
"""
