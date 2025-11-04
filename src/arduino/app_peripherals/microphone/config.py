# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Constants for microphone configuration.

Provides commonly used values for sample rates, channels, formats, and chunk sizes.
Always verify that your hardware supports the chosen parameters - while plughw can
convert, it won't improve quality beyond hardware limits.

Usage:
    from arduino.app_peripherals.microphone import (
        Microphone, RATE_16K, MONO, FORMAT_S16_LE, BALANCED_CHUNK
    )

    # Configure with your hardware's actual specs
    mic = Microphone(0,
                     sample_rate=RATE_16K,
                     channels=MONO,
                     format=FORMAT_S16_LE,
                     chunk_size=BALANCED_CHUNK)
"""

# ============================================================================
# Sample Rate Constants
# ============================================================================

RATE_8K = 8000
"""8 kHz - Telephony bandwidth, VoIP applications"""

RATE_16K = 16000
"""16 kHz - Common for voice processing, speech recognition, keyword spotting"""

RATE_22K = 22050
"""22.05 kHz - Lower quality music/audio, legacy multimedia"""

RATE_32K = 32000
"""32 kHz - Higher quality voice, some broadcast applications"""

RATE_44K = 44100
"""44.1 kHz - CD quality audio standard"""

RATE_48K = 48000
"""48 kHz - Professional audio, broadcast standard, high-quality recording"""

RATE_96K = 96000
"""96 kHz - High-resolution audio (requires professional hardware)"""


# ============================================================================
# Channel Constants
# ============================================================================

MONO = 1
"""Mono - Single channel audio"""

STEREO = 2
"""Stereo - Two channel audio (left and right)"""


# ============================================================================
# Format Constants
# ============================================================================

FORMAT_S16_LE = "S16_LE"
"""Signed 16-bit little-endian (standard for most audio hardware)"""

FORMAT_S24_LE = "S24_LE"
"""Signed 24-bit little-endian (professional audio interfaces)"""

FORMAT_S32_LE = "S32_LE"
"""Signed 32-bit little-endian (high-end professional audio)"""


# ============================================================================
# Chunk Size Constants
# ============================================================================
# These are hardware-independent and control latency vs. CPU usage trade-offs.
# Choose based on your application's real-time requirements.

ULTRA_LOW_LATENCY_CHUNK = 256
"""
Ultra-low latency: ~16ms @ 16kHz, ~5ms @ 48kHz

Use for: Live audio effects, real-time monitoring, low-latency voice chat
Trade-off: Highest CPU usage due to most frequent callbacks
"""

LOW_LATENCY_CHUNK = 512
"""
Low latency: ~32ms @ 16kHz, ~11ms @ 48kHz

Use for: Voice chat, real-time voice commands, interactive audio
Trade-off: Higher CPU usage for lower latency
"""

BALANCED_CHUNK = 1024
"""
Balanced (DEFAULT): ~64ms @ 16kHz, ~21ms @ 48kHz

Use for: Voice commands, keyword spotting, general audio processing
Trade-off: Good balance between latency and efficiency - recommended for most use cases
"""

HIGH_THROUGHPUT_CHUNK = 2048
"""
High throughput: ~128ms @ 16kHz, ~43ms @ 48kHz

Use for: Speech recognition, transcription, batch processing
Trade-off: Higher latency but more efficient processing
"""

HIGH_THROUGHPUT_CHUNK = 4096
"""
Analysis/recording: ~256ms @ 16kHz, ~85ms @ 48kHz

Use for: Audio recording, non-real-time processing
Trade-off: High latency but efficient for batch operations
"""

ULTRA_HIGH_THROUGHPUT_CHUNK = 8192
"""
Spectral analysis: ~512ms @ 16kHz, ~170ms @ 48kHz

Use for: FFT, spectral analysis, frequency domain processing
Trade-off: Very high latency but optimal for the amount of data moved and CPU overhead
"""
