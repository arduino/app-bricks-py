# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ENERGY_THRESHOLD = 80.0
SILENCE_MS = 1800.0
MAX_BUFFER_MS = 12000.0


@dataclass
class VADState:
    buffered_ms: float = 0.0
    silence_ms: float = 0.0
    speaking: bool = False


class VoiceActivityDetector:
    """Energy-based VAD used to decide when to commit buffered audio."""

    def __init__(
        self,
        commit_callback: Callable[[], None],
        min_buffer_ms: float,
        energy_threshold: float = ENERGY_THRESHOLD,
        silence_ms: float = SILENCE_MS,
        max_buffer_ms: float = MAX_BUFFER_MS,
    ):
        self._commit_callback = commit_callback
        self._min_buffer_ms = min_buffer_ms
        self._energy_threshold = energy_threshold
        self._silence_ms_threshold = silence_ms
        self._max_buffer_ms = max_buffer_ms
        self._state = VADState()

    def process_chunk(self, pcm_chunk: bytes, sample_rate: int) -> None:
        """Update VAD state using raw PCM16 bytes and commit buffered audio when thresholds are met."""
        chunk_ms = chunk_duration_ms(pcm_chunk, sample_rate)
        if chunk_ms <= 0:
            return

        pcm_chunk_np = np.frombuffer(pcm_chunk, dtype=np.int16)
        if self._should_commit(pcm_chunk_np, chunk_ms):
            self.commit_buffer()

    def commit_buffer(self) -> None:
        if self._state.buffered_ms >= self._min_buffer_ms:
            self._commit_callback()
        self._state = VADState()

    def flush(self) -> None:
        self.commit_buffer()

    def _chunk_energy(self, pcm_chunk_np: np.ndarray) -> float:
        return float(np.abs(pcm_chunk_np).mean())

    def _should_commit(self, pcm_chunk_np: np.ndarray, chunk_ms: float) -> bool:
        energy = self._chunk_energy(pcm_chunk_np)
        state = self._state
        state.buffered_ms += chunk_ms

        if energy > self._energy_threshold:
            state.speaking = True
            state.silence_ms = 0.0
        elif state.speaking:
            state.silence_ms += chunk_ms
            if state.silence_ms >= self._silence_ms_threshold:
                return True

        if state.buffered_ms >= self._max_buffer_ms:
            return True

        return False


def chunk_duration_ms(pcm_chunk: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    samples = len(pcm_chunk) / 2  # 2 bytes per int16 sample
    return (samples / sample_rate) * 1000.0


__all__ = [
    "MAX_BUFFER_MS",
    "SILENCE_MS",
    "ENERGY_THRESHOLD",
    "VADState",
    "VoiceActivityDetector",
    "chunk_duration_ms",
]
