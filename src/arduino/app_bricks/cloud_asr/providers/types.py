# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Literal

ASRProviderEventType = Literal[
    "speech_start",
    "partial_text",
    "speech_stop",
    "text",
    "utterance_end",
]


@dataclass(frozen=True)
class ASRProviderEvent:
    type: ASRProviderEventType
    data: str | None = None


class ASRProviderError(Exception):
    """Base class for ASR-related errors."""

    pass


@runtime_checkable
class ASRProvider(Protocol):
    """Minimal interface for realtime ASR cloud providers."""

    @property
    def provider_name(self) -> str: ...

    @property
    def partial_mode(self) -> str: ...

    def start(self) -> None: ...

    def send_audio(self, pcm_chunk: bytes) -> None: ...

    def recv(self) -> ASRProviderEvent | None: ...

    def stop(self) -> None: ...
