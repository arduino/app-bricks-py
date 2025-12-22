# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ASREvent:
    event: str
    data: object | None = None


@runtime_checkable
class ASRProvider(Protocol):
    """Minimal interface for realtime ASR cloud providers."""

    @property
    def provider_name(self) -> str: ...

    @property
    def partial_mode(self) -> str: ...

    def send_audio(self, pcm_chunk: bytes) -> None: ...

    def recv(self) -> ASREvent | None: ...

    def stop(self) -> None: ...
