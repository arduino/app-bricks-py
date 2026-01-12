# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

ASREventType = Literal[
    "speech_start",
    "partial_text",
    "speech_stop",
    "text",
]

ASREventTypeValues = get_args(ASREventType)


@dataclass(frozen=True)
class ASREvent:
    type: ASREventType
    data: str | None = None
