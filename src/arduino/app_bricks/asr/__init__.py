# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .local_asr import (
    ASREvent,
    ASRBusyError,
    ASRError,
    ASRServiceBusyError,
    AutomaticSpeechRecognition,
    TranscriptionStream,
)

__all__ = [
    "ASREvent",
    "ASRError",
    "ASRBusyError",
    "ASRServiceBusyError",
    "AutomaticSpeechRecognition",
    "TranscriptionStream",
]
