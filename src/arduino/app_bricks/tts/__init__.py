# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .local_tts import TextToSpeech, TTSBusyError, TTSError

__all__ = ["TextToSpeech", "TTSError", "TTSBusyError"]
