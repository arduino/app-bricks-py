# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .local_tts import SynthesisStream, TextToSpeech, TTSBusyError, TTSError

__all__ = ["TextToSpeech", "TTSError", "TTSBusyError", "SynthesisStream"]
