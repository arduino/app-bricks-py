# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .speaker import Speaker
from .base_speaker import BaseSpeaker
from .alsa_speaker import ALSASpeaker
from .errors import *

__all__ = [
    "Speaker",
    "BaseSpeaker",
    "ALSASpeaker",
    "SpeakerError",
    "SpeakerOpenError",
    "SpeakerWriteError",
    "SpeakerConfigError",
]
