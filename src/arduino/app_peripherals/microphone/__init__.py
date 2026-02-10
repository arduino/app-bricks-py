# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .microphone import Microphone
from .base_microphone import BaseMicrophone, FormatPlain, FormatPacked
from .alsa_microphone import ALSAMicrophone
from .websocket_microphone import WebSocketMicrophone
from .errors import *

__all__ = [
    "Microphone",
    "BaseMicrophone",
    "ALSAMicrophone",
    "WebSocketMicrophone",
    "FormatPlain",
    "FormatPacked",
    "MicrophoneError",
    "MicrophoneConfigError",
    "MicrophoneOpenError",
    "MicrophoneReadError",
]
