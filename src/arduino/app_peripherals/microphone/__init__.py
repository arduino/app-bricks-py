# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from .microphone import Microphone
from .base_microphone import BaseMicrophone, FormatPlain, FormatPacked
from .errors import *
from .utils import PauseDetector, chunk_level

if TYPE_CHECKING:
    from .alsa_microphone import ALSAMicrophone
    from .websocket_microphone import WebSocketMicrophone

# Backends are loaded on first access (Microphone() imports the one it needs): the WebSocket one pulls in asyncio,
# websockets and the stream codecs, which would otherwise slow down the start of every app using a microphone.
_LAZY_EXPORTS = {
    "ALSAMicrophone": "alsa_microphone",
    "WebSocketMicrophone": "websocket_microphone",
}

if not TYPE_CHECKING:  # Type checkers resolve the names through the imports above, and still reject unknown ones

    def __getattr__(name: str) -> object:
        import importlib

        submodule = _LAZY_EXPORTS.get(name)
        if submodule is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f".{submodule}", __name__), name)
        globals()[name] = value  # Later accesses skip __getattr__
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *_LAZY_EXPORTS})


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
    "PauseDetector",
    "chunk_level",
]
