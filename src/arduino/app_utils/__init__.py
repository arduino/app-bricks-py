# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from ._lazy_exports import lazy_exports as _lazy_exports
from .app import *
from .app import install_startup_signal_handler
from .bridge import *
from .logger import configure_library_logger as _configure_library_logger
from .brick import *
from .errors import *
from .errors import install_excepthook as _install_excepthook
from .jsonparser import *
from .logger import *
from .peripheral import *
from .leds import *

if TYPE_CHECKING:
    from .audio import SineGenerator
    from .folderwatch import FolderEventHandler as FolderEventHandler, FolderWatcher
    from .httprequest import HttpClient
    from .ledmatrix import Frame, FrameDesigner
    from .slidingwindowbuffer import SlidingWindowBuffer

# Loaded on first access: their submodules need numpy, requests or watchdog, which cost about a second
# of app start time on the board even when the app never uses them. A star import resolves every name
# in __all__, so the submodules also defer those dependencies to the code that uses them.
_LAZY_EXPORTS = {
    "SineGenerator": "audio",
    "FolderWatcher": "folderwatch",
    "FolderEventHandler": "folderwatch",
    "HttpClient": "httprequest",
    "Frame": "ledmatrix",
    "FrameDesigner": "ledmatrix",
    "SlidingWindowBuffer": "slidingwindowbuffer",
}


if not TYPE_CHECKING:  # Type checkers resolve the names through the imports above
    __getattr__, __dir__ = _lazy_exports(__name__, globals(), _LAZY_EXPORTS)


__all__ = [
    "App",
    "AppError",
    "brick",
    "peripheral",
    "Bridge",
    "notify",
    "call",
    "provide",
    "FolderWatcher",
    "Frame",
    "FrameDesigner",
    "HttpClient",
    "JSONParser",
    "Logger",
    "SineGenerator",
    "SlidingWindowBuffer",
    "Leds",
]

# Report uncaught AppErrors with a user-readable message instead of a bare traceback
_install_excepthook()

# A stop requested before App.run() must not wait for the launcher to kill the process
install_startup_signal_handler()

# Apply the standard log format and level to the arduino-router-bridge library's logger
_configure_library_logger("arduino.router_bridge", display_name="Bridge")
