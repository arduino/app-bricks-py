# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .app import *
from .app import install_startup_signal_handler as _install_startup_signal_handler
from .bridge import *
from .logger import configure_library_logger as _configure_library_logger
from .audio import *
from .brick import *
from .errors import *
from .errors import install_excepthook as _install_excepthook
from .folderwatch import *
from .jsonparser import *
from .ledmatrix import *
from .logger import *
from .peripheral import *
from .slidingwindowbuffer import *
from .leds import *

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
    "JSONParser",
    "Logger",
    "SineGenerator",
    "SlidingWindowBuffer",
    "Leds",
]

# Report uncaught AppErrors with a user-readable message instead of a bare traceback
_install_excepthook()

# A stop requested before App.run() must not wait for the launcher to kill the process
_install_startup_signal_handler()

# Apply the standard log format and level to the arduino-router-bridge library's logger
_configure_library_logger("arduino.router_bridge", display_name="Bridge")
