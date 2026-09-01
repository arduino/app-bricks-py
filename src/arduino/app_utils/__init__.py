# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from . import bridge as bridge_module
from .app import *
from .bridge import *
from .logger import _configure_library_logger
from .audio import *
from .brick import *
from .errors import *
from .errors import install_excepthook as _install_excepthook
from .folderwatch import *
from .httprequest import *
from .jsonparser import *
from .ledmatrix import *
from .logger import *
from .slidingwindowbuffer import *
from .leds import *

__all__ = [
    "App",
    "AppError",
    "brick",
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

# Apply the standard log format and level to the arduino-router-bridge library's logger
_configure_library_logger("arduino.router_bridge", display_name="Bridge")

# Connect the bridge eagerly
bridge_module._get_bridge()
