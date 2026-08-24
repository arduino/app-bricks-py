# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

from . import bridge as _bridge_module
from .app import *
from .bridge import *
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

# Give the arduino-router-bridge library the standard log format and level
adopt_logger("arduino.router_bridge", display_name="Bridge")

# Connect the bridge eagerly when the app environment configures a router address
if "APP_SOCKET" in os.environ:
    _bridge_module._get_bridge()
