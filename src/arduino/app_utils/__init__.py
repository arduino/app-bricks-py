# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from .app import *
from .audio import *
from .brick import *
from .bridge import *
from .folderwatch import *
from .httprequest import *
from .jsonparser import *
from .logger import *
from .ledmatrix import *
from .slidingwindowbuffer import *
from .leds import *

__all__ = [
    "App",
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

# Dynamically load additional bricks from /app/bricks
import sys
import os

# Load bricks from /app/bricks if it exists
if os.path.exists('/app/bricks'):
    sys.path.insert(0, '/app/bricks')
