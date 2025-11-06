# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an input WebSocket audio stream"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
import time
import numpy as np
from arduino.app_peripherals.microphone import Microphone


# Expose a WebSocket microphone stream for clients to connect to
mic = Microphone("ws://0.0.0.0:8080", timeout=5)
mic.start()

start_time = time.time()
while time.time() - start_time < 5:
    audio: np.ndarray = mic.capture()
    # You can process the audio here if needed, e.g save it

mic.stop()
