# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an input WebSocket video"
import time
import numpy as np
from arduino.app_peripherals.camera import Camera


# Expose a WebSocket camera stream for clients to connect to and consume it
camera = Camera("ws://0.0.0.0:8080", timeout=5)
camera.start()

start_time = time.time()
while time.time() - start_time < 5:
    image: np.ndarray = camera.capture()
    # You can process the image here if needed, e.g save it

camera.stop()