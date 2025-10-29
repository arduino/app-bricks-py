# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture a video"
# EXAMPLE_REQUIRES = "Requires a connected camera"
import time
import numpy as np
from arduino.app_peripherals.camera import Camera


# Capture a video for 5 seconds at 15 FPS
camera = Camera(fps=15)
camera.start()

start_time = time.time()
while time.time() - start_time < 5:
    image: np.ndarray = camera.capture()
    # You can process the image here if needed, e.g save it

camera.stop()