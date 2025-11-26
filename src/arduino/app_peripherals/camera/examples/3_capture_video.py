# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture a video"
# EXAMPLE_REQUIRES = "Requires a connected camera"
import time
import numpy as np
from arduino.app_peripherals.camera import Camera


camera = Camera(fps=15)
camera.start()

# You can capture a video by capturing frames in a loop
start_time = time.time()
while time.time() - start_time < 5:
    image: np.ndarray = camera.capture()
    # You can process the image here if needed, e.g save it

# Or you can obtain the same in a single sentence
recording: np.ndarray = camera.record(duration=5)

# Or you can ask for an AVI recording
recording: np.ndarray = camera.record_avi(duration=5)

camera.stop()
