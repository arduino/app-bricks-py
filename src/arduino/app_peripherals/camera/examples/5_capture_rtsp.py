# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an RTSP (Real-Time Streaming Protocol) video"
import time
import numpy as np
from arduino.app_peripherals.camera import Camera


# Capture a freely available RTSP stream for testing
# Note: Public streams can be unreliable and may go offline without notice.
url = "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa"

camera = Camera(url)
camera.start()

start_time = time.time()
while time.time() - start_time < 5:
    image: np.ndarray = camera.capture()
    # You can process the image here if needed, e.g save it

camera.stop()