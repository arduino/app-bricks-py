# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an HLS (HTTP Live Stream) video"
import time
import numpy as np
from arduino.app_peripherals.camera import Camera


# Capture a freely available HLS playlist for testing
# Note: Public streams can be unreliable and may go offline without notice.
url = "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8"

camera = Camera(url)
camera.start()

start_time = time.time()
while time.time() - start_time < 5:
    image: np.ndarray = camera.capture()
    # You can process the image here if needed, e.g save it

camera.stop()
