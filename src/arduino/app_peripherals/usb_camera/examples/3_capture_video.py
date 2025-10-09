# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture a video"
# EXAMPLE_REQUIRES = "Requires a connected camera"
from arduino.app_peripherals.usb_camera import USBCamera
from PIL.Image import Image
import time


# Capture a video for 5 seconds at 15 FPS
camera = USBCamera(fps=15)
camera.start()

start_time = time.time()
while time.time() - start_time < 5:
    image: Image = camera.capture()
    # You can process the image here if needed, e.g save it

camera.stop()
