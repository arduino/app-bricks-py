# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an image"
# EXAMPLE_REQUIRES = "Requires a connected camera"
import numpy as np
from arduino.app_peripherals.camera import Camera


camera = Camera()
camera.start()
image: np.ndarray = camera.capture()
camera.stop()
