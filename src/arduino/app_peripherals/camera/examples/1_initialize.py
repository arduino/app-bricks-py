# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize camera input"
# EXAMPLE_REQUIRES = "Requires a connected camera"
from arduino.app_peripherals.camera import Camera, V4LCamera


default = Camera() # Uses default camera (V4L)

# The following two are equivalent
camera = Camera(2, resolution=(640, 480), fps=15)  # Infers camera type
v4l = V4LCamera(2, (640, 480), 15)  # Explicitly requests V4L camera

# Note: constructor arguments (except source) must be provided in keyword
# format to forward them correctly to the specific camera implementations.