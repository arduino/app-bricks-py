# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize camera input"
from arduino.app_peripherals.usb_camera import USBCamera


default = USBCamera()

custom = USBCamera(camera=0, resolution=(640, 480), fps=15)
