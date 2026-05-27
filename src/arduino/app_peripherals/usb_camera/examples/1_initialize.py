# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Initialize camera input"
from arduino.app_peripherals.usb_camera import USBCamera


default = USBCamera()

custom = USBCamera(camera=0, resolution=(640, 480), fps=15)
