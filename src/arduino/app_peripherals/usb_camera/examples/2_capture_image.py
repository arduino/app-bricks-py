# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Capture an image"
# EXAMPLE_REQUIRES = "Requires a connected camera"
from arduino.app_peripherals.usb_camera import USBCamera
from PIL.Image import Image


camera = USBCamera()
camera.start()
image: Image = camera.capture()
camera.stop()
