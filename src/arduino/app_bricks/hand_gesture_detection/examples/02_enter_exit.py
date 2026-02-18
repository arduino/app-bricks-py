# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Hand enter/exit detection"
# EXAMPLE_REQUIRES = "Requires a connected camera"

from arduino.app_bricks.hand_gesture_detection import HandGestureTracking
from arduino.app_peripherals.camera.websocket_camera import WebSocketCamera
from arduino.app_utils.app import App

camera = WebSocketCamera()
camera.start()
pd = HandGestureTracking(camera)
pd.on_enter(lambda: print("Hi there!"))
pd.on_exit(lambda: print("Goodbye!"))

App.run()
