# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Hand gesture detection"
# EXAMPLE_REQUIRES = "Requires a connected camera"

from arduino.app_bricks.hand_gesture_detection import HandGestureTracking
from arduino.app_peripherals.camera.websocket_camera import WebSocketCamera
from arduino.app_utils.app import App

camera = WebSocketCamera()
camera.start()
pd = HandGestureTracking(camera)
pd.on_gesture("Victory", lambda meta: print("All your bases are belong to us"))
pd.on_gesture("Open_Palm", lambda meta: print("Moving left!"), hand="left")
pd.on_gesture("Open_Palm", lambda meta: print("Moving right!"), hand="right")

App.run()
