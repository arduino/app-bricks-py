# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from typing import Literal

from arduino.app_peripherals.camera import BaseCamera, Camera


class HandGestureTracking:
    def __init__(self, camera: BaseCamera | None = None):
        if camera is None:
            camera = Camera(fps=30)
        self.camera = camera

    def on_gesture(self, gesture, callback, hand: Literal["left", "right", "both"] = "both"):
        pass

    def on_enter(self, callback, hand: Literal["left", "right", "both"] = "both"):
        pass

    def on_exit(self, callback, hand: Literal["left", "right", "both"] = "both"):
        pass

    def on_frame(self, callback):
        pass
