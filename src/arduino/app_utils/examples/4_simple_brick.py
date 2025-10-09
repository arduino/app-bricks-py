# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import App, brick, Logger
import time

logger = Logger("ColorDetectorServer")


@brick
class Greeter:
    def __init__(self, name="World"):
        self.name = name

    def start(self):
        logger.info("Starting Greeter")

    def stop(self):
        logger.info("Stopping Greeter")

    # This is a non-blocking method that will be called repeatedly
    def loop(self):
        logger.info(f"Hello, {self.name}!")
        time.sleep(1)


Greeter(input("Enter your name: "))

App.run()
