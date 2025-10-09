# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import App, brick, Logger
import time

logger = Logger("ColorDetectorServer")


@brick
class ColorDetectorServer:
    def __init__(self, color):
        self.color = color

    def start(self):
        logger.info(f"Starting ColorDetectorServer for color: {self.color}")

    def stop(self):
        logger.info(f"Stopping ColorDetectorServer for color: {self.color}")

    # This is a blocking method that will be run in a separate thread
    def execute(self):
        logger.info("Execute: Starting bluetooth server (simulated 5s task)...")
        time.sleep(5)
        logger.info("Execute: Bluetooth server task finished.")

    # This is an additional blocking method that will be run in a separate thread
    @brick.execute
    def another_execute(self):
        logger.info("Execute: Starting web server (simulated 5s task)...")
        time.sleep(5)
        logger.info("Execute: Web server task finished.")

    # This is a non-blocking method that will be called repeatedly
    def loop(self):
        logger.info("Loop: Reading from webcam.")
        time.sleep(1)

    # This is an additional non-blocking method that will be called repeatedly
    @brick.loop
    def another_loop(self):
        logger.info("Loop: Running AI inference.")
        time.sleep(1)


ColorDetectorServer("blue")

App.run()
