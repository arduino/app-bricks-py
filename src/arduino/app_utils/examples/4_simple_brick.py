# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import App, brick, Logger
import time

logger = Logger("ColorDetectorServer")


@brick
class Greeter:
    def __init__(self, name="World") -> None:
        self.name = name

    def start(self) -> None:
        logger.info("Starting Greeter")

    def stop(self) -> None:
        logger.info("Stopping Greeter")

    # This is a non-blocking method that will be called repeatedly
    def loop(self) -> None:
        logger.info(f"Hello, {self.name}!")
        time.sleep(1)


Greeter(input("Enter your name: "))

App.run()
