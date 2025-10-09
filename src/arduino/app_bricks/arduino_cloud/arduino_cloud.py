# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino_iot_cloud import ArduinoCloudClient
from arduino.app_utils import brick, Logger
import threading
import time
from typing import Any
import os

logger = Logger("ArduinoCloud")


@brick
class ArduinoCloud:
    """Arduino Cloud client for managing devices and data."""

    def __init__(
        self,
        device_id: str = os.getenv("ARDUINO_DEVICE_ID"),
        secret: str = os.getenv("ARDUINO_SECRET"),
        server: str = "iot.arduino.cc",
        port: int = 8884,
    ):
        """Initialize the Arduino Cloud client.

        Args:
            device_id (str): The unique identifier for the device.
                             If omitted, uses ARDUINO_DEVICE_ID environment variable.
            secret (str): The password for Arduino Cloud authentication.
                          If omitted, uses ARDUINO_SECRET environment variable.
            server (str, optional): The server address for Arduino Cloud (default: "iot.arduino.cc").
            port (int, optional): The port to connect to the Arduino Cloud server (default: 8884).

        Raises:
            ValueError: If either device_id or secret is not provided explicitly or via environment variable.

        """
        if device_id is None:
            raise ValueError("'device_id' must be provided or set ARDUINO_DEVICE_ID environment variable")
        if secret is None:
            raise ValueError("'secret' must be provided or set ARDUINO_SECRET environment variable")

        self._client: ArduinoCloudClient = ArduinoCloudClient(
            device_id=device_id, username=device_id, password=secret, server=server, port=port, sync_mode=True
        )
        self._client_lock = threading.Lock()

    def start(self):
        """Start the Arduino IoT Cloud client."""
        self._client.start()

    def loop(self):
        """Run a single iteration of the Arduino IoT Cloud client loop, processing commands and updating state."""
        try:
            with self._client_lock:
                self._client.update()
            time.sleep(0.1)
        except Exception as e:
            logger.exception(f"Loop error: {e}")

    def register(self, aiotobj: str | Any, **kwargs: Any):
        """Register a variable or object with the Arduino Cloud client.

        Args:
            aiotobj (str | Any): The variable name or object from which to derive the variable name to register.
            **kwargs (Any): Additional keyword arguments for registration.
        """
        self._client.register(aiotobj, coro=None, **kwargs)

    def __getattr__(self, name: str):
        """Intercept access to cloud variables as natural attributes."""
        with self._client_lock:
            try:
                return self._client[name]
            except KeyError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Intercept assignment to cloud variables as natural attributes."""
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        with self._client_lock:
            self._client[name] = value
