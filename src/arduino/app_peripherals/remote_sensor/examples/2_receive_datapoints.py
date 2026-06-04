# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Receive telemetry datapoints from a client"
import json

from arduino.app_peripherals.remote_sensor import RemoteSensor
from arduino.app_utils.app import App


def on_datapoint(data: bytes):
    # The payload is whatever the client sent (JSON, raw bytes, etc.)
    try:
        payload = json.loads(data.decode())
        print(f"Received datapoint: {payload}")
    except Exception:
        print(f"Received {len(data)} raw bytes")


def on_status(status: str, info: dict):
    print(f"Sensor status changed to '{status}': {info}")


sensor = RemoteSensor(port=8080)
sensor.on_datapoint(on_datapoint)
sensor.on_status_changed(on_status)
sensor.start()

App.run()
