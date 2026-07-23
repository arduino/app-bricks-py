# Arduino Cloud Brick

This Brick provides integration with the Arduino Cloud platform, enabling IoT devices to communicate and synchronize data seamlessly.

## Overview

The Arduino Cloud Brick lets your application exchange variable values with Arduino Cloud. It does **not** connect to the Cloud itself: connectivity, device provisioning and the cloud handshake are handled by the local **arduino-cloud-connector daemon** running on the board. The Brick talks to that daemon over its localhost REST/SSE API, so your application code stays simple and focused on reading and writing cloud variables.

## Features

- Exchanges variable values with Arduino Cloud through the local daemon
- Natural attribute access to variables (`cloud.my_var = 42`)
- `on_write` / `on_read` / `on_run` callbacks
- Structured objects: `Location`, `Color`, `ColoredLight`, `DimmedLight`, `Schedule`
- Per-variable conflict resolution policy: `DEVICE_WINS`, `CLOUD_WINS`, `MOST_RECENT_WINS`

## Prerequisites

The board must be provisioned and associated with a Thing in Arduino Cloud, and the `arduino-cloud-connector` daemon must be running locally. The daemon owns the device identity and credentials, so the application no longer needs to supply a `device_id` / `secret` to exchange variables.

By default the Brick connects to the daemon at `http://127.0.0.1:5683`. Override it with the `ARDUINO_CLOUD_CONNECTOR_URL` environment variable (or `ARDUINO_CLOUD_CONNECTOR__PORT` to change just the port), or by passing `daemon_url=...` to the constructor.

## Code Example and Usage

```python
from arduino.app_bricks.arduino_cloud import ArduinoCloud
from arduino.app_utils import App, Bridge

iot_cloud = ArduinoCloud()


def led_callback(client: object, value: bool):
    """Called when the LED variable is updated from the cloud."""
    print(f"LED blink value updated from cloud: {value}")
    Bridge.call("set_led_state", value)


iot_cloud.register("led", value=False, on_write=led_callback)

App.run()
```

### Conflict resolution (sync policy)

Each variable can choose how Cloud updates interact with local changes, mirroring the Arduino Cloud (C++) semantics:

- `CLOUD_WINS` (default): an incoming Cloud value is always applied when it differs from the local value.
- `MOST_RECENT_WINS`: a Cloud value is applied only if it is newer than the last local change.
- `DEVICE_WINS`: Cloud values are ignored; the local value is pushed back so the Cloud converges to the device.

```python
from arduino.app_bricks.arduino_cloud import ArduinoCloud, MOST_RECENT_WINS

iot_cloud = ArduinoCloud()
iot_cloud.register("temperature", value=0.0, sync=MOST_RECENT_WINS)
```
