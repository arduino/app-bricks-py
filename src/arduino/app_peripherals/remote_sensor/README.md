# RemoteSensor Peripheral

The `RemoteSensor` peripheral receives telemetry data from a remote client, for example a phone, over a WebSocket connection.

## Overview

The RemoteSensor hosts a WebSocket server that accepts a single client at a time. Each message sent by the client is one datapoint, which is handed as raw bytes to a registered callback. The format of the datapoint is up to the app and its client.

## Usage

```python
import json

from arduino.app_peripherals.remote_sensor import RemoteSensor

def handle_datapoint(data: bytes):
    """Called for each datapoint received from the client."""
    reading = json.loads(data)
    print(f"Temperature: {reading['temperature']}°C, Humidity: {reading['humidity']}%")

# Create the sensor and register the callback
sensor = RemoteSensor(port=8090)
sensor.on_datapoint(handle_datapoint)

# Start the server
sensor.start()

# Server runs in background, the callback is invoked for each datapoint
# ...

# When done
sensor.stop()
```

The datapoint is not parsed. For binary data, state the byte order when you interpret values that span several bytes:

```python
import numpy as np

def handle_datapoint(data: bytes):
    values = np.frombuffer(data, dtype="<i2")  # Little-endian int16
```

## Parameters

- `port`: (optional) port of the WebSocket server (default: 8090)
- `timeout`: (optional) connection timeout in seconds (default: 3)
- `certs_dir_path`: (optional) directory of the TLS certificates (default: "/app/certs")
- `use_tls`: (optional) serve over TLS, which makes `encrypt` redundant (default: False)
- `secret`: (optional) pre-shared secret that clients must use to sign their messages (default: None, no security)
- `encrypt`: (optional) also encrypt the messages, requires a `secret` (default: False)
- `auto_reconnect`: (optional) retry when the server fails to start (default: True)
- `idle_timeout`: (optional) seconds without a datapoint after which the stream is reported as paused (default: 3)

## Connection status

Register a callback with `on_status_changed` to follow the client. It receives the new status and the name and address of the client.

- `connected`: the first valid datapoint of a client has arrived. Opening the WebSocket is not enough.
- `streaming`: datapoints are arriving.
- `paused`: no datapoint has arrived for `idle_timeout` seconds. The connection stays open and the status goes back to `streaming` at the next datapoint.
- `disconnected`: the client has disconnected.

```python
def handle_status(status: str, info: dict):
    print(f"{info.get('client_name')} is now {status}")

sensor.on_status_changed(handle_status)
```

## Client Protocol

A client connects to `ws://<board-address>:<port>`, or `wss://` when TLS is enabled. Only one client is served at a time: a second one receives a "Server busy" error and is disconnected.

The address accepts two optional query parameters:

- `client_name`: a name that identifies the client. Only letters, digits, spaces, hyphens and underscores are kept, up to 64 characters.
- `raw`: set it to `true` to exchange plain payloads with no framing. It is honored only when the sensor has no `secret`.

After the connection opens, the server sends a welcome message: a JSON object with `status`, `message` and `security_mode`. The client then sends one WebSocket message per datapoint. When the server stops, it sends a JSON object with `"status": "disconnecting"` before it closes the connection.

### Message format

Unless the client asked for `raw`, every message in both directions is framed as follows:

```
[Version (1)] [Mode (1)] [Timestamp (8)] [Random (4)] [Payload] [Signature (32) or Tag (16)]
```

- **Version**: always 0.
- **Mode**: 0 with no `secret`, 1 with a `secret`, 2 with a `secret` and `encrypt=True`.
- **Timestamp**: microseconds since the Unix epoch, big-endian. Messages more than 10 seconds away from the board clock are dropped.
- **Random**: 4 random bytes. Timestamp and Random together must be different for every message.
- **Payload**: the datapoint, encrypted in mode 2.
- **Signature**: only in mode 1, the HMAC-SHA256 of the previous fields keyed with the secret.
- **Tag**: only in mode 2, the ChaCha20-Poly1305 tag. The key is the SHA-256 of the secret, the nonce is Timestamp followed by Random, and the first four fields are the associated data.

Messages are binary. A text message is accepted too, as the Base64 encoding of the same bytes.

### Client example

A client of a sensor with no `secret`, sending plain JSON datapoints:

```python
import json
import time

import websockets.sync.client as wsclient

with wsclient.connect("ws://<board-address>:8090?client_name=my-sensor&raw=true") as websocket:
    websocket.recv()  # Welcome message
    while True:
        datapoint = {"temperature": 23.5, "humidity": 45.2}
        websocket.send(json.dumps(datapoint).encode())
        time.sleep(1)
```
