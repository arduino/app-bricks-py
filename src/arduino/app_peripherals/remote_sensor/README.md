# RemoteSensor Peripheral

The `RemoteSensor` peripheral allows you to receive IoT telemetry data from a remote client over WebSocket connections.

## Overview

The RemoteSensor hosts a WebSocket server that accepts a single client connection at a time. Clients can send telemetry data in JSON, CSV, or binary format that is passed to a registered callback function.

## Usage

```python
from arduino.app_peripherals.remote_sensor import RemoteSensor

def handle_telemetry(data: dict):
    """Called when telemetry data is received from the client."""
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    print(f"Temperature: {temperature}°C, Humidity: {humidity}%")

# Create and configure the sensor
sensor = RemoteSensor(host="0.0.0.0", port=9000, data_format="json")

# Register the callback
sensor.on_datapoint(handle_telemetry)

# Start the server
sensor.start()

# Server runs in background, callback is invoked for each message
# ...

# When done
sensor.stop()
```

## Client Protocol

Clients can send data in three formats:

### JSON Format

Clients must send valid JSON objects over the WebSocket connection:

```json
{
  "temperature": 23.5,
  "humidity": 45.2,
  "timestamp": "2025-11-05T12:00:00Z"
}
```

The callback receives the parsed JSON object directly.

### CSV Format

Clients send CSV-formatted data with line separators. Each line represents a sensor reading:

```
temperature,23.5
humidity,45.2
sensor1,100.3,
sensor2,200.5
```

Supported field separators: `,` (comma), `\t` (tab), ` ` (space)
Supported line separators: `\r\n` (CRLF), `\n` (LF)

The callback receives `{"csv": "line_content"}` for each line. If a WebSocket message contains multiple lines, the callback is invoked once for each non-empty line.

### Binary Format

Clients send binary data over the WebSocket connection. The data is converted to a numpy uint8 array.

The callback receives `{"binary": numpy_array}` where numpy_array is the binary data as uint8.

**Byte Ordering:** The binary data is received as-is in a numpy uint8 array. If you need to interpret multi-byte values (int16, int32, float32, etc.), you must explicitly specify the byte order when converting. For example:

```python
def handle_binary(data: dict):
    binary = data["binary"]  # numpy uint8 array
    
    # Interpret as little-endian int16 values
    int16_values = np.frombuffer(binary, dtype='<i2')
    
    # Interpret as big-endian float32 values
    float32_values = np.frombuffer(binary, dtype='>f4')
    
    # Native byte order (platform-dependent)
    int32_values = np.frombuffer(binary, dtype=np.int32)
```
