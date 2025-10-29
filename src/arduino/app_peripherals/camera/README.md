# Camera

The `Camera` peripheral provides a unified abstraction for capturing images from different camera types and protocols.

## Features

- **Universal Interface**: Single API for V4L/USB, IP cameras, and WebSocket cameras
- **Automatic Detection**: Selects appropriate camera implementation based on source
- **Multiple Protocols**: Supports V4L, RTSP, HTTP/MJPEG, and WebSocket streams
- **Thread-Safe**: Safe concurrent access with proper locking
- **Context Manager**: Automatic resource management

## Quick Start

Instantiate the default camera:
```python
from arduino.app_peripherals.camera import Camera

# Default camera (V4L camera at index 0)
camera = Camera()
```

Camera needs to be started and stopped explicitly:

```python
# Specify camera and configuration
camera = Camera(0, resolution=(640, 480), fps=15)
camera.start()

image = camera.capture()

camera.stop()
```

Or you can leverage context support for doing that automatically:
```python
with Camera(source, **options) as camera:
    frame = camera.capture()
    if frame is not None:
        print(f"Captured frame with shape: {frame.shape}")
    # Camera automatically stopped when exiting
```

## Camera Types
The Camera class provides automatic camera type detection based on the format of its source argument. keyword arguments will be propagated to the underlying implementation.

Note: constructor arguments (except source) must be provided in keyword format to forward them correctly to the specific camera implementations.

The underlying camera implementations can be instantiated explicitly (V4LCamera, IPCamera and WebSocketCamera), if needed.

### V4L Cameras
For local USB cameras and V4L-compatible devices.

**Features:**
- Supports cameras compatible with the Video4Linux2 drivers

```python
camera = Camera(0)                    # Camera index
camera = Camera("/dev/video0")        # Device path
camera = V4LCamera(0)
```

### IP Cameras
For network cameras supporting RTSP (Real-Time Streaming Protocol) and HLS (HTTP Live Streaming).

**Features:**
- Supports capturing RTSP, HLS streams
- Authentication support
- Automatic reconnection

```python
camera = Camera("rtsp://admin:secret@192.168.1.100/stream")
camera = Camera("http://camera.local/stream",
                username="admin", password="secret")
camera = IPCamera("http://camera.local/stream", 
                username="admin", password="secret")
```

### WebSocket Cameras
For hosting a WebSocket server that receives frames from a single client at a time.

**Features:**
- **Single client limitation**: Only one client can connect at a time
- Stream data from any client with WebSockets support
- Base64, binary, and JSON frame formats
- Supports 8-bit images (e.g. JPEG, PNG 8-bit)

```python
camera = Camera("ws://0.0.0.0:8080", timeout=5)
camera = WebSocketCamera("0.0.0.0", 8080, timeout=5)
```

Client implementation example:
```python
import time
import base64
import cv2
import websockets.sync.client as wsclient
import websockets.exceptions as wsexc


# Open camera
camera = cv2.VideoCapture(0)
with wsclient.connect("ws://<board-address>:8080") as websocket:
    while True:
        time.sleep(1.0 / 15.0)  # 15 FPS
        ret, frame = camera.read()
        if ret:
            # Compress frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64
            jpeg_b64 = base64.b64encode(buffer).decode('utf-8')
            try:
                websocket.send(jpeg_b64)
            except wsexc.ConnectionClosed:
                break
```

## Migration from Legacy Camera

The new Camera abstraction is backward compatible with the existing Camera implementation. Existing code using the old API will continue to work, but will use the new Camera backend. New code should use the improved abstraction for better flexibility and features.
