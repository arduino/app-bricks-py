# Camera

The `Camera` peripheral provides a unified abstraction for capturing images from different camera types and protocols.

## Features

- **Universal Interface**: Single API for V4L/USB, IP cameras, and WebSocket cameras
- **Automatic Detection**: Automatically selects appropriate camera implementation based on source
- **Multiple Protocols**: Supports V4L, RTSP, HTTP/MJPEG, and WebSocket streams
- **Flexible Configuration**: Resolution, FPS, compression, and protocol-specific settings
- **Thread-Safe**: Safe concurrent access with proper locking
- **Context Manager**: Automatic resource management with `with` statements

## Quick Start

### Basic Usage

```python
from arduino.app_peripherals.camera import Camera

# USB/V4L camera (index 0)
camera = Camera(0, resolution=(640, 480), fps=15)

with camera:
    frame = camera.capture()  # Returns PIL Image
    if frame:
        frame.save("captured.png")
```

### Different Camera Types

```python
# V4L/USB cameras
usb_camera = Camera(0)                    # Camera index
usb_camera = Camera("1")                  # Index as string  
usb_camera = Camera("/dev/video0")        # Device path

# IP cameras
ip_camera = Camera("rtsp://192.168.1.100/stream")
ip_camera = Camera("http://camera.local/mjpeg", 
                   username="admin", password="secret")

# WebSocket cameras  
- `"ws://localhost:8080"` - WebSocket server URL (extracts host and port)
- `"localhost:9090"` - WebSocket server host:port format
```

## API Reference

### Camera Class

The main `Camera` class acts as a factory that creates the appropriate camera implementation:

```python
camera = Camera(source, **options)
```

**Parameters:**
- `source`: Camera source identifier
  - `int`: V4L camera index (0, 1, 2...)
  - `str`: Camera index, device path, or URL
- `resolution`: Tuple `(width, height)` or `None` for default
- `fps`: Target frames per second (default: 10)
- `compression`: Enable PNG compression (default: False)  
- `letterbox`: Make images square with padding (default: False)

**Methods:**
- `start()`: Initialize and start camera
- `stop()`: Stop camera and release resources
- `capture()`: Capture frame as PIL Image
- `capture_bytes()`: Capture frame as bytes
- `is_started()`: Check if camera is running
- `get_camera_info()`: Get camera properties

### Context Manager

```python
with Camera(source, **options) as camera:
    frame = camera.capture()
    # Camera automatically stopped when exiting
```

## Camera Types

### V4L/USB Cameras

For local USB cameras and V4L-compatible devices:

```python
camera = Camera(0, resolution=(1280, 720), fps=30)
```

**Features:**
- Device enumeration via `/dev/v4l/by-id/`
- Resolution validation
- Backend information

### IP Cameras

For network cameras supporting RTSP or HTTP streams:

```python
camera = Camera("rtsp://admin:pass@192.168.1.100/stream", 
                timeout=10, fps=5)
```

**Features:**
- RTSP, HTTP, HTTPS protocols
- Authentication support
- Connection testing
- Automatic reconnection

### WebSocket Cameras

For hosting a WebSocket server that receives frames from clients (single client only):

```python
camera = Camera("ws://0.0.0.0:9090", frame_format="json")
```

**Features:**
- Hosts WebSocket server (not client)
- **Single client limitation**: Only one client can connect at a time
- Additional clients are rejected with error message
- Receives frames from connected client
- Base64, binary, and JSON frame formats
- Frame buffering and queue management
- Bidirectional communication with connected client

**Client Connection:**
Only one client can connect at a time. Additional clients receive an error:
```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8080');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.error) {
        console.log('Connection rejected:', data.message);
    }
};
ws.send(base64EncodedImageData);
```

## Advanced Usage

### Custom Configuration

```python
camera = Camera(
    source="rtsp://camera.local/stream",
    resolution=(1920, 1080),
    fps=15,
    compression=True,      # PNG compression
    letterbox=True,        # Square images
    username="admin",      # IP camera auth
    password="secret",
    timeout=5,             # Connection timeout
    max_queue_size=20      # WebSocket buffer
)
```

### Error Handling

```python
from arduino.app_peripherals.camera.camera import CameraError

try:
    with Camera("invalid://source") as camera:
        frame = camera.capture()
except CameraError as e:
    print(f"Camera error: {e}")
```

### Factory Pattern

```python
from arduino.app_peripherals.camera.camera import CameraFactory

# Create camera directly via factory
camera = CameraFactory.create_camera(
    source="ws://localhost:8080/stream",
    frame_format="json"
)
```

## Dependencies

### Core Dependencies
- `opencv-python` (cv2) - Image processing and V4L/IP camera support
- `Pillow` (PIL) - Image format handling  
- `requests` - HTTP camera connectivity testing

### Optional Dependencies
- `websockets` - WebSocket server support (install with `pip install websockets`)

## Examples

See the `examples/` directory for comprehensive usage examples:
- Basic camera operations
- Different camera types
- Advanced configuration
- Error handling
- Context managers

## Migration from Legacy Camera

The new Camera abstraction is backward compatible with the existing Camera implementation. Existing code using the old API will continue to work, but new code should use the improved abstraction for better flexibility and features.
