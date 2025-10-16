# Camera Examples

This directory contains examples demonstrating how to use the Camera abstraction for different types of cameras and protocols.

## Files

- `camera_examples.py` - Comprehensive examples showing all camera types and usage patterns

## Running Examples

```bash
python examples/camera_examples.py
```

## Example Types Covered

### 1. V4L/USB Cameras
- Basic usage with camera index
- Context manager pattern
- Resolution and FPS configuration
- Frame format options

### 2. IP Cameras  
- RTSP streams
- HTTP/MJPEG streams
- Authentication
- Connection testing

### 3. WebSocket Camera Servers
- Hosting WebSocket servers (single client only)
- Receiving frames from one connected client
- Client rejection when server is at capacity
- Multiple frame formats (base64, binary, JSON)
- Bidirectional communication with client
- Server status monitoring

### 4. Factory Pattern
- Automatic camera type detection
- Multiple instantiation methods
- Error handling

### 5. Advanced Configuration
- Compression settings
- Letterboxing
- Custom parameters
- Performance tuning

## Camera Source Formats

The Camera class automatically detects the appropriate implementation based on the source:

- `0`, `1`, `"0"` - V4L camera indices
- `"/dev/video0"` - V4L device paths  
- `"rtsp://..."` - RTSP streams
- `"http://..."` - HTTP streams
- `"ws://localhost:8080"` - WebSocket server URL
- `"localhost:9090"` - WebSocket server host:port