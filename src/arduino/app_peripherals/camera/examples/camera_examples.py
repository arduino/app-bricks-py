# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Camera Abstraction Usage Examples

This file demonstrates various ways to instantiate and use the Camera abstraction
for different camera types and protocols.
"""

import time
from arduino.app_peripherals.camera import Camera
from arduino.app_peripherals.camera.camera import CameraFactory


def example_v4l_camera():
    """Example: Using a V4L/USB camera"""
    print("=== V4L/USB Camera Example ===")
    
    # Method 1: Using Camera class (recommended)
    camera = Camera(0, resolution=(640, 480), fps=15)
    
    try:
        # Start the camera
        camera.start()
        print(f"Camera started: {camera.get_camera_info()}")
        
        # Capture some frames
        for i in range(5):
            frame = camera.capture()
            if frame:
                print(f"Captured frame {i+1}: {frame.size} pixels")
            else:
                print(f"Failed to capture frame {i+1}")
            time.sleep(0.5)
    
    finally:
        camera.stop()
    
    print()


def example_v4l_camera_context_manager():
    """Example: Using V4L camera with context manager"""
    print("=== V4L Camera with Context Manager ===")
    
    # Context manager automatically handles start/stop
    with Camera("0", resolution=(320, 240), fps=10, letterbox=True) as camera:
        print(f"Camera info: {camera.get_camera_info()}")
        
        # Capture a few frames
        for i in range(3):
            frame = camera.capture()
            if frame:
                print(f"Frame {i+1}: {frame.size}")
            time.sleep(1.0)
    
    print("Camera automatically stopped\n")


def example_ip_camera():
    """Example: Using an IP camera (RTSP/HTTP)"""
    print("=== IP Camera Example ===")
    
    # Example RTSP URL (replace with your camera's URL)
    rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"
    
    # Method 1: Direct instantiation
    camera = Camera(rtsp_url, fps=5)
    
    try:
        # Test connection first
        if hasattr(camera, 'test_connection') and camera.test_connection():
            print("IP camera is accessible")
            
            camera.start()
            print(f"IP camera started: {camera.get_camera_info()}")
            
            # Capture frames
            for i in range(3):
                frame = camera.capture()
                if frame:
                    print(f"IP frame {i+1}: {frame.size}")
                else:
                    print(f"No frame received {i+1}")
                time.sleep(2.0)
        else:
            print("IP camera not accessible (expected for this example)")
    
    except Exception as e:
        print(f"IP camera error (expected): {e}")
    
    finally:
        camera.stop()
    
    print()


def example_websocket_camera():
    """Example: Using a WebSocket camera server (single client only)"""
    print("=== WebSocket Camera Server Example (Single Client) ===")
    
    # Create WebSocket camera server
    try:
        # Method 1: Direct host:port specification
        camera = Camera("ws://localhost:8080", frame_format="base64")
        
        camera.start()
        print(f"WebSocket camera server started: {camera.get_camera_info()}")
        
        # Server is now listening for client connections (max 1 client)
        print("Server is waiting for ONE client to connect and send frames...")
        print("Additional clients will be rejected with an error message")
        print("Clients should connect to ws://localhost:8080 and send base64 encoded images")
        
        # Monitor for incoming frames
        for i in range(10):  # Check for 10 seconds
            frame = camera.capture()
            if frame:
                print(f"Received frame {i+1}: {frame.size}")
            else:
                print(f"No frame received in iteration {i+1}")
            
            time.sleep(1.0)
    
    except Exception as e:
        print(f"WebSocket camera server error (expected if no clients connect): {e}")
    
    finally:
        if 'camera' in locals():
            camera.stop()
    
    print()


def example_websocket_server_with_url():
    """Example: WebSocket server using ws:// URL (single client only)"""
    print("=== WebSocket Server with URL Example (Single Client) ===")
    
    try:
        # Method 2: Using ws:// URL (server extracts host and port)
        camera = Camera("ws://0.0.0.0:9090", frame_format="json")
        
        camera.start()
        
        # Wait briefly for potential connections
        time.sleep(2)
        
        camera.stop()
        print("WebSocket server stopped")
    
    except Exception as e:
        print(f"WebSocket server URL error: {e}")
    
    print()


def example_factory_usage():
    """Example: Using CameraFactory directly"""
    print("=== Camera Factory Example ===")
    
    # Different ways to create cameras using the factory
    sources = [
        0,  # V4L camera index
        "1",  # V4L camera as string
        "/dev/video0",  # V4L device path
        "rtsp://example.com/stream",  # RTSP camera
        "http://example.com/mjpeg",  # HTTP camera
        "ws://localhost:8080",  # WebSocket server URL
        "localhost:9090",  # WebSocket server host:port
        "0.0.0.0:8888",  # WebSocket server on all interfaces
    ]
    
    for source in sources:
        try:
            camera = CameraFactory.create_camera(source, fps=10)
            print(f"Created {camera.__class__.__name__} for source: {source}")
            # Don't start cameras in this example
        except Exception as e:
            print(f"Cannot create camera for {source}: {e}")
    
    print()


def example_advanced_configuration():
    """Example: Advanced camera configuration"""
    print("=== Advanced Configuration Example ===")
    
    # V4L camera with all options
    v4l_config = {
        'resolution': (1280, 720),
        'fps': 30,
        'compression': True,  # PNG compression
        'letterbox': True,    # Square images
    }
    
    try:
        with Camera(0, **v4l_config) as camera:
            print(f"V4L config: {camera.get_camera_info()}")
            
            # Capture compressed frame
            frame = camera.capture()
            if frame:
                print(f"Compressed frame: {frame.format} {frame.size}")
            
            # Capture as bytes
            frame_bytes = camera.capture_bytes()
            if frame_bytes:
                print(f"Frame bytes length: {len(frame_bytes)}")
    
    except Exception as e:
        print(f"Advanced config error: {e}")
    
    # IP camera with authentication
    ip_config = {
        'username': 'admin',
        'password': 'secret',
        'timeout': 5,
        'fps': 10
    }
    
    try:
        ip_camera = Camera("http://192.168.1.100/mjpeg", **ip_config)
        print(f"IP camera with auth created: {ip_camera.__class__.__name__}")
    except Exception as e:
        print(f"IP camera with auth error: {e}")
    
    # WebSocket server with different frame formats
    ws_configs = [
        {'host': 'localhost', 'port': 8080, 'frame_format': 'base64'},
        {'host': '0.0.0.0', 'port': 9090, 'frame_format': 'json'},
        {'host': '127.0.0.1', 'port': 8888, 'frame_format': 'binary'},
    ]
    
    for config in ws_configs:
        try:
            ws_camera = Camera("localhost:8080", **config)  # Will use the config params
            print(f"WebSocket server config: {config}")
        except Exception as e:
            print(f"WebSocket server config error: {e}")
    
    print()


def example_error_handling():
    """Example: Proper error handling"""
    print("=== Error Handling Example ===")
    
    # Try to open non-existent camera
    try:
        camera = Camera(99)  # Non-existent camera
        camera.start()
    except Exception as e:
        print(f"Expected error for invalid camera: {e}")
    
    # Try invalid URL
    try:
        camera = Camera("invalid://url")
    except Exception as e:
        print(f"Expected error for invalid URL: {e}")
    
    print()


if __name__ == "__main__":
    print("Camera Abstraction Examples\n")
    print("Note: Some examples may show errors if cameras are not available.\n")
    
    # Run examples
    example_factory_usage()
    example_advanced_configuration()
    example_error_handling()
    
    # Uncomment these if you have actual cameras available:
    # example_v4l_camera()
    # example_v4l_camera_context_manager()
    # example_ip_camera()
    # example_websocket_camera()
    # example_websocket_server_with_url()
    
    print("Examples completed!")