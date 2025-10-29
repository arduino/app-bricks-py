# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union
from urllib.parse import urlparse

from .base_camera import BaseCamera
from .errors import CameraConfigError


class Camera:
    """
    Unified Camera class that can be configured for different camera types.
    
    This class serves as both a factory and a wrapper, automatically creating
    the appropriate camera implementation based on the provided configuration.
    """
    
    def __new__(cls, source: Union[str, int] = 0, **kwargs) -> BaseCamera:
        """Create a camera instance based on the source type.
        
        Args:
            source (Union[str, int]): Camera source identifier. Supports:
                - int: V4L camera index (e.g., 0, 1)
                - str: Camera index as string (e.g., "0", "1") for V4L
                - str: Device path (e.g., "/dev/video0") for V4L
                - str: URL for IP cameras (e.g., "rtsp://...", "http://...")
                - str: WebSocket URL (e.g., "ws://0.0.0.0:8080")
            **kwargs: Camera-specific configuration parameters grouped by type:
                Common Parameters:
                    resolution (tuple, optional): Frame resolution as (width, height). 
                        Default: None (auto)
                    fps (int, optional): Target frames per second. Default: 10
                    adjustments (callable, optional): Function pipeline to adjust frames that takes a
                        numpy array and returns a numpy array. Default: None
                V4L Camera Parameters:
                    device_index (int, optional): V4L device index override
                    capture_format (str, optional): Video capture format (e.g., 'MJPG', 'YUYV')
                    buffer_size (int, optional): Number of frames to buffer
                IP Camera Parameters:
                    username (str, optional): Authentication username
                    password (str, optional): Authentication password
                    timeout (float, optional): Connection timeout in seconds. Default: 10.0
                WebSocket Camera Parameters:
                    host (str, optional): WebSocket server host. Default: "0.0.0.0"
                    port (int, optional): WebSocket server port. Default: 8080
                    timeout (float, optional): Connection timeout in seconds. Default: 10.0
                    frame_format (str, optional): Expected frame format ("base64", "binary", 
                        "json"). Default: "base64"
            
        Returns:
            BaseCamera: Appropriate camera implementation instance
            
        Raises:
            CameraConfigError: If source type is not supported or parameters are invalid
            
        Examples:
            V4L/USB Camera:
            
            ```python
            camera = Camera(0, resolution=(640, 480), fps=30)
            camera = Camera("/dev/video1", fps=15)
            ```
            
            IP Camera:
            
            ```python
            camera = Camera("rtsp://192.168.1.100:554/stream", username="admin", password="secret", timeout=15.0)
            camera = Camera("http://192.168.1.100:8080/video", retry_attempts=5)
            ```
            
            WebSocket Camera:
            
            ```python   
            camera = Camera("ws://0.0.0.0:8080", frame_format="json", max_queue_size=20)
            camera = Camera("ws://192.168.1.100:8080", ping_interval=30)
            ```
        """
        # Dynamic imports to avoid circular dependencies
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # V4L Camera
            from .v4l_camera import V4LCamera
            return V4LCamera(source, **kwargs)
        elif isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme in ['http', 'https', 'rtsp']:
                # IP Camera
                from .ip_camera import IPCamera
                return IPCamera(source, **kwargs)
            elif parsed.scheme in ['ws', 'wss']:
                # WebSocket Camera - extract host and port from URL
                from .websocket_camera import WebSocketCamera
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                return WebSocketCamera(host=host, port=port, **kwargs)
            elif source.startswith('/dev/video') or source.isdigit():
                # V4L device path or index as string
                from .v4l_camera import V4LCamera
                return V4LCamera(source, **kwargs)
            else:
                raise CameraConfigError(f"Unsupported camera source: {source}")
        else:
            raise CameraConfigError(f"Invalid source type: {type(source)}")
