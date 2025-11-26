# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0


import pytest

from arduino.app_peripherals.camera import Camera, V4LCamera, IPCamera, WebSocketCamera, CameraConfigError

from conftest import v4l_device_argument  # noqa: F401


def test_camera_factory_with_v4l_device(v4l_device_argument):
    """Test Camera factory with multiple device paths (V4L)."""
    camera = Camera(v4l_device_argument)
    assert isinstance(camera, V4LCamera)
    assert camera.v4l_path == "/dev/v4l/by-id/usb-Camera-video-index0"


def test_camera_factory_with_rtsp_url():
    """Test Camera factory with RTSP URL (IP Camera)."""
    camera = Camera("rtsp://192.168.1.100/stream")
    assert isinstance(camera, IPCamera)
    assert camera.url == "rtsp://192.168.1.100/stream"


def test_camera_factory_with_http_url():
    """Test Camera factory with HTTP URL (IP Camera)."""
    camera = Camera("http://192.168.1.100:8080/video")
    assert isinstance(camera, IPCamera)
    assert camera.url == "http://192.168.1.100:8080/video"


def test_camera_factory_with_https_url():
    """Test Camera factory with HTTPS URL (IP Camera)."""
    camera = Camera("https://192.168.1.100:8080/video")
    assert isinstance(camera, IPCamera)
    assert camera.url == "https://192.168.1.100:8080/video"


def test_camera_factory_with_ws_url_default_port():
    """Test Camera factory with WebSocket URL without port."""
    camera = Camera("ws://localhost")
    assert isinstance(camera, WebSocketCamera)
    assert camera.url == "ws://0.0.0.0:8080"
    assert camera.port == 8080  # Default port


def test_camera_factory_with_ws_url():
    """Test Camera factory with WebSocket URL."""
    camera = Camera("ws://0.0.0.0:8080")
    assert isinstance(camera, WebSocketCamera)
    assert camera.url == "ws://0.0.0.0:8080"
    assert camera.port == 8080


def test_camera_factory_with_wss_url():
    """Test Camera factory with secure WebSocket URL."""
    camera = Camera("wss://192.168.1.100:9090")
    assert isinstance(camera, WebSocketCamera)
    assert camera.url == "ws://0.0.0.0:9090"  # IP is always ignored
    assert camera.port == 9090


def test_camera_factory_with_ip_camera_kwargs():
    """Test Camera factory with IP camera specific kwargs."""
    camera = Camera("rtsp://192.168.1.100/stream", username="admin", password="secret", timeout=30)
    assert isinstance(camera, IPCamera)
    assert camera.username == "admin"
    assert camera.password == "secret"
    assert camera.timeout == 30


def test_camera_factory_with_websocket_camera_kwargs():
    """Test Camera factory with WebSocket camera specific kwargs."""
    camera = Camera("ws://0.0.0.0:8080", secret="topsecret", timeout=20)
    assert isinstance(camera, WebSocketCamera)
    assert camera.secret == "topsecret"
    assert camera.timeout == 20


def test_camera_factory_invalid_source_type():
    """Test Camera factory with invalid source type."""
    with pytest.raises(CameraConfigError, match="Invalid source type"):
        Camera({"invalid": "type"})


def test_camera_factory_unsupported_source():
    """Test Camera factory with unsupported source string."""
    with pytest.raises(CameraConfigError, match="Unsupported camera source"):
        Camera("invalid-source")


def test_camera_factory_all_parameters(v4l_device_argument):
    """Test Camera factory with all common parameters."""
    adjustment = lambda x: x * 2

    camera = Camera(source=v4l_device_argument, resolution=(1280, 720), fps=60, adjustments=adjustment)
    assert isinstance(camera, V4LCamera)
    assert camera.resolution == (1280, 720)
    assert camera.fps == 60
    assert camera.adjustments == adjustment


def test_camera_factory_returns_v4l_instance(v4l_device_argument):
    """Test that Camera factory returns V4LCamera instance for V4L sources."""
    camera = Camera(v4l_device_argument)
    assert isinstance(camera, V4LCamera)


def test_camera_factory_returns_ip_instance():
    """Test that Camera factory returns IPCamera instance for IP sources."""
    camera = Camera("rtsp://192.168.1.100/stream")
    assert isinstance(camera, IPCamera)


def test_camera_factory_returns_websocket_instance():
    """Test that Camera factory returns WebSocketCamera instance for WS sources."""
    camera = Camera("ws://0.0.0.0:8080")
    assert isinstance(camera, WebSocketCamera)


def test_camera_factory_rtsp_with_port():
    """Test RTSP URL with custom port."""
    camera = Camera("rtsp://192.168.1.100:554/stream1")
    assert isinstance(camera, IPCamera)
    assert camera.url == "rtsp://192.168.1.100:554/stream1"


def test_camera_factory_http_with_path():
    """Test HTTP URL with path."""
    camera = Camera("http://example.com/cameras/cam1/stream.mjpg")
    assert isinstance(camera, IPCamera)
    assert camera.url == "http://example.com/cameras/cam1/stream.mjpg"
