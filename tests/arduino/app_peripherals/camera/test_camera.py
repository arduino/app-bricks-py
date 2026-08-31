# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0


import pytest

from arduino.app_peripherals.camera import Camera, CSICamera, V4LCamera, IPCamera, WebSocketCamera, CameraConfigError, CameraOpenError

from conftest import two_csi_cameras_only, two_v4l_cameras, usb_camera_with_metadata_node, v4l_device_argument  # noqa: F401


def test_camera_factory_with_v4l_device(v4l_device_argument):
    """Test Camera factory with multiple device paths (V4L)."""
    print(f">>>>>>: Testing with V4L device argument: {v4l_device_argument}")
    camera = Camera(v4l_device_argument)
    assert isinstance(camera, V4LCamera)
    assert camera.v4l_path == "/dev/v4l/by-id/usb-Camera-video-index0"


def test_auto_selection_assigns_distinct_cameras(two_v4l_cameras):
    """Auto-selected cameras must not contend for the same device."""
    cam1 = Camera()
    cam2 = Camera()
    assert cam1.v4l_path == "/dev/v4l/by-id/usb-CamA-video-index0"
    assert cam2.v4l_path == "/dev/v4l/by-id/usb-CamB-video-index0"


def test_auto_selection_raises_when_all_cameras_are_in_use(two_v4l_cameras):
    """Auto-selection never reuses a camera assigned to another instance."""
    cam1, cam2 = Camera(), Camera()
    assert cam1.v4l_path != cam2.v4l_path

    with pytest.raises(CameraOpenError):
        Camera()


def test_auto_selection_releases_camera_when_instance_is_dropped(two_v4l_cameras):
    """A camera claimed by a dropped instance becomes available again."""
    import gc

    cam = Camera()
    first_path = cam.v4l_path
    del cam
    gc.collect()

    assert Camera().v4l_path == first_path


def test_auto_selection_skips_explicitly_selected_cameras(two_v4l_cameras):
    """Auto-selection routes around cameras claimed by explicit selections."""
    explicit = Camera(0)
    auto = Camera()
    assert explicit.v4l_path == "/dev/v4l/by-id/usb-CamA-video-index0"
    assert auto.v4l_path == "/dev/v4l/by-id/usb-CamB-video-index0"


def test_explicit_selection_reuses_a_camera_already_in_use(two_v4l_cameras):
    """Explicit selection tolerates reuse: contention only surfaces at start()."""
    cam1 = Camera(0)
    cam2 = Camera(0)
    assert cam2.v4l_path == cam1.v4l_path


def test_non_capture_nodes_are_not_listed_as_cameras(usb_camera_with_metadata_node):
    """A UVC metadata node must not be enumerated as a camera."""
    assert V4LCamera.list_devices() == [10]


def test_auto_selection_never_selects_non_capture_nodes(usb_camera_with_metadata_node):
    """A claimed camera leaves only its metadata node behind: not a selectable camera."""
    cam1 = Camera()
    assert cam1.v4l_path == "/dev/v4l/by-id/usb-Cam-video-index0"

    with pytest.raises(CameraOpenError):
        Camera()


def test_explicit_index_selects_nth_plugged_camera(two_v4l_cameras):
    """An explicit index counts the plugged cameras, in use or not."""
    auto = Camera()
    assert auto.v4l_path == "/dev/v4l/by-id/usb-CamA-video-index0"

    cam2 = Camera(1)
    assert cam2.v4l_path == "/dev/v4l/by-id/usb-CamB-video-index0"


def test_csi_source_ignores_unsupported_kwargs(two_csi_cameras_only):
    """A kwarg CSICamera does not support must not break its construction."""
    camera = Camera("csi:0", resolution=(1280, 960), fps=30, codec="MJPG", bogus=1)
    assert isinstance(camera, CSICamera)
    assert not hasattr(camera, "codec")
    assert not hasattr(camera, "bogus")


def test_csi_source_keeps_shared_kwargs(two_csi_cameras_only):
    """Kwargs supported by both camera types are still forwarded to CSI cameras."""
    camera = Camera("csi:0", auto_reconnect=False, codec="MJPG")
    assert isinstance(camera, CSICamera)
    assert camera.auto_reconnect is False


def test_auto_selected_csi_camera_ignores_v4l_only_kwargs(two_csi_cameras_only):
    """Auto-selection falling back to CSI must tolerate V4L-only kwargs."""
    camera = Camera(resolution=(1280, 960), fps=30, codec="MJPG")
    assert isinstance(camera, CSICamera)
    assert camera.csi_path == "CAMERA0"


def test_v4l_source_still_rejects_unknown_kwargs(v4l_device_argument):
    """Filtering is CSI-specific: a USB camera still rejects unknown kwargs."""
    with pytest.raises(TypeError):
        Camera(v4l_device_argument, bogus=1)


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
