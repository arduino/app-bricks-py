# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.camera import IPCamera, CameraConfigError, CameraOpenError


@pytest.fixture
def mock_videocapture():
    """Fixture for mocking cv2.VideoCapture."""
    with patch("arduino.app_peripherals.camera.ip_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_vc.return_value = mock_cap
        yield mock_vc, mock_cap


@pytest.fixture
def mock_requests():
    """Fixture for mocking requests."""
    with patch("arduino.app_peripherals.camera.ip_camera.requests") as mock_req:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_req.head.return_value = mock_response
        yield mock_req


def test_ip_camera_init_default():
    """Test IPCamera initialization with default parameters."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    assert camera.url == "rtsp://192.168.1.100/stream"
    assert camera.username is None
    assert camera.password is None
    assert camera.timeout == 10
    assert camera.resolution == (640, 480)
    assert camera.fps == 10


def test_ip_camera_init_with_auth():
    """Test IPCamera initialization with authentication."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream", username="admin", password="secret")
    assert camera.username == "admin"
    assert camera.password == "secret"


def test_ip_camera_init_custom_params():
    """Test IPCamera initialization with custom parameters."""
    camera = IPCamera(url="http://192.168.1.100:8080/video", timeout=30, resolution=(1920, 1080), fps=30)
    assert camera.timeout == 30
    assert camera.resolution == (1920, 1080)
    assert camera.fps == 30


def test_ip_camera_validate_url_rtsp():
    """Test URL validation for RTSP."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    assert camera.url == "rtsp://192.168.1.100/stream"


def test_ip_camera_validate_url_http():
    """Test URL validation for HTTP."""
    camera = IPCamera(url="http://192.168.1.100:8080/video")
    assert camera.url == "http://192.168.1.100:8080/video"


def test_ip_camera_validate_url_https():
    """Test URL validation for HTTPS."""
    camera = IPCamera(url="https://192.168.1.100:8080/video")
    assert camera.url == "https://192.168.1.100:8080/video"


def test_ip_camera_validate_url_invalid_scheme():
    """Test URL validation with invalid scheme."""
    with pytest.raises(CameraConfigError, match="Unsupported URL scheme"):
        IPCamera(url="ftp://192.168.1.100/stream")


def test_ip_camera_validate_url_malformed():
    """Test URL validation with malformed URL."""
    with pytest.raises(CameraConfigError, match="Invalid URL format"):
        IPCamera(url="not a valid url")


def test_ip_camera_build_url_no_auth():
    """Test building URL without authentication."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    url = camera._build_url()
    assert url == "rtsp://192.168.1.100/stream"


def test_ip_camera_build_url_with_auth():
    """Test building URL with authentication."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream", username="admin", password="secret")
    url = camera._build_url()
    assert url == "rtsp://admin:secret@192.168.1.100/stream"


def test_ip_camera_build_url_with_auth_and_port():
    """Test building URL with authentication and port."""
    camera = IPCamera(url="rtsp://192.168.1.100:554/stream", username="admin", password="secret")
    url = camera._build_url()
    assert url == "rtsp://admin:secret@192.168.1.100:554/stream"


def test_ip_camera_build_url_override_existing_auth():
    """Test that provided credentials override URL credentials."""
    camera = IPCamera(url="rtsp://olduser:oldpass@192.168.1.100/stream", username="newuser", password="newpass")
    url = camera._build_url()
    assert url == "rtsp://newuser:newpass@192.168.1.100/stream"


def test_ip_camera_start_rtsp(mock_videocapture, mock_requests):
    """Test starting RTSP camera."""
    mock_vc, _ = mock_videocapture

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.start()

    assert camera.is_started()
    mock_vc.assert_called_once_with("rtsp://192.168.1.100/stream")
    # RTSP should not test HTTP connectivity
    mock_requests.head.assert_not_called()


def test_ip_camera_start_http(mock_videocapture, mock_requests):
    """Test starting HTTP camera."""
    camera = IPCamera(url="http://192.168.1.100:8080/video")
    camera.start()

    assert camera.is_started()
    # HTTP should test HTTP connectivity
    mock_requests.head.assert_called_once()


def test_ip_camera_start_http_with_auth(mock_videocapture, mock_requests):
    """Test starting HTTP camera with authentication."""
    camera = IPCamera(url="http://192.168.1.100:8080/video", username="admin", password="secret")
    camera.start()

    # Should pass auth to requests
    mock_requests.head.assert_called_once()
    call_kwargs = mock_requests.head.call_args[1]
    assert call_kwargs["auth"] == ("admin", "secret")


def test_ip_camera_start_fails_to_open(mock_videocapture, mock_requests):
    """Test error when camera fails to open."""
    _, mock_cap = mock_videocapture
    mock_cap.isOpened.return_value = False

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.auto_reconnect_delay = 0

    with pytest.raises(CameraOpenError, match="Failed to open IP camera"):
        camera.start()

    assert mock_cap.isOpened.call_count == 10


def test_ip_camera_start_fails_to_read_frame(mock_videocapture, mock_requests):
    """Test error when cannot read initial frame."""
    _, mock_cap = mock_videocapture
    mock_cap.read.return_value = (False, None)

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.auto_reconnect_delay = 0

    with pytest.raises(CameraOpenError, match="Read test failed for IP camera at rtsp://192.168.1.100/stream"):
        camera.start()

    assert mock_cap.read.call_count == 10


def test_ip_camera_start_http_connectivity_test_failed(mock_videocapture, mock_requests):
    """Test error when HTTP connectivity test fails."""
    mock_requests.head.return_value.status_code = 404
    mock_requests.RequestException = Exception  # Mock the exception class

    camera = IPCamera(url="http://192.168.1.100:8080/video")
    camera.auto_reconnect_delay = 0

    with pytest.raises(CameraOpenError, match="HTTP camera returned status 404"):
        camera.start()

    assert mock_requests.head.call_count == 10


def test_ip_camera_start_http_connectivity_network_fail(mock_videocapture, mock_requests):
    """Test error when HTTP request raises exception."""

    # Create a real exception to raise
    class MockRequestException(Exception):
        pass

    mock_requests.RequestException = MockRequestException
    mock_requests.head.side_effect = MockRequestException("Network error")

    camera = IPCamera(url="http://192.168.1.100:8080/video")
    camera.auto_reconnect_delay = 0

    with pytest.raises(CameraOpenError, match="Cannot connect to HTTP camera"):
        camera.start()

    assert mock_requests.head.call_count == 10


def test_ip_camera_start_http_connectivity_test_206(mock_videocapture, mock_requests):
    """Test HTTP connectivity with 206 Partial Content response."""
    _, mock_cap = mock_videocapture
    mock_requests.head.return_value.status_code = 206

    camera = IPCamera(url="http://192.168.1.100:8080/video")
    camera.start()

    assert camera.is_started()
    mock_cap.isOpened.assert_called_once()


def test_ip_camera_stop(mock_videocapture, mock_requests):
    """Test stopping IP camera."""
    _, mock_cap = mock_videocapture

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.start()
    camera.stop()

    assert not camera.is_started()
    mock_cap.release.assert_called_once()


def test_ip_camera_read_frame(mock_videocapture, mock_requests):
    """Test reading a frame from IP camera."""
    _, mock_cap = mock_videocapture
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    mock_cap.read.return_value = (True, test_frame)

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.start()

    frame = camera.capture()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert np.array_equal(frame, test_frame)


def test_ip_camera_read_frame_auto_reconnect(mock_videocapture, mock_requests):
    """Test automatic reconnection when reading frame."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    # Don't start the camera, let _read_frame trigger reconnection
    camera._is_started = True

    frame = camera.capture()

    # Should attempt to reconnect
    assert frame is not None


def test_ip_camera_read_frame_reconnect_failure(mock_videocapture, mock_requests):
    """Test when reconnection fails during frame read."""
    _, mock_cap = mock_videocapture
    mock_cap.isOpened.return_value = False
    mock_cap.isOpened.assert_called = True

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera._is_started = True

    frame = camera._read_frame()

    # Should return None when reconnection fails
    assert frame is None


def test_ip_camera_read_frame_connection_dropped(mock_videocapture, mock_requests):
    """Test when connection drops during frame read."""
    _, mock_cap = mock_videocapture

    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    camera.start()

    # Simulate connection drop
    mock_cap.read.return_value = (False, None)
    mock_cap.isOpened.return_value = False

    frame = camera._read_frame()

    # Should return None and close connection
    assert frame is None
    assert camera._cap is None


def test_ip_camera_timeout_custom(mock_videocapture, mock_requests):
    """Test IP camera with custom timeout."""
    camera = IPCamera(url="http://192.168.1.100:8080/video", timeout=30)
    camera.start()

    call_kwargs = mock_requests.head.call_args[1]
    assert call_kwargs["timeout"] == 30


def test_events(mock_videocapture, mock_requests):
    """Test that IPCamera emits events on connect and disconnect."""
    camera = IPCamera(url="rtsp://192.168.1.100/stream")
    events = []

    def event_callback(event, data):
        events.append((event, data))

    camera.on_status_changed(event_callback)

    camera.start()
    camera.stop()

    # The events list is modified from another thread, so a brief sleep
    # helps ensure the main thread sees the appended items before asserting.
    time.sleep(0.1)

    assert len(events) == 2
    assert "connected" in events[0][0]
    assert "disconnected" in events[1][0]
