# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import time
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.camera import V4LCamera, CameraOpenError

from conftest import v4l_device_argument  # noqa: F401


@pytest.fixture(autouse=True)
def autouse_v4l_device_argument(v4l_device_argument):
    return v4l_device_argument


@pytest.fixture
def mock_successful_connect() -> MagicMock:
    """Mock successful connection for V4LCamera."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return mock_cap


@pytest.fixture
def mock_failed_connect_open() -> MagicMock:
    """Mock failed connection due to open error for V4LCamera."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap.get.return_value = 640
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return mock_cap


@pytest.fixture
def mock_failed_connect_read() -> MagicMock:
    """Mock failed connection due to test read error for V4LCamera."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640
    mock_cap.read.return_value = (False, None)
    return mock_cap


class TestV4LCameraInitialization:
    def test_initialization_with_all_parameters(self, v4l_device_argument):
        """Test that V4LCamera properly initializes with all V4L-specific parameters."""

        def dummy_adjustment(frame):
            return frame

        # Test initialization without triggering camera operations
        camera = V4LCamera(device=v4l_device_argument, resolution=(1280, 720), fps=25, adjustments=dummy_adjustment)

        # Verify V4L-specific device resolution worked
        assert camera.v4l_path == "/dev/v4l/by-id/usb-Camera-video-index0"

        # Verify BaseCamera parameters are preserved
        assert camera.resolution == (1280, 720)
        assert camera.fps == 25
        assert camera.adjustments == dummy_adjustment

    def test_device_resolution(self, v4l_device_argument):
        """Test that V4LCamera correctly resolves device path identifiers."""
        camera = V4LCamera(device=v4l_device_argument)
        assert camera.v4l_path == "/dev/v4l/by-id/usb-Camera-video-index0"

    def test_device_resolution_failure(self):
        """Test that V4LCamera raises appropriate error for invalid device identifiers."""
        with pytest.raises(CameraOpenError, match="Unrecognized device identifier"):
            V4LCamera(device="invalid")

        with pytest.raises(CameraOpenError, match="out of range"):
            V4LCamera(device=1)


class TestV4LCameraStartStop:
    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_start_success(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera start() calls V4L-specific _open_camera and sets up hardware correctly."""

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            elif prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        mock_successful_connect.get.side_effect = get_caps
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera(resolution=(640, 480), fps=10)

        assert not camera.is_started()

        camera.start()

        assert camera.is_started()
        mock_videocapture.assert_called_once_with("/dev/v4l/by-id/usb-Camera-video-index0")

        # Verify V4L camera setup calls
        assert mock_successful_connect.set.call_count == 4
        set_call_args = [call.args for call in mock_successful_connect.set.call_args_list]
        assert (cv2.CAP_PROP_BUFFERSIZE, 1) in set_call_args
        assert (cv2.CAP_PROP_FRAME_WIDTH, 640) in set_call_args
        assert (cv2.CAP_PROP_FRAME_HEIGHT, 480) in set_call_args
        assert (cv2.CAP_PROP_FPS, 10) in set_call_args

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_start_already_started(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera doesn't reinitialize when already started."""
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()

        # Start camera first time
        camera.start()
        assert camera.is_started()
        assert mock_videocapture.call_count == 1

        # Start camera second time
        camera.start()

        # Should still be started but no additional VideoCapture creation
        assert camera.is_started()
        assert mock_videocapture.call_count == 1  # No additional calls

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_hardware_adaptation_resolution_mismatch(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera adapts when hardware doesn't support requested resolution."""

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 320
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 240
            elif prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        mock_successful_connect.get.side_effect = get_caps
        mock_videocapture.return_value = mock_successful_connect

        # Request 640x480 but hardware only supports 320x240
        camera = V4LCamera(resolution=(640, 480), fps=10)
        camera.start()

        # Should adapt to actual hardware capabilities
        assert camera.resolution == (320, 240)
        assert camera.is_started()

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_hardware_adaptation_fps_mismatch(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera adapts when hardware doesn't support requested FPS."""

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            elif prop == cv2.CAP_PROP_FPS:
                return 15
            return 0

        mock_successful_connect.get.side_effect = get_caps
        mock_videocapture.return_value = mock_successful_connect

        # Request 30fps but hardware only supports 15fps
        camera = V4LCamera(resolution=(640, 480), fps=30)
        camera.start()

        assert camera.fps == 15
        assert camera.is_started()

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_stop_success(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera stop() properly releases V4L resources."""
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()
        camera.start()
        assert camera.is_started()

        camera.stop()

        assert not camera.is_started()
        mock_successful_connect.release.assert_called_once()  # Should release cv2.VideoCapture

    def test_stop_not_started(self):
        """Test that V4LCamera stop() is safe when not started."""
        camera = V4LCamera()
        assert not camera.is_started()

        camera.stop()  # Should not raise any exception

        assert not camera.is_started()

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_is_started(self, mock_videocapture, mock_successful_connect):
        """Test V4LCamera is_started() reflects actual V4L camera state."""
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()

        assert not camera.is_started()

        camera.start()
        assert camera.is_started()

        camera.stop()
        assert not camera.is_started()


class TestV4LCameraRecovery:
    """Test suite for camera disconnection and recovery mechanisms."""

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_initial_connection_with_retry(self, mock_videocapture):
        """Test that initial connection retries on failure."""
        # First two attempts fail, third succeeds
        mock_cap_fail = MagicMock()
        mock_cap_fail.isOpened.return_value = False

        mock_cap_success = MagicMock()
        mock_cap_success.isOpened.return_value = True
        mock_cap_success.get.return_value = 640
        mock_cap_success.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        mock_videocapture.side_effect = [mock_cap_fail, mock_cap_fail, mock_cap_success]

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0
        camera.start()

        assert camera.is_started()
        assert mock_videocapture.call_count == 3

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_start_open_error(self, mock_videocapture, mock_failed_connect_open):
        """Test that V4LCamera raises an exception for open failures."""
        mock_videocapture.return_value = mock_failed_connect_open

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0
        with pytest.raises(CameraOpenError):
            camera.start()

        assert not camera.is_started()
        assert mock_videocapture.call_count == 10

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_start_read_error(self, mock_videocapture, mock_failed_connect_read):
        """Test that V4LCamera raises an exception for read failures."""
        mock_videocapture.return_value = mock_failed_connect_read

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0
        with pytest.raises(CameraOpenError):
            camera.start()

        assert not camera.is_started()
        assert mock_videocapture.call_count == 10

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_no_auto_reconnect_when_disabled(self, mock_videocapture, mock_successful_connect):
        """Test that auto-reconnect doesn't happen when disabled."""
        mock_successful_connect.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during start() succeeds
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First capture succeeds
            (False, None),  # Second capture fails
        ]
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera(auto_reconnect=False)
        camera.start()

        # First read succeeds
        frame1 = camera.capture()
        assert frame1 is not None

        # Second read fails and returns None (no reconnect)
        frame2 = camera.capture()
        assert frame2 is None

        # Verify only initial connection, no reconnection
        assert mock_videocapture.call_count == 1

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_auto_reconnect_on_read_failure(self, mock_videocapture, mock_successful_connect):
        """Test automatic reconnection when frame read fails."""
        # Setup connection, first actual capture() and simulate disconnect on second capture()
        mock_successful_connect.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during start() succeeds
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First capture succeeds
            (False, None),  # Second capture fails
        ]

        # Setup reconnection success
        mock_cap_reconnect = MagicMock()
        mock_cap_reconnect.isOpened.return_value = True
        mock_cap_reconnect.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        mock_videocapture.side_effect = [
            mock_successful_connect,  # Used for initial connection
            mock_cap_reconnect,  # Used for reconnection
        ]

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0
        camera.start()

        # First capture works
        frame1 = camera.capture()
        assert frame1 is not None

        # Second capture fails
        frame2 = camera.capture()
        assert frame2 is None

        # Third capture reconnects and succeeds
        frame3 = camera.capture()
        assert frame3 is not None

        # Verify reconnection happened
        assert mock_videocapture.call_count == 2

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_auto_reconnect_on_read_failure_by_exception(self, mock_videocapture, mock_successful_connect):
        """Test automatic reconnection when frame read fails."""
        # Setup connection, first actual capture() and simulate disconnect on second capture()
        mock_successful_connect.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during start() succeeds
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First capture succeeds
            Exception("Simulated read exception"),  # Second capture fails by exception
        ]

        # Setup reconnection success
        mock_cap_reconnect = MagicMock()
        mock_cap_reconnect.isOpened.return_value = True
        mock_cap_reconnect.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        mock_videocapture.side_effect = [
            mock_successful_connect,  # Used for initial connection
            mock_cap_reconnect,  # Used for reconnection
        ]

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0
        camera.start()

        # First capture works
        frame1 = camera.capture()
        assert frame1 is not None

        # Second capture fails
        frame2 = camera.capture()
        assert frame2 is None

        # Third capture reconnects and succeeds
        frame3 = camera.capture()
        assert frame3 is not None

        # Verify reconnection happened
        assert mock_videocapture.call_count == 2

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_exponential_backoff_on_open(self, mock_videocapture, mock_successful_connect, mock_failed_connect_open, mock_failed_connect_read):
        """Test that exponential backoff is used during camera opening."""
        sleep_calls = []

        def spy_sleep(seconds):
            sleep_calls.append(seconds)

        # Patch time.sleep in the v4l_camera module only for this test
        with patch("arduino.app_peripherals.camera.v4l_camera.time.sleep", side_effect=spy_sleep):
            # Fail, attempt 5 times, succeed at last one
            mock_videocapture.side_effect = [
                mock_failed_connect_open,
                mock_failed_connect_read,
                mock_failed_connect_open,
                mock_failed_connect_read,
                mock_successful_connect,
            ]

            camera = V4LCamera()
            camera.auto_reconnect_delay = 0.1
            camera.start()

        # Check that sleep was called with exponentially increasing delays
        # Attempt 0 fails -> sleep(0.1 * 2^0) = 0.1
        # Attempt 1 fails -> sleep(0.1 * 2^1) = 0.2
        # Attempt 2 fails -> sleep(0.1 * 2^2) = 0.4
        # Attempt 3 fails -> sleep(0.1 * 2^3) = 0.8
        # Attempt 5 succeeds -> no sleep
        assert len(sleep_calls) == 4
        assert sleep_calls[0] == 0.1
        assert sleep_calls[1] == 0.2
        assert sleep_calls[2] == 0.4
        assert sleep_calls[3] == 0.8

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_reconnect_rate_limiting(self, mock_videocapture, mock_successful_connect, mock_failed_connect_open):
        """Test that reconnection attempts are rate-limited."""
        import time

        mock_successful_connect.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during _open_camera() succeeds
            (False, None),  # First capture fails
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during _open_camera() succeeds
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Second capture succeeds
        ]

        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()
        camera.start()
        camera.auto_reconnect_delay = 0.1

        # Fails for disconnection
        frame = camera.capture()
        assert frame is None

        # After waiting, should allow reconnect attempt
        time.sleep(0.15)

        frame = camera.capture()
        assert frame is not None
        assert camera._last_reconnection_attempt > 0

    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    @patch("arduino.app_peripherals.camera.v4l_camera.os.path.exists")
    def test_auto_reconnect_on_device_not_found(self, mock_path_exists, mock_videocapture, mock_successful_connect):
        """Test automatic reconnection when the device path is not found for a period of time."""
        mock_path_exists.side_effect = [
            True,  # For _resolve_stable_path
            True,  # For _resolve_name
            True,  # For _resolve_name
            True,  # For _open_camera
            False,  # For _safe_connect tentative for third capture()
            True,  # For _safe_connect check for fourth capture()
        ]
        mock_successful_connect.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during _safe_connect() for initial connection
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First capture() succeeds
            Exception("Simulated read exception"),  # Second capture() fails by exception
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Read during _safe_connect() for reconnection
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Fourth capture() succeeds
        ]

        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()
        camera.auto_reconnect_delay = 0.1
        camera.start()

        # First capture() succeeds
        frame1 = camera.capture()
        assert frame1 is not None

        # Second capture() fails: simulated disconnection via exception
        frame2 = camera.capture()
        assert frame2 is None
        mock_successful_connect.release.assert_called_once()  # Ensure camera was closed

        # Third capture() fails: device is still disconnected and reconnection attempt fails
        frame3 = camera.capture()
        assert frame3 is None
        assert mock_videocapture.call_count == 1  # Called only during initialization, _safe_connect not called because device not found

        # Fourth capture() succeeds: device reappears, reconnection is triggered and capture() returns a frame
        frame4 = camera.capture()
        assert frame4 is not None
        assert mock_videocapture.call_count == 2  # Initial + after device reappeared


class TestV4LCameraEventCallbacks:
    @patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture")
    def test_events(self, mock_videocapture, mock_successful_connect):
        """Test that V4LCamera emits events on connect and disconnect."""
        mock_videocapture.return_value = mock_successful_connect

        camera = V4LCamera()
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
