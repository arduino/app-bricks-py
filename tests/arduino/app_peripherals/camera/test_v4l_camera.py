# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import cv2
from unittest.mock import MagicMock, patch

from arduino.app_peripherals.camera import V4LCamera, CameraOpenError, CameraReadError


def test_initialization_with_all_parameters():
    """Test that V4LCamera properly initializes with all V4L-specific parameters."""

    def dummy_adjustment(frame):
        return frame

    # Test initialization without triggering camera operations
    camera = V4LCamera(device="/dev/video1", resolution=(1280, 720), fps=25, adjustments=dummy_adjustment)

    # Verify V4L-specific device resolution worked
    assert camera.device == 1  # Should extract 1 from "/dev/video1"

    # Verify BaseCamera parameters are preserved
    assert camera.resolution == (1280, 720)
    assert camera.fps == 25
    assert camera.adjustments == dummy_adjustment


def test_device_resolution_integer():
    """Test that V4LCamera correctly resolves integer device identifiers."""
    camera = V4LCamera(device=0)
    assert camera.device == 0

    camera = V4LCamera(device=1)
    assert camera.device == 1


def test_device_resolution_string_numeric():
    """Test that V4LCamera correctly resolves numeric string device identifiers."""
    # Test with device mapping available
    with patch.object(V4LCamera, "_get_video_devices_by_index", return_value={1: "2"}):
        camera = V4LCamera(device="1")
        assert camera.device == 2

    # Test with no device mapping (fallback)
    with patch.object(V4LCamera, "_get_video_devices_by_index", return_value={}):
        camera = V4LCamera(device="3")
        assert camera.device == 3


def test_device_resolution_path():
    """Test that V4LCamera correctly resolves device path identifiers."""
    camera = V4LCamera(device="/dev/video0")
    assert camera.device == 0

    camera = V4LCamera(device="/dev/video2")
    assert camera.device == 2


def test_device_resolution_invalid():
    """Test that V4LCamera raises appropriate error for invalid device identifiers."""
    with pytest.raises(CameraOpenError, match="Cannot resolve camera identifier: invalid"):
        V4LCamera(device="invalid")

    with pytest.raises(CameraOpenError, match="Cannot resolve camera identifier: not_a_device"):
        V4LCamera(device="not_a_device")


def test_device_mapping():
    """Test that V4LCamera uses device mapping when available for string indices."""
    device_mapping = {0: "1", 1: "3", 2: "0"}

    with patch.object(V4LCamera, "_get_video_devices_by_index", return_value=device_mapping):
        camera = V4LCamera(device="1")
        assert camera.device == 3

        camera = V4LCamera(device="0")
        assert camera.device == 1


def test_device_mapping_fallback():
    """Test that V4LCamera falls back to direct conversion when no mapping available."""
    with patch.object(V4LCamera, "_get_video_devices_by_index", return_value={}):
        # When no mapping is available, should convert string directly to int
        camera = V4LCamera(device="5")
        assert camera.device == 5


def test_hardware_adaptation_resolution_mismatch():
    """Test that V4LCamera adapts when hardware doesn't support requested resolution."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 320
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 240
            elif prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        mock_cap.get.side_effect = get_caps
        mock_vc.return_value = mock_cap

        # Request 640x480 but hardware only supports 320x240
        camera = V4LCamera(device=0, resolution=(640, 480), fps=10)
        camera.start()

        # Should adapt to actual hardware capabilities
        assert camera.resolution == (320, 240)
        assert camera.is_started()


def test_hardware_adaptation_fps_mismatch():
    """Test that V4LCamera adapts when hardware doesn't support requested FPS."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            elif prop == cv2.CAP_PROP_FPS:
                return 15
            return 0

        mock_cap.get.side_effect = get_caps
        mock_vc.return_value = mock_cap

        # Request 30fps but hardware only supports 15fps
        camera = V4LCamera(device=0, resolution=(640, 480), fps=30)
        camera.start()

        assert camera.fps == 15
        assert camera.is_started()


def test_read_frame_error_message():
    """Test that V4LCamera provides specific error messages for read failures."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=3)
        camera.start()

        with pytest.raises(CameraReadError, match="Failed to read from V4L camera 3"):
            camera.capture()


def test_start_success():
    """Test that V4LCamera start() calls V4L-specific _open_camera and sets up hardware correctly."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        def get_caps(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            elif prop == cv2.CAP_PROP_FPS:
                return 10
            return 0

        mock_cap.get.side_effect = get_caps
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=2, resolution=(640, 480), fps=10)

        assert not camera.is_started()

        camera.start()

        assert camera.is_started()
        mock_vc.assert_called_once_with(2)  # Should open correct device

        # Verify V4L camera setup calls
        set_call_args = [call.args for call in mock_cap.set.call_args_list]

        # Check that buffer size was set to 1
        assert (cv2.CAP_PROP_BUFFERSIZE, 1) in set_call_args

        # Check that resolution was set to 640x480
        assert (cv2.CAP_PROP_FRAME_WIDTH, 640) in set_call_args
        assert (cv2.CAP_PROP_FRAME_HEIGHT, 480) in set_call_args

        # Check that FPS was set to 10
        assert (cv2.CAP_PROP_FPS, 10) in set_call_args


def test_start_already_started():
    """Test that V4LCamera doesn't reinitialize when already started."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=0)

        # Start camera first time
        camera.start()
        assert camera.is_started()
        assert mock_vc.call_count == 1

        # Start camera second time
        camera.start()

        # Should still be started but no additional VideoCapture creation
        assert camera.is_started()
        assert mock_vc.call_count == 1  # No additional calls


def test_start_camera_fails_to_open():
    """Test V4LCamera start() error handling when cv2.VideoCapture fails to open."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False  # Camera fails to open
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=5)

        with pytest.raises(CameraOpenError, match="Failed to open V4L camera 5"):
            camera.start()

        assert not camera.is_started()


def test_stop_success():
    """Test that V4LCamera stop() properly releases V4L resources."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=1)
        camera.start()
        assert camera.is_started()

        camera.stop()

        assert not camera.is_started()
        mock_cap.release.assert_called_once()  # Should release cv2.VideoCapture


def test_stop_not_started():
    """Test that V4LCamera stop() is safe when not started."""
    camera = V4LCamera(device=0)
    assert not camera.is_started()

    camera.stop()  # Should not raise any exception
    assert not camera.is_started()


def test_is_started():
    """Test V4LCamera is_started() reflects actual V4L camera state."""
    with patch("arduino.app_peripherals.camera.v4l_camera.cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_vc.return_value = mock_cap

        camera = V4LCamera(device=0)

        assert not camera.is_started()

        camera.start()
        assert camera.is_started()

        camera.stop()
        assert not camera.is_started()
