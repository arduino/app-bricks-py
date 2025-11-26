# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import time
import numpy as np
import tempfile
import cv2

from arduino.app_peripherals.camera import BaseCamera, CameraTransformError
from arduino.app_peripherals.usb_camera import CameraReadError
from arduino.app_utils.image.pipeable import PipeableFunction


class MockedCamera(BaseCamera):
    """Concrete implementation of BaseCamera for testing."""

    def __init__(self, *args, **kwargs):
        # Extract test configuration
        self.should_fail_open = kwargs.pop("should_fail_open", False)
        self.should_fail_close = kwargs.pop("should_fail_close", False)
        self.should_fail_read = kwargs.pop("should_fail_read", False)
        self.open_error_message = kwargs.pop("open_error_message", "Camera open failed")
        self.close_error_message = kwargs.pop("close_error_message", "Camera close failed")
        self.read_error_message = kwargs.pop("read_error_message", "Frame read failed")
        self.frame = kwargs.pop("frame", np.zeros((480, 640, 3), dtype=np.uint8))

        super().__init__(*args, **kwargs)

        # Track method calls for verification
        self.open_call_count = 0
        self.close_call_count = 0
        self.read_call_count = 0

    def _open_camera(self):
        """Mock implementation of _open_camera."""
        self.open_call_count += 1
        if self.should_fail_open:
            raise RuntimeError(self.open_error_message)
        else:
            self._set_status("connected")

    def _close_camera(self):
        """Mock implementation of _close_camera."""
        self.close_call_count += 1
        if self.should_fail_close:
            raise RuntimeError(self.close_error_message)
        else:
            self._set_status("disconnected")

    def _read_frame(self):
        """Mock implementation that returns a dummy frame."""
        self.read_call_count += 1
        if self.should_fail_read:
            raise RuntimeError(self.read_error_message)
        if not self._is_started:
            return None
        return self.frame


def test_base_camera_init_default():
    """Test BaseCamera initialization with default parameters."""
    camera = MockedCamera()
    assert camera.resolution == (640, 480)
    assert camera.fps == 10
    assert camera.adjustments is None
    assert camera.auto_reconnect
    assert not camera.is_started()


def test_base_camera_init_custom():
    """Test BaseCamera initialization with custom parameters."""
    adj_func = lambda x: x
    camera = MockedCamera(resolution=(1920, 1080), fps=30, adjustments=adj_func, auto_reconnect=False)
    assert camera.resolution == (1920, 1080)
    assert camera.fps == 30
    assert camera.adjustments == adj_func
    assert not camera.auto_reconnect
    assert not camera.is_started()


def test_base_camera_init_invalid():
    """Test BaseCamera initialization with invalid parameters."""
    with pytest.raises(ValueError):
        MockedCamera(fps=0)


def test_is_started_state_transitions():
    """Test is_started return value through different state transitions."""
    camera = MockedCamera()

    assert not camera.is_started()
    camera.start()
    assert camera.is_started()
    camera.stop()
    assert not camera.is_started()

    camera.start()
    assert camera.is_started()
    camera.stop()
    assert not camera.is_started()


def test_start_success():
    """Test that start() calls _open_camera and updates state correctly."""
    camera = MockedCamera()

    assert not camera.is_started()
    assert camera.open_call_count == 0

    camera.start()

    assert camera.is_started()
    assert camera.open_call_count == 1


def test_start_already_started():
    """Test that start() doesn't call _open_camera again when already started."""
    camera = MockedCamera()

    camera.start()
    assert camera.open_call_count == 1
    assert camera.is_started()

    camera.start()
    # Should not call _open_camera again
    assert camera.open_call_count == 1
    assert camera.is_started()


def test_start_error_reporting():
    """Test that errors from _open_camera are reported clearly."""
    camera = MockedCamera(should_fail_open=True, open_error_message="Mock camera failure", auto_reconnect=False)

    # Verify error from _open_camera is propagated as-is
    with pytest.raises(RuntimeError, match="Mock camera failure"):
        camera.start()

    # Verify camera state remains stopped on error
    assert not camera.is_started()
    assert camera.open_call_count == 1


def test_stop_success():
    """Test that stop() calls _close_camera and updates state correctly."""
    camera = MockedCamera()
    camera.start()
    assert camera.is_started()

    camera.stop()

    # Verify _close_camera was called and state updated
    assert not camera.is_started()
    assert camera.close_call_count == 1


def test_stop_not_started():
    """Test that stop() doesn't call _close_camera when not started."""
    camera = MockedCamera()
    assert not camera.is_started()

    camera.stop()

    # Should not call _close_camera
    assert camera.close_call_count == 0
    assert not camera.is_started()


def test_stop_error_reporting():
    """Test that errors from _close_camera are handled gracefully."""
    camera = MockedCamera(should_fail_close=True, close_error_message="Mock close failure")

    camera.start()
    assert camera.is_started()

    # stop() does not raise exceptions
    camera.stop()

    assert camera.close_call_count == 1
    assert camera.is_started()  # Should still be started due to close error


def test_capture_when_started():
    """Test that capture() calls _read_frame when started."""
    camera = MockedCamera()
    camera.start()

    initial_read_count = camera.read_call_count
    frame = camera.capture()

    assert camera.read_call_count == initial_read_count + 1
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)


def test_capture_when_stopped():
    """Test that capture() returns an exception when camera is not started."""
    camera = MockedCamera()

    with pytest.raises(CameraReadError):
        camera.capture()

    assert camera.read_call_count == 0


def test_capture_read_frame_error_reporting():
    """Test that errors from _read_frame are not caught by capture()."""
    camera = MockedCamera(should_fail_read=True, read_error_message="Mock read failure")
    camera.start()

    # Verify error from _read_frame is propagated as-is
    with pytest.raises(RuntimeError, match="Mock read failure"):
        camera.capture()

    assert camera.read_call_count == 1


def test_capture_with_adjustments():
    """Test that adjustments are applied correctly to captured frames."""

    def adjustment(frame):
        return frame + 10

    camera = MockedCamera(adjustments=adjustment)
    camera.start()

    frame = camera.capture()

    assert camera.read_call_count == 1
    assert frame is not None
    assert np.all(frame == 10)


def test_capture_adjustment_error_reporting():
    """Test that adjustment errors are reported clearly."""

    def bad_adjustment(frame):
        raise ValueError("Adjustment failed")

    camera = MockedCamera(adjustments=bad_adjustment)
    camera.start()

    with pytest.raises(CameraTransformError, match="Frame transformation failed"):
        camera.capture()

    assert camera.read_call_count == 1


def test_capture_rate_limiting():
    """Test that FPS throttling/rate limiting is applied correctly."""
    camera = MockedCamera(fps=10)  # 0.1 seconds between frames
    camera.start()

    start_time = time.monotonic()
    frame1 = camera.capture()
    frame2 = camera.capture()
    elapsed = time.monotonic() - start_time

    # Should take ~0.1 seconds due to throttling
    assert elapsed >= 0.09
    assert frame1 is not None
    assert frame2 is not None
    assert camera.read_call_count == 2


def test_stream_can_be_stopped_by_user_code():
    """Test that stream() can be stopped by user code breaking out of the loop."""
    camera = MockedCamera()
    camera.start()

    n_frames = 0
    for i, frame in enumerate(camera.stream()):
        n_frames += 1
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        if i >= 4:  # Get 5 frames then break
            break

    assert n_frames == 5
    assert camera.is_started()  # Camera should still be running


def test_stream_stops_when_camera_stopped():
    """Test that stream() stops automatically when camera is stopped."""
    camera = MockedCamera()
    camera.start()

    frames = []
    for i, frame in enumerate(camera.stream()):
        frames.append(frame)
        if i >= 4:  # Get 5 frames then call stop()
            camera.stop()

    assert len(frames) == 5
    assert not camera.is_started()


def test_stream_exception_propagation():
    """Test that exceptions from capture() are correctly propagated outside the consuming loop."""

    def bad_adjustment(frame):
        raise ValueError("Stream adjustment failed")

    camera = MockedCamera(adjustments=bad_adjustment)
    camera.start()

    # Exception should propagate out of the stream loop
    with pytest.raises(CameraTransformError, match="Frame transformation failed"):
        for _ in camera.stream():
            pass


def test_context_manager_calls_start_and_stop():
    """Test that context manager calls start() and stop() when entering and exiting."""
    camera = MockedCamera()

    assert not camera.is_started()
    assert camera.open_call_count == 0
    assert camera.close_call_count == 0

    with camera as ctx_camera:
        assert camera.is_started()
        assert camera.open_call_count == 1
        assert camera.close_call_count == 0
        assert ctx_camera is camera  # Should return self

        frame = camera.capture()
        assert frame is not None

    assert not camera.is_started()
    assert camera.open_call_count == 1
    assert camera.close_call_count == 1


def test_context_manager_with_exception():
    """Test that context manager calls stop() even when exception occurs."""
    camera = MockedCamera()

    try:
        with camera:
            assert camera.is_started()
            assert camera.open_call_count == 1

            raise RuntimeError("Test exception")
    except RuntimeError:
        pass

    # Verify stop() was called despite exception
    assert not camera.is_started()
    assert camera.close_call_count == 1


def test_capture_multiple_adjustments():
    """Test that adjustment pipelines are applied correctly."""

    def adj1(frame):
        return frame + 5

    adjustment1 = PipeableFunction(adj1)

    def adj2(frame):
        return frame * 2

    adjustment2 = PipeableFunction(adj2)

    camera = MockedCamera(adjustments=adjustment1 | adjustment2)
    camera.start()

    frame = camera.capture()

    # Verify _read_frame was called and adjustments applied: (0 + 5) * 2 = 10
    assert camera.read_call_count == 1
    assert np.all(frame == 10)


def test_events():
    camera = MockedCamera(fps=5)
    events = []

    def event_callback(event, data):
        events.append((event, data))

    camera.on_status_changed(event_callback)

    camera.start()  # Should emit "connected" event

    assert camera.status == "connected"

    camera.capture()  # Should emit "streaming" event

    assert camera.status == "streaming"

    camera.frame = None
    camera.capture()
    camera.capture()
    camera.capture()  # Should emit "paused" event
    camera.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert camera.status == "paused"

    camera.capture()  # Should emit "streaming" event

    assert camera.status == "streaming"

    camera.stop()  # Should emit "disconnected" event

    assert camera.status == "disconnected"

    # The events list is modified from another thread, so a brief sleep
    # helps ensure the main thread sees the appended items before asserting.
    time.sleep(0.1)

    assert len(events) == 5
    assert "connected" in events[0][0]
    assert "streaming" in events[1][0]
    assert "paused" in events[2][0]
    assert "streaming" in events[3][0]
    assert "disconnected" in events[4][0]


def test_record_zero_duration():
    camera = MockedCamera()
    camera.start()
    with pytest.raises(ValueError):
        camera.record(0)
    camera.stop()


def test_record():
    camera = MockedCamera(fps=5)
    camera.frame = np.ones((480, 640, 3), dtype=np.uint8)
    camera.start()

    duration = 1.0
    expected_frames = int(camera.fps * duration)
    frames = camera.record(duration)

    assert isinstance(frames, np.ndarray)
    assert frames.shape[0] == expected_frames
    assert frames.shape[1:] == camera.frame.shape
    assert frames.dtype == camera.frame.dtype
    assert np.all(frames == camera.frame)

    camera.stop()


def test_record_avi():
    camera = MockedCamera(fps=5)
    camera.frame = np.ones((480, 640, 3), dtype=np.uint8)
    camera.start()

    duration = 1.0
    expected_frames = int(camera.fps * duration)
    avi_bytes = camera.record_avi(duration)

    assert isinstance(avi_bytes, np.ndarray)
    assert avi_bytes.dtype == np.uint8
    assert avi_bytes.size > 0

    with tempfile.NamedTemporaryFile(suffix=".avi") as tmp:
        tmp.write(avi_bytes.tobytes())
        tmp.flush()

        read_count = 0
        cap = cv2.VideoCapture(tmp.name)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            assert frame is not None
            assert frame.dtype == np.uint8
            read_count += 1

        cap.release()

    assert read_count == expected_frames

    camera.stop()


def test_record_avi_uint8_conversion():
    camera = MockedCamera(fps=5)
    # Use float32 frame, should be converted to uint8 in AVI
    camera.frame = np.ones((10, 10, 3), dtype=np.float64)
    camera.start()

    duration = 1.0
    expected_frames = int(camera.fps * duration)
    avi_bytes = camera.record_avi(duration)

    assert isinstance(avi_bytes, np.ndarray)
    assert avi_bytes.dtype == np.uint8
    assert avi_bytes.size > 0

    with tempfile.NamedTemporaryFile(suffix=".avi") as tmp:
        tmp.write(avi_bytes.tobytes())
        tmp.flush()

        read_count = 0
        cap = cv2.VideoCapture(tmp.name)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            assert frame is not None
            assert frame.dtype == np.uint8
            read_count += 1

        cap.release()

    assert read_count == expected_frames

    camera.stop()
