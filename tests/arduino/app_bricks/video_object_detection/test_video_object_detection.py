# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.video_objectdetection import VideoObjectDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_COMPOSE = {"services": {"ei-object-detection": {}}}

TIMEOUT = 2.0  # seconds to wait for async handler execution


def _make_classification_msg(bounding_boxes: list) -> str:
    """Return a JSON-encoded WS classification message."""
    return json.dumps({"type": "classification", "result": {"bounding_boxes": bounding_boxes}})


def _make_box(label: str, value: float, x=0, y=0, width=10, height=10) -> dict:
    return {"label": label, "value": value, "x": x, "y": y, "width": width, "height": height}


def _wait(event: threading.Event, timeout: float = TIMEOUT) -> bool:
    """Wait for an event to be set; return True if it fired within the timeout."""
    return event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Patch infrastructure so no real network calls are made."""
    monkeypatch.setattr(
        "arduino.app_bricks.video_objectdetection.load_brick_compose_file",
        lambda cls: FAKE_COMPOSE,
    )
    monkeypatch.setattr(
        "arduino.app_bricks.video_objectdetection.resolve_address",
        lambda host: "127.0.0.1",
    )
    monkeypatch.setattr(
        "arduino.app_bricks.video_objectdetection.Camera",
        lambda: MagicMock(),
    )


@pytest.fixture
def detector():
    """VideoObjectDetection instance with default confidence=0.3."""
    d = VideoObjectDetection(confidence=0.3, debounce_sec=0.0)
    yield d
    d._executor.shutdown(wait=False)


@pytest.fixture
def ws():
    """Fake WebSocket connection (not used in classification path)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


def test_on_detect_rejects_non_function(detector: VideoObjectDetection):
    with pytest.raises(TypeError):
        detector.on_detect("cat", "not_a_function")


def test_on_detect_all_rejects_non_function(detector: VideoObjectDetection):
    with pytest.raises(TypeError):
        detector.on_detect_all(42)


def test_on_detect_overwrites_existing_handler(detector: VideoObjectDetection, ws):
    called = []
    detector.on_detect("cat", lambda: called.append(1))
    detector.on_detect("cat", lambda: called.append(2))  # overwrite

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert called == [2], "Second handler should overwrite the first"


# ---------------------------------------------------------------------------
# _execute_handler — per-label callbacks
# ---------------------------------------------------------------------------


def test_handler_no_params_is_called(detector: VideoObjectDetection, ws):
    """Handler with no parameters is invoked on a matching detection."""
    fired = threading.Event()
    detector.on_detect("cat", lambda: fired.set())

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    detector._process_message(ws, msg)

    assert _wait(fired), "Handler should be invoked within timeout"


def test_handler_with_detection_details_receives_payload(detector: VideoObjectDetection, ws):
    """Handler accepting one positional arg receives the detection_details dict."""
    received = {}

    def handler(details):
        received.update(details)

    detector.on_detect("dog", handler)

    msg = _make_classification_msg([_make_box("dog", 0.8, x=5, y=10, width=20, height=30)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert received["confidence"] == pytest.approx(0.8)
    assert received["bounding_box_xyxy"] == (5, 10, 25, 40)


def test_handler_with_frame_kwarg_receives_none_when_preview_disabled(detector: VideoObjectDetection, ws):
    """When camera_preview=False (default), frame is None even if handler accepts it."""
    received_frame = []

    def handler(details, frame=None):
        received_frame.append(frame)

    detector.on_detect("cat", handler)

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert received_frame == [None]


def test_handler_with_frame_kwarg_receives_bytes_when_preview_enabled(monkeypatch: pytest.MonkeyPatch):
    """When camera_preview=True and a frame is buffered, handler receives jpeg bytes."""
    d = VideoObjectDetection(confidence=0.3, camera_preview=True)

    fake_jpeg = b"\xff\xd8\xff\xe0fake_jpeg_data"
    encoded = "data:image/jpeg;base64," + base64.b64encode(fake_jpeg).decode("utf-8")
    with d._camera_preview_lock:
        d._last_camera_frame = encoded

    received_frame = []

    def handler(details, frame=None):
        received_frame.append(frame)

    d.on_detect("cat", handler)

    ws = MagicMock()
    msg = _make_classification_msg([_make_box("cat", 0.9)])
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert received_frame[0] == fake_jpeg


# ---------------------------------------------------------------------------
# _execute_handler — confidence filtering
# ---------------------------------------------------------------------------


def test_handler_not_called_below_confidence(detector: VideoObjectDetection, ws):
    """Detection below confidence threshold must not trigger any handler."""
    called = threading.Event()
    detector.on_detect("cat", lambda: called.set())

    # 0.2 < default confidence 0.3
    msg = _make_classification_msg([_make_box("cat", 0.2)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert not called.is_set(), "Handler must not fire for low-confidence detection"


def test_handler_called_at_exact_confidence_threshold(detector: VideoObjectDetection, ws):
    """Detection exactly at the confidence threshold must NOT trigger (strict <)."""
    called = threading.Event()
    detector.on_detect("cat", lambda: called.set())

    msg = _make_classification_msg([_make_box("cat", 0.3)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    # confidence 0.3 is not < 0.3, so handler should fire
    assert called.is_set()


def test_unregistered_label_does_not_trigger_handler(detector: VideoObjectDetection, ws):
    """Detection for a label with no registered handler must not crash or fire."""
    called = threading.Event()
    detector.on_detect("cat", lambda: called.set())

    msg = _make_classification_msg([_make_box("dog", 0.9)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert not called.is_set()


# ---------------------------------------------------------------------------
# _execute_handler — debounce
# ---------------------------------------------------------------------------


def test_debounce_suppresses_rapid_repeat(ws):
    """Second detection within the debounce window must not invoke the handler."""
    d = VideoObjectDetection(confidence=0.3, debounce_sec=5.0)
    count = []

    def handler():
        count.append(1)

    d.on_detect("cat", handler)

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    d._process_message(ws, msg)
    d._execute_handler.__func__  # ensure the method exists
    d._executor.shutdown(wait=True)

    # Immediately send again — still within debounce window
    count.clear()
    d._executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert count == [], "Handler must be suppressed within debounce window"


def test_debounce_allows_after_window(ws):
    """Handler must fire again after the debounce window has elapsed."""
    d = VideoObjectDetection(confidence=0.3, debounce_sec=0.05)
    count = []

    def handler():
        count.append(1)

    d.on_detect("cat", handler)
    msg = _make_classification_msg([_make_box("cat", 0.9)])

    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)
    assert len(count) == 1

    time.sleep(0.1)  # let debounce window expire

    d._executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert len(count) == 2, "Handler should fire again after debounce window"


# ---------------------------------------------------------------------------
# _execute_handler — lock / concurrency
# ---------------------------------------------------------------------------


def test_handler_skipped_when_lock_already_held(detector: VideoObjectDetection, ws):
    """If the per-detection lock is already held (handler running), new detections are skipped."""
    barrier = threading.Barrier(2)
    released = threading.Event()
    invocations = []

    def slow_handler():
        invocations.append("start")
        barrier.wait()  # sync with test thread
        released.wait(timeout=TIMEOUT)

    detector.on_detect("cat", slow_handler)

    msg = _make_classification_msg([_make_box("cat", 0.9)])

    # First message — slow_handler starts
    detector._process_message(ws, msg)
    barrier.wait()  # wait until slow_handler is inside

    # Second message — lock is held, should be skipped
    second_fired = threading.Event()

    def quick_handler():
        second_fired.set()

    # Temporarily swap handler to a quick one and manually call _execute_handler
    # The original slow_handler still owns the lock — a second _execute_handler call should skip
    detector._process_message(ws, msg)

    released.set()  # let slow_handler finish
    detector._executor.shutdown(wait=True)

    # Only one invocation should have happened (second was dropped)
    assert invocations.count("start") == 1, "Second detection must be skipped while lock is held"


def test_blocking_handler_discards_detections_sent_while_running(detector: VideoObjectDetection, ws):
    """Detections arriving while the handler is still executing must all be discarded,
    while detections for other labels are processed normally.

    Scenario:
      1. Register a blocking 'cat' handler and a quick 'dog' handler.
      2. Send the first 'cat' detection — blocking handler starts and holds the cat lock.
      3. Send 50 more 'cat' detections (all discarded) interleaved with 'dog' detections.
      4. Unblock the cat handler.
      5. Assert cat handler was called exactly once and dog handler was called
         for every dog detection sent while cat was blocked.
    """
    EXTRA_DETECTIONS = 50
    cat_invocation_count = []
    dog_invocation_count = []
    handler_started = threading.Event()
    handler_unblock = threading.Event()

    def blocking_cat_handler():
        cat_invocation_count.append(1)
        handler_started.set()  # signal that we are inside the handler
        handler_unblock.wait(timeout=30)  # block until the test releases us

    def dog_handler():
        dog_invocation_count.append(1)

    detector.on_detect("cat", blocking_cat_handler)
    detector.on_detect("dog", dog_handler)

    cat_msg = _make_classification_msg([_make_box("cat", 0.9)])
    dog_msg = _make_classification_msg([_make_box("dog", 0.9)])

    # 1st cat detection — starts the blocking handler
    detector._process_message(ws, cat_msg)

    # Wait until the cat handler is actually running and holding the lock
    assert handler_started.wait(timeout=TIMEOUT), "Cat handler did not start in time"

    # Interleave 50 cat detections (all dropped) with dog detections.
    # Per-label locks are independent — dog must never be blocked by cat's lock.
    DOG_DETECTIONS = 3
    for i in range(EXTRA_DETECTIONS):
        detector._process_message(ws, cat_msg)
        if i < DOG_DETECTIONS:
            detector._process_message(ws, dog_msg)

    # Release the blocking cat handler
    handler_unblock.set()
    detector._executor.shutdown(wait=True)

    assert len(cat_invocation_count) == 1, f"Cat handler should have been invoked exactly once, but was called {len(cat_invocation_count)} times"
    assert len(dog_invocation_count) == DOG_DETECTIONS, (
        f"Dog handler should have been invoked {DOG_DETECTIONS} times, but was called {len(dog_invocation_count)} times"
    )


# ---------------------------------------------------------------------------
# ALL_HANDLERS_KEY (on_detect_all)
# ---------------------------------------------------------------------------


def test_global_handler_receives_all_detections(detector: VideoObjectDetection, ws):
    """on_detect_all callback receives a dict with all detections above threshold."""
    received = {}

    def handler(detections):
        received.update(detections)

    detector.on_detect_all(handler)

    msg = _make_classification_msg([
        _make_box("cat", 0.9),
        _make_box("dog", 0.7),
        _make_box("bird", 0.1),  # below threshold — must be filtered
    ])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert "cat" in received
    assert "dog" in received
    assert "bird" not in received


def test_global_handler_not_called_when_no_detections_above_threshold(detector: VideoObjectDetection, ws):
    """on_detect_all must not be called if no detection passes the confidence threshold."""
    called = threading.Event()
    detector.on_detect_all(lambda d: called.set())

    msg = _make_classification_msg([_make_box("cat", 0.1)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert not called.is_set()


def test_global_handler_with_frame_receives_bytes_when_preview_enabled():
    """on_detect_all callback with frame kwarg receives jpeg bytes when preview is enabled."""
    d = VideoObjectDetection(confidence=0.3, camera_preview=True)

    fake_jpeg = b"\xff\xd8\xff\xe0test"
    encoded = "data:image/jpeg;base64," + base64.b64encode(fake_jpeg).decode("utf-8")
    with d._camera_preview_lock:
        d._last_camera_frame = encoded

    received_frame = []

    def handler(detections, frame=None):
        received_frame.append(frame)

    d.on_detect_all(handler)

    ws = MagicMock()
    msg = _make_classification_msg([_make_box("cat", 0.9)])
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert received_frame[0] == fake_jpeg


def test_global_handler_debounce(ws):
    """on_detect_all respects debounce just like per-label handlers."""
    d = VideoObjectDetection(confidence=0.3, debounce_sec=5.0)
    count = []
    d.on_detect_all(lambda dets: count.append(1))

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)
    assert len(count) == 1

    # Second call within debounce window
    from concurrent.futures import ThreadPoolExecutor

    d._executor = ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert len(count) == 1, "Global handler must be debounced"


# ---------------------------------------------------------------------------
# Message type handling
# ---------------------------------------------------------------------------


def test_unknown_message_type_does_not_raise(detector: VideoObjectDetection, ws):
    """Processing an unknown WS message type must not raise an exception."""
    msg = json.dumps({"type": "unknown-type", "data": "something"})
    detector._process_message(ws, msg)  # should not raise


def test_handling_message_success_is_silently_ignored(detector: VideoObjectDetection, ws):
    """handling-message-success messages must be silently ignored."""
    msg = json.dumps({"type": "handling-message-success"})
    detector._process_message(ws, msg)  # should not raise


def test_classification_message_with_empty_bounding_boxes(detector: VideoObjectDetection, ws):
    """Classification message with empty bounding_boxes list must not raise."""
    called = threading.Event()
    detector.on_detect("cat", lambda: called.set())

    msg = json.dumps({"type": "classification", "result": {"bounding_boxes": []}})
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert not called.is_set()


def test_per_label_and_global_handler_both_fire(detector: VideoObjectDetection, ws):
    """Both the per-label and the global handler must fire for the same detection."""
    label_fired = threading.Event()
    global_fired = threading.Event()

    detector.on_detect("cat", lambda: label_fired.set())
    detector.on_detect_all(lambda d: global_fired.set())

    msg = _make_classification_msg([_make_box("cat", 0.9)])
    detector._process_message(ws, msg)
    detector._executor.shutdown(wait=True)

    assert label_fired.is_set(), "Per-label handler must fire"
    assert global_fired.is_set(), "Global handler must fire"
