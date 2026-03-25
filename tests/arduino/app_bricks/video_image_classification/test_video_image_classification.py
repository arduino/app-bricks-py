# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.video_imageclassification import VideoImageClassification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_COMPOSE = {"services": {"ei-image-classification": {}}}

TIMEOUT = 2.0  # seconds to wait for async handler execution


def _make_classification_msg(classifications: dict) -> str:
    """Return a JSON-encoded WS classification message."""
    return json.dumps({"type": "classification", "result": {"classification": classifications}})


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
        "arduino.app_bricks.video_imageclassification.load_brick_compose_file",
        lambda cls: FAKE_COMPOSE,
    )
    monkeypatch.setattr(
        "arduino.app_bricks.video_imageclassification.resolve_address",
        lambda host: "127.0.0.1",
    )
    monkeypatch.setattr(
        "arduino.app_bricks.video_imageclassification.Camera",
        lambda: MagicMock(),
    )


@pytest.fixture
def classifier():
    """VideoImageClassification instance with default confidence=0.3."""
    d = VideoImageClassification(confidence=0.3, debounce_sec=0.0)
    yield d
    d._executor.shutdown(wait=False)


@pytest.fixture
def ws():
    """Fake WebSocket connection (not used in classification path)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


def test_on_detect_rejects_non_function(classifier: VideoImageClassification):
    with pytest.raises(TypeError):
        classifier.on_detect("cat", "not_a_function")


def test_on_detect_rejects_callback_with_params(classifier: VideoImageClassification):
    with pytest.raises(ValueError):
        classifier.on_detect("cat", lambda x: x)


def test_on_detect_all_rejects_non_function(classifier: VideoImageClassification):
    with pytest.raises(TypeError):
        classifier.on_detect_all(42)


def test_on_detect_all_rejects_wrong_arg_count(classifier: VideoImageClassification):
    with pytest.raises(ValueError):
        classifier.on_detect_all(lambda: None)  # 0 args — must be exactly 1

    with pytest.raises(ValueError):
        classifier.on_detect_all(lambda a, b: None)  # 2 args — must be exactly 1


def test_on_detect_overwrites_existing_handler(classifier: VideoImageClassification, ws):
    called = []
    classifier.on_detect("cat", lambda: called.append(1))
    classifier.on_detect("cat", lambda: called.append(2))  # overwrite

    msg = _make_classification_msg({"cat": 0.9})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert called == [2], "Second handler should overwrite the first"


# ---------------------------------------------------------------------------
# _execute_handler — per-label callbacks
# ---------------------------------------------------------------------------


def test_handler_no_params_is_called(classifier: VideoImageClassification, ws):
    """Handler with no parameters is invoked on a matching classification."""
    fired = threading.Event()
    classifier.on_detect("cat", lambda: fired.set())

    msg = _make_classification_msg({"cat": 0.9})
    classifier._process_message(ws, msg)

    assert _wait(fired), "Handler should be invoked within timeout"


def test_unregistered_label_does_not_trigger_handler(classifier: VideoImageClassification, ws):
    """Classification for a label with no registered handler must not crash or fire."""
    called = threading.Event()
    classifier.on_detect("cat", lambda: called.set())

    msg = _make_classification_msg({"dog": 0.9})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert not called.is_set()


# ---------------------------------------------------------------------------
# _execute_handler — confidence filtering
# ---------------------------------------------------------------------------


def test_handler_not_called_below_confidence(classifier: VideoImageClassification, ws):
    """Classification below confidence threshold must not trigger any handler."""
    called = threading.Event()
    classifier.on_detect("cat", lambda: called.set())

    # 0.2 < default confidence 0.3
    msg = _make_classification_msg({"cat": 0.2})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert not called.is_set(), "Handler must not fire for low-confidence classification"


def test_handler_called_at_confidence_threshold(classifier: VideoImageClassification, ws):
    """Classification exactly at the threshold fires (value is not < threshold)."""
    called = threading.Event()
    classifier.on_detect("cat", lambda: called.set())

    msg = _make_classification_msg({"cat": 0.3})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert called.is_set()


# ---------------------------------------------------------------------------
# _execute_handler — debounce
# ---------------------------------------------------------------------------


def test_debounce_suppresses_rapid_repeat(ws):
    """Second classification within the debounce window must not invoke the handler."""
    d = VideoImageClassification(confidence=0.3, debounce_sec=5.0)
    count = []

    d.on_detect("cat", lambda: count.append(1))

    msg = _make_classification_msg({"cat": 0.9})
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    # Immediately send again — still within debounce window
    count.clear()
    d._executor = ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert count == [], "Handler must be suppressed within debounce window"


def test_debounce_allows_after_window(ws):
    """Handler must fire again after the debounce window has elapsed."""
    d = VideoImageClassification(confidence=0.3, debounce_sec=0.05)
    count = []

    d.on_detect("cat", lambda: count.append(1))
    msg = _make_classification_msg({"cat": 0.9})

    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)
    assert len(count) == 1

    time.sleep(0.1)  # let debounce window expire

    d._executor = ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert len(count) == 2, "Handler should fire again after debounce window"


# ---------------------------------------------------------------------------
# _execute_handler — lock / concurrency
# ---------------------------------------------------------------------------


def test_handler_skipped_when_lock_already_held(classifier: VideoImageClassification, ws, monkeypatch: pytest.MonkeyPatch):
    """If the per-classification lock is already held, new classifications are skipped."""
    monkeypatch.setattr(VideoImageClassification, "_DETECTION_LOCK_TO", 0.001)

    barrier = threading.Barrier(2)
    released = threading.Event()
    invocations = []

    def slow_handler():
        invocations.append("start")
        barrier.wait()  # sync with test thread
        released.wait(timeout=TIMEOUT)

    classifier.on_detect("cat", slow_handler)

    msg = _make_classification_msg({"cat": 0.9})

    # First message — slow_handler starts
    classifier._process_message(ws, msg)
    barrier.wait()  # wait until slow_handler is inside

    # Second message — lock is held, should be skipped
    classifier._process_message(ws, msg)

    released.set()  # let slow_handler finish
    classifier._executor.shutdown(wait=True)

    assert invocations.count("start") == 1, "Second classification must be skipped while lock is held"


def test_blocking_handler_discards_classifications_sent_while_running(classifier: VideoImageClassification, ws, monkeypatch: pytest.MonkeyPatch):
    """Classifications arriving while the handler is still executing must all be discarded,
    while classifications for other labels are processed normally.

    Scenario:
      1. Register a blocking 'cat' handler and a quick 'dog' handler.
      2. Send the first 'cat' classification — blocking handler starts and holds the cat lock.
      3. Send 50 more 'cat' classifications (all discarded) interleaved with 'dog' classifications.
      4. Unblock the cat handler.
      5. Assert cat handler was called exactly once and dog handler was called
         for every dog classification sent while cat was blocked.
    """
    EXTRA_CLASSIFICATIONS = 50
    # Make failed lock-acquire attempts return in ~1 ms to keep the test fast.
    monkeypatch.setattr(VideoImageClassification, "_DETECTION_LOCK_TO", 0.001)

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

    classifier.on_detect("cat", blocking_cat_handler)
    classifier.on_detect("dog", dog_handler)

    cat_msg = _make_classification_msg({"cat": 0.9})
    dog_msg = _make_classification_msg({"dog": 0.9})

    # 1st cat classification — starts the blocking handler
    classifier._process_message(ws, cat_msg)

    # Wait until the cat handler is actually running and holding the lock
    assert handler_started.wait(timeout=TIMEOUT), "Cat handler did not start in time"

    # Interleave 50 cat classifications (all dropped) with dog classifications.
    # Per-label locks are independent — dog must never be blocked by cat's lock.
    DOG_CLASSIFICATIONS = 3
    for i in range(EXTRA_CLASSIFICATIONS):
        classifier._process_message(ws, cat_msg)
        if i < DOG_CLASSIFICATIONS:
            classifier._process_message(ws, dog_msg)

    # Release the blocking cat handler
    handler_unblock.set()
    classifier._executor.shutdown(wait=True)

    assert len(cat_invocation_count) == 1, f"Cat handler should have been invoked exactly once, but was called {len(cat_invocation_count)} times"
    assert len(dog_invocation_count) == DOG_CLASSIFICATIONS, (
        f"Dog handler should have been invoked {DOG_CLASSIFICATIONS} times, but was called {len(dog_invocation_count)} times"
    )


# ---------------------------------------------------------------------------
# ALL_HANDLERS_KEY (on_detect_all)
# ---------------------------------------------------------------------------


def test_global_handler_receives_all_classifications(classifier: VideoImageClassification, ws):
    """on_detect_all callback receives a dict of all classifications above threshold."""
    received = {}

    def handler(detections):
        received.update(detections)

    classifier.on_detect_all(handler)

    msg = _make_classification_msg({"cat": 0.9, "dog": 0.7, "bird": 0.1})  # bird below threshold
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert "cat" in received
    assert "dog" in received
    assert "bird" not in received


def test_global_handler_not_called_when_nothing_above_threshold(classifier: VideoImageClassification, ws):
    """on_detect_all must not be called if no classification passes the confidence threshold."""
    called = threading.Event()
    classifier.on_detect_all(lambda d: called.set())

    msg = _make_classification_msg({"cat": 0.1})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert not called.is_set()


def test_global_handler_debounce(ws):
    """on_detect_all respects debounce just like per-label handlers."""
    d = VideoImageClassification(confidence=0.3, debounce_sec=5.0)
    count = []
    d.on_detect_all(lambda dets: count.append(1))

    msg = _make_classification_msg({"cat": 0.9})
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)
    assert len(count) == 1

    # Second call within debounce window — must be suppressed
    d._executor = ThreadPoolExecutor(max_workers=5)
    d._process_message(ws, msg)
    d._executor.shutdown(wait=True)

    assert len(count) == 1, "Global handler must be debounced"


def test_per_label_and_global_handler_both_fire(classifier: VideoImageClassification, ws):
    """Both the per-label and the global handler must fire for the same classification."""
    label_fired = threading.Event()
    global_fired = threading.Event()

    classifier.on_detect("cat", lambda: label_fired.set())
    classifier.on_detect_all(lambda d: global_fired.set())

    msg = _make_classification_msg({"cat": 0.9})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert label_fired.is_set(), "Per-label handler must fire"
    assert global_fired.is_set(), "Global handler must fire"


# ---------------------------------------------------------------------------
# Message type handling
# ---------------------------------------------------------------------------


def test_unknown_message_type_does_not_raise(classifier: VideoImageClassification, ws):
    """Processing an unknown WS message type must not raise an exception."""
    msg = json.dumps({"type": "unknown-type", "data": "something"})
    classifier._process_message(ws, msg)  # should not raise


def test_handling_message_success_is_silently_ignored(classifier: VideoImageClassification, ws):
    """handling-message-success messages must be silently ignored."""
    msg = json.dumps({"type": "handling-message-success"})
    classifier._process_message(ws, msg)  # should not raise


def test_classification_message_with_empty_classifications(classifier: VideoImageClassification, ws):
    """Classification message with empty classification dict must not trigger any handler."""
    called = threading.Event()
    classifier.on_detect("cat", lambda: called.set())

    msg = json.dumps({"type": "classification", "result": {"classification": {}}})
    classifier._process_message(ws, msg)
    classifier._executor.shutdown(wait=True)

    assert not called.is_set()
