# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.gesture_recognition import GestureRecognition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAND_RIGHT = {"hand": "right", "gesture": "open", "confidence": 0.9, "landmarks": [], "bounding_box_xyxy": [0, 0, 10, 10]}
_HAND_LEFT = {"hand": "left", "gesture": "open", "confidence": 0.9, "landmarks": [], "bounding_box_xyxy": [0, 0, 10, 10]}
_META_RIGHT = {"hands": [_HAND_RIGHT]}
_META_LEFT = {"hands": [_HAND_LEFT]}
_META_NONE = {"hands": []}

WAIT_TIMEOUT = 2.0  # seconds – maximum time to wait for a callback to complete


def _detection_with(hands: list[dict]) -> dict:
    return {"hands": hands}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gr(monkeypatch: pytest.MonkeyPatch) -> GestureRecognition:
    """Return a GestureRecognition instance with infrastructure mocked out."""
    fake_compose = {"services": {"gesture-runner": {}}}
    monkeypatch.setattr(
        "arduino.app_bricks.gesture_recognition.load_brick_compose_file",
        lambda cls: fake_compose,
    )
    monkeypatch.setattr(
        "arduino.app_bricks.gesture_recognition.resolve_address",
        lambda host: "127.0.0.1",
    )

    camera = MagicMock()
    instance = GestureRecognition(camera=camera)

    # Provide a real executor so callbacks actually run in threads
    instance._executor = ThreadPoolExecutor(max_workers=4)
    instance._is_running = True

    yield instance

    instance._executor.shutdown(wait=True)
    instance._executor = None


# ---------------------------------------------------------------------------
# Utility: wait for a threading.Event with a clear error on timeout
# ---------------------------------------------------------------------------


def _wait(event: threading.Event, msg: str = ""):
    assert event.wait(timeout=WAIT_TIMEOUT), f"Timed out waiting for: {msg}"


# ---------------------------------------------------------------------------
# Enter / Exit callback tests
# ---------------------------------------------------------------------------


class TestEnterExitCallbacks:
    def test_enter_called_when_hands_appear(self, gr: GestureRecognition):
        called = threading.Event()
        gr.on_enter(lambda: called.set())

        gr._had_hands = False
        gr._process_detection(_META_RIGHT)

        _wait(called, "on_enter callback")

    def test_exit_called_when_hands_disappear(self, gr: GestureRecognition):
        called = threading.Event()
        gr.on_exit(lambda: called.set())

        gr._had_hands = True
        gr._process_detection(_META_NONE)

        _wait(called, "on_exit callback")

    def test_enter_not_called_when_hands_already_visible(self, gr: GestureRecognition):
        call_count = [0]
        done = threading.Event()

        def cb():
            call_count[0] += 1
            done.set()

        gr.on_enter(cb)
        gr._had_hands = True  # hands already present
        gr._process_detection(_META_RIGHT)

        # Give the executor a moment – nothing should be submitted
        time.sleep(0.05)
        assert call_count[0] == 0

    def test_exit_not_called_when_no_hands_already(self, gr: GestureRecognition):
        call_count = [0]
        gr.on_exit(lambda: call_count.__setitem__(0, call_count[0] + 1))

        gr._had_hands = False  # no hands to exit from
        gr._process_detection(_META_NONE)

        time.sleep(0.05)
        assert call_count[0] == 0

    def test_enter_callback_receives_no_args(self, gr: GestureRecognition):
        """Enter callback must be callable with zero arguments."""
        called = threading.Event()

        def cb():
            called.set()

        gr.on_enter(cb)
        gr._had_hands = False
        gr._process_detection(_META_RIGHT)

        _wait(called, "on_enter zero-arg callback")

    def test_unregister_enter_callback(self, gr: GestureRecognition):
        call_count = [0]
        gr.on_enter(lambda: call_count.__setitem__(0, call_count[0] + 1))
        gr.on_enter(None)  # unregister

        gr._had_hands = False
        gr._process_detection(_META_RIGHT)

        time.sleep(0.05)
        assert call_count[0] == 0


# ---------------------------------------------------------------------------
# Gesture callback tests
# ---------------------------------------------------------------------------


class TestGestureCallbacks:
    def test_gesture_callback_called_with_metadata(self, gr: GestureRecognition):
        received = [None]
        done = threading.Event()

        def cb(metadata):
            received[0] = metadata
            done.set()

        gr.on_gesture("open", cb)
        gr._had_hands = True
        gr._process_detection(_META_RIGHT)

        _wait(done, "gesture callback")
        assert received[0] is not None
        assert "hands" in received[0]

    def test_gesture_both_wildcard_fires_for_right(self, gr: GestureRecognition):
        called = threading.Event()
        gr.on_gesture("open", lambda m: called.set(), hand="both")

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)

        _wait(called, "both-wildcard right")

    def test_gesture_both_wildcard_fires_for_left(self, gr: GestureRecognition):
        called = threading.Event()
        gr.on_gesture("open", lambda m: called.set(), hand="both")

        gr._had_hands = True
        gr._process_detection(_META_LEFT)

        _wait(called, "both-wildcard left")

    def test_gesture_exact_hand_right_fires_for_right_only(self, gr: GestureRecognition):
        right_called = threading.Event()
        left_called = threading.Event()

        gr.on_gesture("open", lambda m: right_called.set(), hand="right")
        gr.on_gesture("open", lambda m: left_called.set(), hand="left")

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)

        _wait(right_called, "right callback on right hand")
        time.sleep(0.05)
        assert not left_called.is_set(), "left callback should not fire for right hand"

    def test_gesture_exact_hand_left_fires_for_left_only(self, gr: GestureRecognition):
        right_called = threading.Event()
        left_called = threading.Event()

        gr.on_gesture("open", lambda m: right_called.set(), hand="right")
        gr.on_gesture("open", lambda m: left_called.set(), hand="left")

        gr._had_hands = True
        gr._process_detection(_META_LEFT)

        _wait(left_called, "left callback on left hand")
        time.sleep(0.05)
        assert not right_called.is_set(), "right callback should not fire for left hand"

    def test_both_exact_and_wildcard_fire_together(self, gr: GestureRecognition):
        exact_called = threading.Event()
        both_called = threading.Event()

        gr.on_gesture("open", lambda m: exact_called.set(), hand="right")
        gr.on_gesture("open", lambda m: both_called.set(), hand="both")

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)

        _wait(exact_called, "exact callback")
        _wait(both_called, "both callback")

    def test_unregistered_gesture_callback_not_called(self, gr: GestureRecognition):
        call_count = [0]
        gr.on_gesture("open", lambda m: call_count.__setitem__(0, call_count[0] + 1))
        gr.on_gesture("open", None)  # unregister

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)

        time.sleep(0.05)
        assert call_count[0] == 0

    def test_invalid_hand_raises(self, gr: GestureRecognition):
        with pytest.raises(ValueError):
            gr.on_gesture("open", lambda m: None, hand="invalid")

    def test_unknown_gesture_not_called(self, gr: GestureRecognition):
        called = threading.Event()
        gr.on_gesture("closed", lambda m: called.set())

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)  # gesture is "open", not "closed"

        time.sleep(0.05)
        assert not called.is_set()


# ---------------------------------------------------------------------------
# Discard-while-running tests
# ---------------------------------------------------------------------------


class TestDiscardWhileRunning:
    def _slow_callback_factory(self, call_count: list, started: threading.Event, allow_finish: threading.Event):
        """Returns a callback that signals when it starts and blocks until released."""

        def cb(*_args):
            call_count[0] += 1
            started.set()
            allow_finish.wait(timeout=WAIT_TIMEOUT)

        return cb

    def test_enter_discards_while_running(self, gr: GestureRecognition):
        call_count = [0]
        started = threading.Event()
        allow_finish = threading.Event()

        gr.on_enter(self._slow_callback_factory(call_count, started, allow_finish))

        # First dispatch – starts the slow callback
        gr._had_hands = False
        gr._process_detection(_META_RIGHT)
        _wait(started, "first enter callback to start")

        # 49 more dispatches – all should be discarded since lock is held
        for _ in range(49):
            gr._had_hands = False
            gr._process_detection(_META_RIGHT)

        allow_finish.set()
        gr._executor.shutdown(wait=True)

        assert call_count[0] == 1, f"Expected 1 invocation, got {call_count[0]}"

    def test_exit_discards_while_running(self, gr: GestureRecognition):
        call_count = [0]
        started = threading.Event()
        allow_finish = threading.Event()

        gr.on_exit(self._slow_callback_factory(call_count, started, allow_finish))

        gr._had_hands = True
        gr._process_detection(_META_NONE)
        _wait(started, "first exit callback to start")

        for _ in range(49):
            gr._had_hands = True
            gr._process_detection(_META_NONE)

        allow_finish.set()
        gr._executor.shutdown(wait=True)

        assert call_count[0] == 1, f"Expected 1 invocation, got {call_count[0]}"

    def test_gesture_discards_while_running(self, gr: GestureRecognition):
        call_count = [0]
        started = threading.Event()
        allow_finish = threading.Event()

        gr.on_gesture("open", self._slow_callback_factory(call_count, started, allow_finish), hand="right")

        gr._had_hands = True
        gr._process_detection(_META_RIGHT)
        _wait(started, "first gesture callback to start")

        for _ in range(49):
            gr._process_detection(_META_RIGHT)

        allow_finish.set()
        gr._executor.shutdown(wait=True)

        assert call_count[0] == 1, f"Expected 1 invocation, got {call_count[0]}"

    def test_independent_gesture_keys_do_not_block_each_other(self, gr: GestureRecognition):
        """A slow 'open' (right) callback must not prevent 'closed' (right) from running."""
        open_count = [0]
        closed_count = [0]
        open_started = threading.Event()
        open_allow = threading.Event()
        closed_done = threading.Event()

        def slow_open(m):
            open_count[0] += 1
            open_started.set()
            open_allow.wait(timeout=WAIT_TIMEOUT)

        def fast_closed(m):
            closed_count[0] += 1
            closed_done.set()

        gr.on_gesture("open", slow_open, hand="right")
        gr.on_gesture("closed", fast_closed, hand="right")

        # Trigger open – starts slow callback
        gr._had_hands = True
        gr._process_detection(_detection_with([{**_HAND_RIGHT, "gesture": "open"}]))
        _wait(open_started, "open callback to start")

        # Trigger closed – should NOT be blocked by open's lock
        gr._process_detection(_detection_with([{**_HAND_RIGHT, "gesture": "closed"}]))
        _wait(closed_done, "closed callback to complete independently")

        open_allow.set()
        gr._executor.shutdown(wait=True)

        assert open_count[0] == 1
        assert closed_count[0] == 1


# ---------------------------------------------------------------------------
# Exception-safety tests
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    def test_callback_exception_releases_lock(self, gr: GestureRecognition):
        """A failing callback must release its lock so subsequent events are not blocked."""
        call_count = [0]
        second_done = threading.Event()

        def failing_cb():
            raise RuntimeError("intentional error")

        def second_cb():
            call_count[0] += 1
            second_done.set()

        gr.on_enter(failing_cb)

        # First call – callback raises
        gr._had_hands = False
        gr._process_detection(_META_RIGHT)
        time.sleep(0.1)  # let executor finish the failing callback

        # Re-register a working callback and trigger again
        gr.on_enter(second_cb)
        gr._had_hands = False
        gr._process_detection(_META_RIGHT)

        _wait(second_done, "second enter callback after exception")
        assert call_count[0] == 1
