# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.pose_estimation import KEYPOINT_NAMES, Keypoint, Person, PoseEstimation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WAIT_TIMEOUT = 2.0  # seconds - maximum time to wait for a callback to complete


def _pose_dict(score: float = 0.8, x: int = 100, y: int = 50) -> dict:
    return {
        "score": score,
        "keypoints": [{"name": name, "x": x + i, "y": y + i, "score": 0.9} for i, name in enumerate(KEYPOINT_NAMES)],
        "bounding_box_xyxy": [x, y, x + 50, y + 150],
    }


def _detection_with(poses: list[dict]) -> dict:
    return {"poses": poses}


def _wait(event: threading.Event, msg: str = ""):
    assert event.wait(timeout=WAIT_TIMEOUT), f"Timed out waiting for: {msg}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pe(monkeypatch: pytest.MonkeyPatch):
    """Return a PoseEstimation instance with infrastructure mocked out."""
    yield from _make_instance(monkeypatch)


@pytest.fixture()
def pe_debounced(monkeypatch: pytest.MonkeyPatch):
    """Return a PoseEstimation instance with a 0.3s debounce."""
    yield from _make_instance(monkeypatch, debounce_sec=0.3)


def _make_instance(monkeypatch: pytest.MonkeyPatch, **kwargs):
    fake_compose = {"services": {"pose-runner": {}}}
    monkeypatch.setattr(
        "arduino.app_bricks.pose_estimation.pose_estimation.load_brick_compose_file",
        lambda cls: fake_compose,
    )
    monkeypatch.setattr(
        "arduino.app_bricks.pose_estimation.pose_estimation.resolve_address",
        lambda host: "127.0.0.1",
    )

    camera = MagicMock()
    instance = PoseEstimation(camera=camera, **kwargs)

    # Provide a real executor so callbacks actually run in threads
    instance._executor = ThreadPoolExecutor(max_workers=4)
    instance._is_running = True

    yield instance

    instance._executor.shutdown(wait=True)
    instance._executor = None


# ---------------------------------------------------------------------------
# Detection parsing tests
# ---------------------------------------------------------------------------


class TestDetectionParsing:
    def test_person_round_trip(self, pe: PoseEstimation):
        received = []
        done = threading.Event()

        def on_kps(person):
            received.append(person)
            done.set()

        pe.on_keypoints(on_kps)

        pe._process_detection(_detection_with([_pose_dict(x=10, y=20)]))

        _wait(done, "keypoints callback")
        person = received[0]
        assert isinstance(person, Person)
        assert person.bounding_box_xyxy == (10, 20, 60, 170)
        assert list(person.keypoints) == list(KEYPOINT_NAMES)
        assert all(isinstance(kp, Keypoint) for kp in person.keypoints.values())
        nose = person.keypoints["nose"]
        assert (nose.x, nose.y) == (10, 20)
        assert person.keypoints["left_wrist"].name == "left_wrist"

    def test_missing_fields_are_tolerated(self, pe: PoseEstimation):
        received = []
        done = threading.Event()

        def on_kps(person):
            received.append(person)
            done.set()

        pe.on_keypoints(on_kps)

        pe._process_detection({"poses": [{"score": 0.9}]})

        _wait(done, "keypoints callback")
        assert received[0].keypoints == {}
        assert received[0].bounding_box_xyxy == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Enter / Exit / Count callback tests
# ---------------------------------------------------------------------------


class TestPresenceCallbacks:
    def test_enter_called_when_person_appears(self, pe: PoseEstimation):
        called = threading.Event()
        pe.on_enter(lambda: called.set())

        pe._process_detection(_detection_with([_pose_dict()]))

        _wait(called, "enter callback")

    def test_exit_called_when_person_leaves(self, pe: PoseEstimation):
        called = threading.Event()
        pe.on_exit(lambda: called.set())

        pe._process_detection(_detection_with([_pose_dict()]))
        pe._process_detection(_detection_with([]))

        _wait(called, "exit callback")

    def test_count_change_receives_new_count(self, pe: PoseEstimation):
        counts = []
        done = threading.Event()

        def on_count(count: int):
            counts.append(count)
            if len(counts) == 2:
                done.set()

        pe.on_count_change(on_count)

        pe._process_detection(_detection_with([_pose_dict()]))
        time.sleep(0.1)  # Let the first callback complete to avoid the busy-discard
        pe._process_detection(_detection_with([_pose_dict(), _pose_dict(x=300)]))

        _wait(done, "count change callbacks")
        assert counts == [1, 2]

    def test_low_confidence_poses_are_filtered(self, pe: PoseEstimation):
        entered = threading.Event()
        got_keypoints = threading.Event()
        pe.on_enter(lambda: entered.set())
        pe.on_keypoints(lambda person: got_keypoints.set())

        pe._process_detection(_detection_with([_pose_dict(score=0.1)]))

        assert not entered.wait(timeout=0.3)
        assert not got_keypoints.is_set()

    def test_presence_flicker_is_debounced(self, pe_debounced: PoseEstimation):
        exited = threading.Event()
        pe_debounced.on_exit(lambda: exited.set())

        # Person appears: the initial transition fires immediately
        pe_debounced._process_detection(_detection_with([_pose_dict()]))
        # Flicker: person disappears right away, within the debounce window
        pe_debounced._process_detection(_detection_with([]))

        assert not exited.wait(timeout=0.1), "exit should have been debounced"

        # After the debounce window the change is accepted
        time.sleep(0.3)
        pe_debounced._process_detection(_detection_with([]))
        _wait(exited, "debounced exit callback")


# ---------------------------------------------------------------------------
# Keypoint stream callback tests
# ---------------------------------------------------------------------------


class TestKeypointCallbacks:
    def test_called_once_per_person(self, pe: PoseEstimation):
        received = []
        done = threading.Event()

        def on_kps(person):
            received.append(person)
            if len(received) == 2:
                done.set()

        pe.on_keypoints(on_kps)

        pe._process_detection(_detection_with([_pose_dict(x=10), _pose_dict(x=300)]))

        _wait(done, "per-person keypoints callbacks")
        assert [person.keypoints["nose"].x for person in received] == [10, 300]

    def test_same_frame_people_are_not_discarded(self, pe: PoseEstimation):
        # People of one frame are delivered within a single dispatch: a slow
        # callback must not cause other people of the SAME frame to be dropped.
        calls = []

        def slow_callback(person):
            calls.append(person.keypoints["nose"].x)
            time.sleep(0.2)

        pe.on_keypoints(slow_callback)

        pe._process_detection(_detection_with([_pose_dict(x=10), _pose_dict(x=300)]))

        time.sleep(0.8)
        assert calls == [10, 300]

    def test_busy_callback_discards_new_frames(self, pe: PoseEstimation):
        release = threading.Event()
        calls = []

        def slow_callback(person):
            calls.append(person)
            release.wait(timeout=WAIT_TIMEOUT)

        pe.on_keypoints(slow_callback)

        pe._process_detection(_detection_with([_pose_dict()]))
        time.sleep(0.1)  # Let the first callback start and hold the lock
        pe._process_detection(_detection_with([_pose_dict()]))
        release.set()

        time.sleep(0.2)
        assert len(calls) == 1

    def test_unregister_callback(self, pe: PoseEstimation):
        called = threading.Event()
        pe.on_keypoints(lambda person: called.set())
        pe.on_keypoints(None)

        pe._process_detection(_detection_with([_pose_dict()]))

        assert not called.wait(timeout=0.3)


# ---------------------------------------------------------------------------
# on_pose stub tests
# ---------------------------------------------------------------------------


class TestOnPoseStub:
    def test_on_pose_raises_not_implemented(self, pe: PoseEstimation):
        with pytest.raises(NotImplementedError):
            pe.on_pose("arms_up", lambda pose: None)
