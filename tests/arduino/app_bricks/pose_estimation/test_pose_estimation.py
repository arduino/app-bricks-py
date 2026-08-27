# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.pose_estimation import KEYPOINT_NAMES, POSE_NAMES, Keypoint, Person, PoseEstimation

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
    return {"persons": poses}


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
# Constructor tests
# ---------------------------------------------------------------------------


def test_stop_resets_the_temporal_state_with_the_asset_thresholds(pe):
    pe._pose_ema.update({"sitting": 1.0}, dt=1.0)
    pe.stop()
    pe._executor = ThreadPoolExecutor(max_workers=1)  # the fixture teardown shuts one down

    assert pe._pose_ema.smoothed["sitting"] == 0.0
    assert pe._pose_ema.enter_threshold == pe._pose_thresholds["enter"]
    assert pe._pose_ema.exit_threshold == pe._pose_thresholds["exit"]


# ---------------------------------------------------------------------------
# Detection parsing tests
# ---------------------------------------------------------------------------


def test_pose_names_lists_the_asset_poses():
    assert POSE_NAMES == ("left_arm_raised", "right_arm_raised", "sitting", "standing")


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

        pe._process_detection({"persons": [{"score": 0.9}]})

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
# on_pose event tests
# ---------------------------------------------------------------------------


def _skeleton_keypoints(score: float = 0.9) -> dict:
    """A plausible upright person: distinct shoulder/hip heights, sided joints."""
    height = {"nose": 20, "eye": 18, "ear": 22, "shoulder": 60, "elbow": 100, "wrist": 140, "hip": 140, "knee": 210, "ankle": 280}
    keypoints = {}
    for name in KEYPOINT_NAMES:
        side = -20 if name.startswith("left") else 20 if name.startswith("right") else 0
        keypoints[name] = Keypoint(name=name, x=100 + side, y=height[name.split("_")[-1]], score=score)
    return keypoints


def _person(bbox=(60, 0, 140, 300), score: float = 0.9) -> Person:
    return Person(keypoints=_skeleton_keypoints(score), bounding_box_xyxy=bbox)


class TestPoseEvents:
    def _feed(self, pe: PoseEstimation, people: list[Person], steps: int, start: float) -> float:
        now = start
        for _ in range(steps):
            now += 0.1
            pe._update_pose_classification(people, now)
        return now

    def test_out_of_frame_joints_make_the_frame_unreadable(self, pe: PoseEstimation):
        pe._frame_hw = (480, 640)
        assert pe._classify_person(_person()) is not None

        far_out = _person()
        for name in ("left_ankle", "right_ankle"):
            kp = far_out.keypoints[name]
            far_out.keypoints[name] = Keypoint(name=name, x=kp.x, y=kp.y + 400, score=kp.score)
        assert pe._classify_person(far_out) is None

        pe._frame_hw = None
        assert pe._classify_person(far_out) is not None

    def test_a_tip_extrapolated_outside_from_a_seen_joint_is_still_readable(self, pe: PoseEstimation):
        pe._frame_hw = (480, 640)
        sitting_close = _person()
        for name in ("left_ankle", "right_ankle"):
            kp = sitting_close.keypoints[name]
            sitting_close.keypoints[name] = Keypoint(name=name, x=kp.x, y=560, score=0.01)
        assert pe._classify_person(sitting_close) is not None

    def test_uncertain_joints_inside_the_image_still_classify(self, pe: PoseEstimation):
        pe._frame_hw = (480, 640)
        guessed_legs = _person()
        for name in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
            kp = guessed_legs.keypoints[name]
            guessed_legs.keypoints[name] = Keypoint(name=name, x=kp.x, y=kp.y, score=0.03)
        assert pe._classify_person(guessed_legs) is not None

    def test_every_gate_outcome_is_counted(self, pe: PoseEstimation):
        pe._frame_hw = (480, 640)
        assert pe._classify_person(_person()) is not None
        assert pe._classify_person(_person(score=0.01)) is None
        assert pe._gate_counts["classified"] == 1
        assert pe._gate_counts["anchors"] == 1
        assert pe._gate_last[0] == "anchors"

    def test_enter_then_exit_fire_with_stable_classifications(self, pe: PoseEstimation, monkeypatch):
        events = []
        got_enter, got_exit = threading.Event(), threading.Event()

        def on_sitting(pose):
            events.append(pose)
            (got_enter if pose.event == "enter" else got_exit).set()

        pe.on_pose("sitting", on_sitting)
        person = _person()

        monkeypatch.setattr(pe, "_classify_person", lambda p: {"sitting": 1.0})
        now = self._feed(pe, [person], steps=20, start=0.0)
        _wait(got_enter, "pose enter")
        time.sleep(0.1)  # let the enter callback finish so exit is not busy-discarded

        monkeypatch.setattr(pe, "_classify_person", lambda p: {"sitting": 0.0})
        self._feed(pe, [person], steps=20, start=now)
        _wait(got_exit, "pose exit")

        assert [pose.event for pose in events] == ["enter", "exit"]
        enter = events[0]
        assert enter.name == "sitting"
        assert 0.0 < enter.confidence <= 1.0
        assert enter.bounding_box_xyxy == person.bounding_box_xyxy
        assert enter.keypoints is person.keypoints

    def test_the_largest_person_is_the_tracked_subject(self, pe: PoseEstimation, monkeypatch):
        seen = []
        monkeypatch.setattr(pe, "_classify_person", lambda p: seen.append(p))
        small = Person(keypoints={}, bounding_box_xyxy=(0, 0, 50, 50))
        big = Person(keypoints={}, bounding_box_xyxy=(200, 0, 400, 300))

        pe._update_pose_classification([small, big], now=1.0)

        assert seen == [big]

    def test_a_fully_guessed_torso_makes_the_frame_unreadable(self, pe: PoseEstimation):
        assert pe._classify_person(_person(score=0.01)) is None

    def test_one_observed_anchor_is_enough_to_classify(self, pe: PoseEstimation):
        person = _person(score=0.01)
        kp = person.keypoints["left_shoulder"]
        person.keypoints["left_shoulder"] = Keypoint(name="left_shoulder", x=kp.x, y=kp.y, score=0.5)
        assert pe._classify_person(person) is not None

    def test_a_readable_skeleton_yields_a_probability_dict(self, pe: PoseEstimation):
        probs = pe._classify_person(_person())
        assert isinstance(probs, dict)
        assert set(probs) == set(pe._pose_knn.classes)

    def test_unknown_pose_name_raises(self, pe: PoseEstimation):
        with pytest.raises(ValueError, match="unknown pose"):
            pe.on_pose("jumping", lambda pose: None)

    def test_unregister_stops_events(self, pe: PoseEstimation, monkeypatch):
        called = threading.Event()
        pe.on_pose("standing", lambda pose: called.set())
        pe.on_pose("standing", None)

        monkeypatch.setattr(pe, "_classify_person", lambda p: {"standing": 1.0})
        self._feed(pe, [_person()], steps=20, start=0.0)

        assert not called.wait(timeout=0.3)


class TestSetConfidence:
    def test_updates_the_filter_and_validates_input(self, pe: PoseEstimation):
        pe.set_confidence(0.8)
        assert pe._confidence == 0.8

        with pytest.raises(ValueError):
            pe.set_confidence(1.5)
        with pytest.raises(ValueError):
            pe.set_confidence(-0.1)
        with pytest.raises(ValueError):
            pe.set_confidence("high")
        with pytest.raises(ValueError):
            pe.set_confidence(True)
        assert pe._confidence == 0.8


class TestSetDrawBboxes:
    def test_updates_the_flag_and_validates_input(self, pe: PoseEstimation):
        assert pe._draw_bboxes is False

        pe.set_draw_bboxes(True)
        assert pe._draw_bboxes is True

        with pytest.raises(ValueError):
            pe.set_draw_bboxes(1)
        with pytest.raises(ValueError):
            pe.set_draw_bboxes("on")
        with pytest.raises(ValueError):
            pe.set_draw_bboxes(None)
        assert pe._draw_bboxes is True


class TestSetDrawUncertain:
    def test_updates_the_flag_and_validates_input(self, pe: PoseEstimation):
        assert pe._draw_uncertain is True

        pe.set_draw_uncertain(False)
        assert pe._draw_uncertain is False

        with pytest.raises(ValueError):
            pe.set_draw_uncertain(0)
        with pytest.raises(ValueError):
            pe.set_draw_uncertain("off")
        assert pe._draw_uncertain is False


class TestSetBboxPadding:
    def test_replaces_the_padding_and_validates_input(self, pe: PoseEstimation):
        assert pe._bbox_padding == (0.0, 0.0, 0.0, 0.0)

        pe.set_bbox_padding((0.15, 0.20, 0.15, 0.20))  # (top, right, bottom, left), come CSS
        assert pe._bbox_padding == (0.15, 0.20, 0.15, 0.20)

        pe.set_bbox_padding(0.1)  # scalare: tutti i lati
        assert pe._bbox_padding == (0.1, 0.1, 0.1, 0.1)

        with pytest.raises(ValueError):
            pe.set_bbox_padding(1.5)
        with pytest.raises(ValueError):
            pe.set_bbox_padding((0.1, 0.2))
        with pytest.raises(ValueError):
            pe.set_bbox_padding((0.1, 0.2, 0.3, -0.1))
        with pytest.raises(ValueError):
            pe.set_bbox_padding("high")
        with pytest.raises(ValueError):
            pe.set_bbox_padding(True)
        assert pe._bbox_padding == (0.1, 0.1, 0.1, 0.1)
