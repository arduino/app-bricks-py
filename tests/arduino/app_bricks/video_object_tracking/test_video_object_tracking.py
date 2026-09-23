# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import queue
import threading
import time
from pathlib import Path

import pytest

import arduino.app_internal.ei_inference as ei_inference
from arduino.app_bricks.video_object_tracking import VideoObjectTracking
from arduino.app_internal.ei_inference import Box, Result

TIMEOUT = 3.0  # seconds to wait for a callback
TRACKER = "ootb/ei/tracker"  # the name of the configured model on the service
RECORDED_WALK = Path(__file__).parent / "data" / "walk_640x480.json"
# The threshold blocks of a tracking model, as the service reports them
DETECTION_BLOCK = {"id": 27, "type": "object_detection", "min_score": 0.2}
TRACKING_BLOCK = {"id": 28, "type": "object_tracking", "max_age": 1, "min_hits": 3, "iou_threshold": 0.3}


def _track(label: str = "person", object_id: int = 1, x: int = 0, y: int = 0, width: int = 80, height: int = 200, value: float = 0.9) -> dict:
    """A tracked object as the service reports it, in frame coordinates."""
    return {"label": label, "score": value, "x": x, "y": y, "w": width, "h": height, "id": object_id}


def _result(tracks: list[dict], seq: int = 1) -> Result:
    return Result(model=TRACKER, seq=seq, ts_ns=time.monotonic_ns(), source_size=(640, 480), tracks=[Box(**track) for track in tracks])


def _replay(tracker: VideoObjectTracking, frames: list[list[dict]]) -> None:
    """Feed the brick the results of consecutive frames, as its receiver does."""
    for seq, tracks in enumerate(frames, 1):
        tracker._process_result(_result(tracks, seq))


def _walk(points: list[tuple[int, int]], label: str = "person", object_id: int = 1, size: tuple[int, int] = (80, 200)) -> list[list[dict]]:
    return [[_track(label=label, object_id=object_id, x=x, y=y, width=size[0], height=size[1])] for x, y in points]


def _straight(start: tuple[int, int], end: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    return [(round(start[0] + (end[0] - start[0]) * i / steps), round(start[1] + (end[1] - start[1]) * i / steps)) for i in range(steps + 1)]


def _recorded_walk_frames() -> list[list[dict]]:
    recording = json.loads(RECORDED_WALK.read_text())
    fields = recording["track_fields"]
    return [[_track(**dict(zip(fields, track, strict=True))) for track in frame["tracks"]] for frame in recording["frames"]]


def _wait_for(condition, timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture(autouse=True)
def configured_model(monkeypatch):
    """The app configured the fake model for the brick, as the CLI does through the model variable."""
    monkeypatch.setenv("EI_V_OBJ_TRACKING_MODEL", "/models/ootb/ei/tracker.eim")


@pytest.fixture
def service(ei_service, monkeypatch):
    """The fake service with a 100x100 "tracker" model whose tracks the test sets."""
    monkeypatch.setattr(ei_inference, "DEFAULT_SOCKET_PATH", ei_service.socket_path)
    ei_service.models[TRACKER] = {
        "width": 100,
        "height": 100,
        "labels": ["person"],
        "object_tracking": True,
        "thresholds": [dict(DETECTION_BLOCK), dict(TRACKING_BLOCK)],
        "tracks": [],
    }
    return ei_service


@pytest.fixture
def tracker(service, camera):
    """A brick built against the service and not started: the tests feed it results directly."""
    brick = VideoObjectTracking(camera=camera, stream_port=0)
    yield brick
    brick.stop()


@pytest.fixture
def running(service, camera):
    """A started brick with its tracking loop in a thread, built with the given options."""
    started = []

    def start(**options):
        brick = VideoObjectTracking(camera=camera, stream_port=0, **options)
        brick.start()
        thread = threading.Thread(target=brick.inference_loop, daemon=True)
        thread.start()
        started.append((brick, thread))
        return brick

    yield start
    for brick, thread in started:
        brick.stop()
        thread.join(TIMEOUT)


# ---------------------------------------------------------------- tracked objects


def test_tracked_objects_reach_the_callbacks_with_their_ids(running, service, camera):
    service.models[TRACKER]["tracks"] = [_track(object_id=7, x=10, y=20, width=30, height=40, value=0.75)]
    received = queue.Queue()
    brick = running()
    brick.on_detect_all(lambda detections: received.put(detections))
    camera.push()

    assert received.get(timeout=TIMEOUT) == {"person": [{"object_id": 7, "confidence": 0.75, "bounding_box_xyxy": (10, 20, 40, 60)}]}
    assert service.frames[0][2].shape == (120, 160, 3), "the camera frame reaches the service as it is"


def test_every_object_of_a_label_is_reported(running, service, camera):
    service.models[TRACKER]["tracks"] = [_track(object_id=1, x=0), _track(object_id=2, x=300)]
    received = queue.Queue()
    brick = running()
    brick.on_detect_all(lambda detections: received.put(detections))
    camera.push()

    assert [detection["object_id"] for detection in received.get(timeout=TIMEOUT)["person"]] == [1, 2]


def test_track_id_zero_is_reported(running, service, camera):
    service.models[TRACKER]["tracks"] = [_track(object_id=0)]
    fired = threading.Event()
    brick = running()
    brick.on_detect("person", lambda: fired.set())
    camera.push()

    assert fired.wait(TIMEOUT)


def test_labels_to_track_filters_the_callbacks_too(running, service, camera):
    service.models[TRACKER]["tracks"] = [_track(label="microwave", object_id=5)]
    received = queue.Queue()
    brick = running(labels_to_track=["person"])
    brick.on_detect_all(lambda detections: received.put(detections))
    camera.push()

    with pytest.raises(queue.Empty):
        received.get(timeout=0.3)
    assert brick.get_unique_objects_count() == {}


def test_the_confidence_is_applied_by_the_service(running, service, camera):
    service.models[TRACKER]["tracks"] = [_track(value=0.1)]
    received = queue.Queue()
    brick = running(confidence=0.9)
    brick.on_detect_all(lambda detections: received.put(detections))
    camera.push()
    with pytest.raises(queue.Empty):
        received.get(timeout=0.3)

    brick.override_threshold(0.05)
    assert (TRACKER, {"confidence": 0.05}) in service.configured
    camera.push()
    assert received.get(timeout=TIMEOUT)["person"][0]["confidence"] == pytest.approx(0.1)


def test_the_video_boxes_carry_the_object_ids(tracker):
    _replay(tracker, [[_track(object_id=7), _track(label="dog", object_id=2, x=300)]])

    assert set(tracker._boxes.visible()) == {"person #7", "dog #2"}


# ---------------------------------------------------------------- thresholds


def test_the_tracker_knobs_are_set_on_the_model_as_soon_as_it_is_opened(tracker, service):
    assert service.configured == [(TRACKER, {"id": 28, "max_age": 3, "min_hits": 3, "iou_threshold": 0.1})]


def test_overrides_reach_the_model_at_runtime(tracker, service):
    tracker.override_keep_grace(5)
    tracker.override_min_detections(2)
    tracker.override_iou_threshold(0.25)
    tracker.override_threshold(0.7)

    assert service.configured[1:] == [
        (TRACKER, {"id": 28, "max_age": 5}),
        (TRACKER, {"id": 28, "min_hits": 2}),
        (TRACKER, {"id": 28, "iou_threshold": 0.25}),
        (TRACKER, {"confidence": 0.7}),
    ]
    assert service.models[TRACKER]["thresholds"][1]["max_age"] == 5


def test_override_values_must_be_numbers(tracker, service):
    with pytest.raises(TypeError):
        tracker.override_keep_grace("many")
    with pytest.raises(TypeError):
        tracker.override_threshold(True)
    assert len(service.configured) == 1


def test_the_centroid_knob_is_skipped_on_a_bounding_box_model(tracker, service):
    tracker.override_euclidean_distance_threshold(50)

    assert len(service.configured) == 1


def test_a_centroid_model_is_matched_by_distance(service, camera):
    service.models[TRACKER]["model_type"] = "constrained_object_detection"
    service.models[TRACKER]["thresholds"] = [
        dict(DETECTION_BLOCK),
        {"id": 28, "type": "object_tracking", "max_age": 1, "min_hits": 3, "threshold": 40},
    ]
    tracker = VideoObjectTracking(camera=camera, stream_port=0, euclidean_distance_threshold=60)
    try:
        assert service.configured[-1] == (TRACKER, {"id": 28, "max_age": 3, "min_hits": 3, "threshold": 60})
        tracker.override_iou_threshold(0.5)
        assert len(service.configured) == 1, "the overlap knob is skipped"
        tracker.override_euclidean_distance_threshold(70)
        assert service.configured[-1] == (TRACKER, {"id": 28, "threshold": 70})
    finally:
        tracker.stop()


def test_a_knob_the_tracking_block_does_not_declare_is_refused(service, camera):
    service.models[TRACKER]["thresholds"] = [dict(DETECTION_BLOCK), {"id": 28, "type": "object_tracking", "max_age": 1, "iou_threshold": 0.3}]
    tracker = VideoObjectTracking(camera=camera, stream_port=0)
    try:
        assert service.configured[-1] == (TRACKER, {"id": 28, "max_age": 3, "iou_threshold": 0.1}), "the knobs the block has are set"
        with pytest.raises(RuntimeError, match="max_age, iou_threshold"):
            tracker.override_min_detections(4)
    finally:
        tracker.stop()


def test_overrides_before_the_connection_are_set_when_the_model_is_opened(service, camera, monkeypatch):
    monkeypatch.setattr(ei_inference, "DEFAULT_SOCKET_PATH", "/nonexistent/ei.sock")
    tracker = VideoObjectTracking(camera=camera, stream_port=0)
    tracker.override_keep_grace(6)
    assert service.configured == [], "nothing to set on while the service is away"

    monkeypatch.setattr(ei_inference, "DEFAULT_SOCKET_PATH", service.socket_path)
    monkeypatch.setattr(VideoObjectTracking, "_RETRY_SEC", 0.1)
    tracker.start()
    thread = threading.Thread(target=tracker.inference_loop, daemon=True)
    thread.start()
    try:
        assert _wait_for(lambda: len(service.configured) == 1)
        assert service.configured[0] == (TRACKER, {"id": 28, "max_age": 6, "min_hits": 3, "iou_threshold": 0.1})
    finally:
        tracker.stop()
        thread.join(TIMEOUT)


# ---------------------------------------------------------------- the model


def test_a_model_without_the_tracking_block_refuses_to_start(service, camera):
    service.models[TRACKER]["object_tracking"] = False

    with pytest.raises(RuntimeError, match="no object tracking block"):
        VideoObjectTracking(camera=camera, stream_port=0)
    assert service.closed.wait(TIMEOUT), "the model is released"


def test_an_unreachable_service_lets_the_brick_start(camera, monkeypatch):
    monkeypatch.setattr(ei_inference, "DEFAULT_SOCKET_PATH", "/nonexistent/ei.sock")

    tracker = VideoObjectTracking(camera=camera, stream_port=0)
    tracker.stop()


def test_a_refused_model_lets_the_brick_start(service, camera, monkeypatch):
    monkeypatch.setenv("EI_V_OBJ_TRACKING_MODEL", "/models/ootb/ei/nope.eim")
    tracker = VideoObjectTracking(camera=camera, stream_port=0)
    tracker.stop()


def test_a_new_connection_forgets_the_identifiers_and_keeps_the_counts(running, service, camera, monkeypatch):
    monkeypatch.setattr(VideoObjectTracking, "_RETRY_SEC", 0.1)
    service.models[TRACKER]["tracks"] = [_track(object_id=3)]
    seen = queue.Queue()
    brick = running()
    brick.on_detect("person", lambda details: seen.put(details["object_id"]))
    camera.push()
    assert seen.get(timeout=TIMEOUT) == 3
    assert brick.get_unique_objects_count() == {"person": 1}

    service.stop()
    service.start()
    for _ in range(10):
        camera.push()
        try:
            seen.get(timeout=0.5)
            break
        except queue.Empty:
            continue
    else:
        pytest.fail("no tracked object after the service restarted")

    assert brick.get_unique_objects_count() == {"person": 2}, "the same id on a new run of the tracker is a new object"


# ---------------------------------------------------------------- counting


def test_each_track_id_is_counted_once(tracker):
    _replay(tracker, _walk(_straight((0, 240), (400, 240), steps=20)))

    assert tracker.get_unique_objects_count() == {"person": 1}


def test_the_same_object_coming_back_with_a_new_id_is_counted_again(tracker):
    _replay(tracker, _walk(_straight((0, 240), (400, 240), steps=10), object_id=1))
    _replay(tracker, _walk(_straight((0, 240), (400, 240), steps=10), object_id=2))

    assert tracker.get_unique_objects_count() == {"person": 2}


def test_vertical_line_is_crossed_by_the_box_centre(tracker):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, _walk(_straight((240, 240), (400, 240), steps=8)))

    assert tracker.get_line_crossing_counts() == {"person": 1}


def test_a_box_straddling_the_line_is_not_counted_until_its_centre_crosses(tracker):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, _walk(_straight((240, 240), (270, 240), steps=6)))

    assert tracker.get_line_crossing_counts() == {}


def test_both_crossing_directions_add_to_the_same_counter(tracker):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, _walk(_straight((240, 240), (400, 240), steps=8)))
    _replay(tracker, _walk(_straight((400, 240), (240, 240), steps=8)))

    assert tracker.get_line_crossing_counts() == {"person": 2}


def test_horizontal_line_helper_spans_480_pixels_whatever_the_camera_width(tracker):
    tracker.set_horizontal_crossing_line(240)

    assert tracker._line_coordinates == (0, 240, 480, 240)


def test_with_no_line_set_nothing_is_counted(tracker):
    _replay(tracker, _walk([(100, 20), (100, 0), (100, 240)]))

    assert tracker.get_line_crossing_counts() == {}


def test_setting_the_line_keeps_the_objects_already_counted(tracker):
    _replay(tracker, _walk(_straight((0, 240), (200, 240), steps=5), object_id=1))
    _replay(tracker, _walk(_straight((0, 240), (200, 240), steps=5), object_id=2))
    assert tracker.get_unique_objects_count() == {"person": 2}

    tracker.set_vertical_crossing_line(320)
    _replay(tracker, _walk(_straight((200, 240), (400, 240), steps=5), object_id=2))

    assert tracker.get_unique_objects_count() == {"person": 2}
    assert tracker.get_line_crossing_counts() == {"person": 1}


def test_direction_is_mirrored_on_the_horizontal_axis(tracker):
    _replay(tracker, _walk(_straight((0, 240), (400, 240), steps=8)))

    assert tracker.get_objects_directions() == {1: ["left"]}


def test_movement_below_the_threshold_reports_no_direction(tracker):
    _replay(tracker, _walk([(100, 240), (105, 243)]))

    assert tracker.get_objects_directions() == {}


def test_diagonal_direction_needs_an_exact_45_degree_step(tracker):
    _replay(tracker, _walk([(100, 100), (150, 150), (200, 201)]))

    assert tracker.get_objects_directions() == {1: ["down-left", "down"]}


def test_reset_counters_keeps_the_direction_history(tracker):
    tracker.set_vertical_crossing_line(320)
    _replay(tracker, _walk(_straight((240, 240), (400, 240), steps=8)))

    tracker.reset_counters()

    assert tracker.get_unique_objects_count() == {}
    assert tracker.get_line_crossing_counts() == {}
    assert tracker.get_objects_directions() == {1: ["left"]}


def test_recorded_walk_is_one_track_crossing_the_line_eight_times(tracker):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, _recorded_walk_frames())

    assert tracker.get_unique_objects_count() == {"person": 1}
    assert tracker.get_line_crossing_counts() == {"person": 8}
    assert len(tracker.get_objects_directions()[3]) == 30
