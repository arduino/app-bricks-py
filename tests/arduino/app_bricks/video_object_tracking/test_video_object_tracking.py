# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.video_object_tracking import VideoObjectTracking, VideoObjectTrackingError

FAKE_COMPOSE = {"services": {"ei-video-obj-tracking-runner": {}}}

TIMEOUT = 2.0

RECORDED_WALK = Path(__file__).parent / "data" / "walk_640x480.json"


def _track(label: str = "person", object_id: int = 1, x: int = 0, y: int = 0, width: int = 80, height: int = 200, value: float = 0.9) -> dict:
    return {"label": label, "object_id": object_id, "x": x, "y": y, "width": width, "height": height, "value": value}


def _classification(tracks: list[dict], boxes: list[dict] | None = None) -> str:
    result = {"object_tracking": tracks, "bounding_boxes": boxes if boxes is not None else tracks}
    return json.dumps({"type": "classification", "result": result, "timeMs": 4})


def _hello(thresholds: list[dict] | None = None, has_object_tracking: bool = True) -> str:
    blocks = (
        thresholds
        if thresholds is not None
        else [
            {"id": 27, "type": "object_detection", "min_score": 0.2},
            {"id": 28, "type": "object_tracking", "max_age": 1, "min_hits": 3, "iou_threshold": 0.3},
        ]
    )
    return json.dumps({
        "type": "hello",
        "modelParameters": {
            "model_type": "object_detection",
            "has_object_tracking": has_object_tracking,
            "image_input_width": 416,
            "image_input_height": 416,
            "labels": ["person"],
            "thresholds": blocks,
        },
    })


def _replay(tracker: VideoObjectTracking, ws, frames: list[list[dict]]) -> None:
    for tracks in frames:
        tracker._process_message(ws, _classification(tracks))


def _walk(points: list[tuple[int, int]], label: str = "person", object_id: int = 1, size: tuple[int, int] = (80, 200)) -> list[list[dict]]:
    return [[_track(label=label, object_id=object_id, x=x, y=y, width=size[0], height=size[1])] for x, y in points]


def _straight(start: tuple[int, int], end: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    return [(round(start[0] + (end[0] - start[0]) * i / steps), round(start[1] + (end[1] - start[1]) * i / steps)) for i in range(steps + 1)]


def _recorded_walk_frames() -> list[list[dict]]:
    recording = json.loads(RECORDED_WALK.read_text())
    fields = recording["track_fields"]
    return [[dict(zip(fields, track, strict=True)) for track in frame["tracks"]] for frame in recording["frames"]]


class _Handshake:
    """Stands in for the runner during the constructor handshake."""

    def __init__(self, hello: str | None):
        self.hello = hello
        self.frames_sent = 0

    def create_connection(self, address, timeout=None):
        return self

    def sendall(self, payload: bytes) -> None:
        self.frames_sent += 1

    def connect(self, uri, open_timeout=None):
        if self.hello is None:
            raise TimeoutError("no runner")
        return self

    def recv(self, timeout=None) -> str:
        return self.hello

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def handshake(monkeypatch: pytest.MonkeyPatch) -> _Handshake:
    runner = _Handshake(_hello())
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking.socket.create_connection", runner.create_connection)
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking.connect", runner.connect)
    return runner


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
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
def tracker():
    t = VideoObjectTracking(debounce_sec=0.0)
    yield t
    t._executor.shutdown(wait=False)


@pytest.fixture
def ws():
    return MagicMock()


@pytest.fixture
def logged_errors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking.logger.error", messages.append)
    return messages


@pytest.fixture
def overrides(monkeypatch: pytest.MonkeyPatch):
    class Recorder:
        def __init__(self):
            self.sent = []
            self.connections = 0

        def connect(self, uri):
            self.connections += 1
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def send(self, payload):
            message = json.loads(payload)
            self.sent.append((message["id"], message["key"], message["value"]))

    recorder = Recorder()
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking.connect", recorder.connect)
    monkeypatch.setattr("arduino.app_bricks.video_objectdetection.connect", recorder.connect)
    return recorder


def test_detect_all_payload_lists_each_track_with_its_confidence(tracker: VideoObjectTracking, ws):
    received = []
    tracker.on_detect_all(lambda detections: received.append(detections))

    tracker._process_message(ws, _classification([_track(object_id=7, x=10, y=20, width=30, height=40, value=0.75)]))
    tracker._executor.shutdown(wait=True)

    assert received == [{"person": [{"object_id": 7, "confidence": 0.75, "bounding_box_xyxy": (10, 20, 40, 60)}]}]


def test_detect_all_reports_every_object_of_the_same_label(tracker: VideoObjectTracking, ws):
    received = []
    tracker.on_detect_all(lambda detections: received.append(detections))

    two_people = [_track(object_id=1, x=0), _track(object_id=2, x=300)]
    tracker._process_message(ws, _classification(two_people))
    tracker._executor.shutdown(wait=True)

    assert len(received) == 1
    assert [detection["object_id"] for detection in received[0]["person"]] == [1, 2]


def test_track_id_zero_is_reported(tracker: VideoObjectTracking, ws):
    fired = threading.Event()
    tracker.on_detect("person", lambda: fired.set())

    tracker._process_message(ws, _classification([_track(object_id=0)]))

    assert fired.wait(TIMEOUT)


def test_box_without_track_id_is_discarded(tracker: VideoObjectTracking, ws):
    fired = threading.Event()
    tracker.on_detect("person", lambda: fired.set())

    without_id = {"label": "person", "x": 0, "y": 0, "width": 10, "height": 10, "value": 0.9}
    tracker._process_message(ws, _classification([without_id]))

    assert not fired.wait(0.2)
    assert tracker.get_unique_objects_count() == {}


def test_each_track_id_is_counted_once(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk(_straight((0, 240), (400, 240), steps=20)))

    assert tracker.get_unique_objects_count() == {"person": 1}


def test_the_same_object_coming_back_with_a_new_id_is_counted_again(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk(_straight((0, 240), (400, 240), steps=10), object_id=1))
    _replay(tracker, ws, _walk(_straight((0, 240), (400, 240), steps=10), object_id=2))

    assert tracker.get_unique_objects_count() == {"person": 2}


def test_labels_to_track_filters_the_callbacks_too(ws):
    tracker = VideoObjectTracking(debounce_sec=0.0, labels_to_track=["person"])
    received = []
    tracker.on_detect_all(lambda detections: received.append(detections))

    tracker._process_message(ws, _classification([_track(label="microwave", object_id=5)]))
    tracker._executor.shutdown(wait=True)

    assert tracker.get_unique_objects_count() == {}
    assert received == []


def test_confidence_is_not_filtered_by_the_brick(ws):
    tracker = VideoObjectTracking(debounce_sec=0.0, confidence=0.9)
    received = []
    tracker.on_detect_all(lambda detections: received.append(detections))

    tracker._process_message(ws, _classification([_track(value=0.1)]))
    tracker._executor.shutdown(wait=True)

    assert len(received) == 1


def test_vertical_line_is_crossed_by_the_box_centre(tracker: VideoObjectTracking, ws):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, ws, _walk(_straight((240, 240), (400, 240), steps=8)))

    assert tracker.get_line_crossing_counts() == {"person": 1}


def test_a_box_straddling_the_line_is_not_counted_until_its_centre_crosses(tracker: VideoObjectTracking, ws):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, ws, _walk(_straight((240, 240), (270, 240), steps=6)))

    assert tracker.get_line_crossing_counts() == {}


def test_both_crossing_directions_add_to_the_same_counter(tracker: VideoObjectTracking, ws):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, ws, _walk(_straight((240, 240), (400, 240), steps=8)))
    _replay(tracker, ws, _walk(_straight((400, 240), (240, 240), steps=8)))

    assert tracker.get_line_crossing_counts() == {"person": 2}


def test_horizontal_line_helper_spans_480_pixels_whatever_the_camera_width(tracker: VideoObjectTracking):
    tracker.set_horizontal_crossing_line(240)

    assert tracker._line_coordinates == (0, 240, 480, 240)


def test_with_no_line_set_nothing_is_counted(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk([(100, 20), (100, 0), (100, 240)]))

    assert tracker.get_line_crossing_counts() == {}


def test_setting_the_line_keeps_the_objects_already_counted(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk(_straight((0, 240), (200, 240), steps=5), object_id=1))
    _replay(tracker, ws, _walk(_straight((0, 240), (200, 240), steps=5), object_id=2))
    assert tracker.get_unique_objects_count() == {"person": 2}

    tracker.set_vertical_crossing_line(320)
    _replay(tracker, ws, _walk(_straight((200, 240), (400, 240), steps=5), object_id=2))

    assert tracker.get_unique_objects_count() == {"person": 2}
    assert tracker.get_line_crossing_counts() == {"person": 1}


def test_direction_is_mirrored_on_the_horizontal_axis(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk(_straight((0, 240), (400, 240), steps=8)))

    assert tracker.get_objects_directions() == {1: ["left"]}


def test_movement_below_the_threshold_reports_no_direction(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk([(100, 240), (105, 243)]))

    assert tracker.get_objects_directions() == {}


def test_diagonal_direction_needs_an_exact_45_degree_step(tracker: VideoObjectTracking, ws):
    _replay(tracker, ws, _walk([(100, 100), (150, 150), (200, 201)]))

    assert tracker.get_objects_directions() == {1: ["down-left", "down"]}


def test_reset_counters_keeps_the_direction_history(tracker: VideoObjectTracking, ws):
    tracker.set_vertical_crossing_line(320)
    _replay(tracker, ws, _walk(_straight((240, 240), (400, 240), steps=8)))

    tracker.reset_counters()

    assert tracker.get_unique_objects_count() == {}
    assert tracker.get_line_crossing_counts() == {}
    assert tracker.get_objects_directions() == {1: ["left"]}


def test_thresholds_are_configured_over_the_receiving_connection(tracker: VideoObjectTracking, ws, overrides):
    tracker._process_message(ws, _hello())

    sent = [json.loads(call.args[0]) for call in ws.send.call_args_list]
    assert [(m["id"], m["key"], m["value"]) for m in sent] == [
        (27, "min_score", 0.4),
        (28, "max_age", 3),
        (28, "min_hits", 3),
        (28, "iou_threshold", 0.1),
    ]
    assert overrides.connections == 0


def test_the_centroid_knob_is_skipped_on_a_bounding_box_model(tracker: VideoObjectTracking, ws, overrides):
    tracker._process_message(ws, _hello())
    overrides.sent.clear()

    tracker.override_euclidean_distance_threshold(50)

    assert overrides.sent == []


def test_a_knob_the_tracking_block_does_not_declare_is_refused(tracker: VideoObjectTracking, ws, overrides):
    tracker._process_message(ws, _hello())

    with pytest.raises(RuntimeError, match="max_age, min_hits, iou_threshold"):
        tracker._override_config_value(overrides, "threshold", 50)


def test_a_model_without_the_tracking_block_reports_no_tracks(tracker: VideoObjectTracking, ws):
    only_detection = [{"id": 12, "type": "object_detection", "min_score": 0.3}]
    tracker._process_message(ws, _hello(thresholds=only_detection, has_object_tracking=False))

    received = []
    tracker.on_detect_all(lambda detections: received.append(detections))
    tracker._process_message(ws, _classification([], boxes=[_track()]))
    tracker._executor.shutdown(wait=True)

    assert received == []


def test_recorded_walk_is_one_track_crossing_the_line_eight_times(tracker: VideoObjectTracking, ws):
    tracker.set_vertical_crossing_line(320)

    _replay(tracker, ws, _recorded_walk_frames())

    assert tracker.get_unique_objects_count() == {"person": 1}
    assert tracker.get_line_crossing_counts() == {"person": 8}
    assert len(tracker.get_objects_directions()[3]) == 30


def test_a_new_hello_forgets_the_identifiers_and_keeps_the_counts(tracker: VideoObjectTracking, ws, overrides):
    _replay(tracker, ws, _walk(_straight((0, 240), (200, 240), steps=5), object_id=3))
    assert tracker.get_unique_objects_count() == {"person": 1}

    tracker._process_message(ws, _hello())
    _replay(tracker, ws, _walk(_straight((0, 240), (200, 240), steps=5), object_id=3))

    assert tracker.get_unique_objects_count() == {"person": 2}
    assert tracker.get_objects_directions() == {3: ["left"]}


def test_the_constructor_asks_the_runner_about_the_model(handshake: _Handshake, tracker: VideoObjectTracking):
    assert handshake.frames_sent == 1


def test_a_model_without_the_tracking_block_refuses_to_start(handshake: _Handshake):
    handshake.hello = _hello(thresholds=[{"id": 12, "type": "object_detection", "min_score": 0.3}], has_object_tracking=False)

    with pytest.raises(VideoObjectTrackingError, match="no object tracking block"):
        VideoObjectTracking()


def test_a_runner_that_does_not_answer_lets_the_brick_start(handshake: _Handshake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking._HANDSHAKE_TIMEOUT", 0.1)
    monkeypatch.setattr("arduino.app_bricks.video_object_tracking._HANDSHAKE_STEP", 0.01)
    handshake.hello = None

    tracker = VideoObjectTracking()
    tracker._executor.shutdown(wait=False)


def test_a_model_without_the_tracking_block_is_reported_as_an_error(tracker: VideoObjectTracking, ws, overrides, logged_errors):
    only_detection = [{"id": 12, "type": "object_detection", "min_score": 0.3}]

    tracker._process_message(ws, _hello(thresholds=only_detection, has_object_tracking=False))

    assert any("no object tracking block" in message for message in logged_errors)
    assert overrides.sent == []
