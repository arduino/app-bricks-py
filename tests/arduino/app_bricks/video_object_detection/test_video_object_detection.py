# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import http.client
import queue
import threading
import time

import cv2
import numpy as np
import pytest

import arduino.app_internal.edge_impulse.model as model_module
import arduino.app_internal.ei_inference as ei_inference
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

TIMEOUT = 3.0  # seconds to wait for a callback
# The fake camera produces 160x120 frames, the fake model input is 100x100
CAT = {"label": "cat", "score": 0.9, "x": 16, "y": 24, "w": 48, "h": 48}  # in frame coordinates, as the service maps them
DOG = {"label": "dog", "score": 0.2, "x": 0, "y": 0, "w": 10, "h": 10}


class RunningDetector:
    """A started brick with its detection loop running in a thread."""

    def __init__(self, detector):
        self.detector = detector
        self.camera = detector._camera
        detector.start()
        self.thread = threading.Thread(target=detector.inference_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.detector.stop()
        self.thread.join(TIMEOUT)


@pytest.fixture(autouse=True)
def configured_model(monkeypatch):
    """The app configured the fake model for the brick, as the CLI does through the model variable."""
    monkeypatch.setenv("EI_V_OBJ_DETECTION_MODEL", "/models/ootb/ei/det.eim")


@pytest.fixture
def service(ei_service, monkeypatch):
    monkeypatch.setattr(ei_inference, "DEFAULT_SOCKET_PATH", ei_service.socket_path)
    ei_service.models["det"]["boxes"] = [CAT, DOG]
    ei_service.models["ootb/ei/det"] = ei_service.models["det"]  # the name the configured path resolves to
    return ei_service


@pytest.fixture
def running(service, camera):
    """A running detector with confidence 0.3 and no debounce, the test registers its handlers first."""
    runners = []

    def start(**kwargs):
        detector = VideoObjectDetection(camera=camera, **{"confidence": 0.3, "debounce_sec": 0.0, "stream_port": 0, **kwargs})
        runner = RunningDetector(detector)
        runners.append(runner)
        return runner

    yield start
    for runner in runners:
        runner.stop()


# ---------------------------------------------------------------- configuration


def test_model_from_the_configured_path(camera, monkeypatch):
    monkeypatch.setenv("EI_V_OBJ_DETECTION_MODEL", "/var/lib/arduino-app-cli/models/custom-ei/abc/model.eim")
    monkeypatch.setattr("arduino.app_internal.ei_inference.client.DEFAULT_MODELS_DIR", "/var/lib/arduino-app-cli/models")
    detector = VideoObjectDetection(camera=camera)
    assert detector.model == "custom-ei/abc/model"


def test_default_model_comes_from_the_models_list(camera, monkeypatch):
    from arduino.app_internal.core.module import ModelBrickConfig, ModelEntry

    monkeypatch.delenv("EI_V_OBJ_DETECTION_MODEL", raising=False)
    monkeypatch.setattr(model_module, "get_brick_config", lambda cls: {"id": "arduino:video_object_detection"})
    monkeypatch.setattr(model_module, "get_brick_configured_model", lambda brick_id, brick_config: "yolox-qnn-object-detection")
    entry = ModelEntry(
        "yolox-qnn-object-detection",
        bricks=[
            ModelBrickConfig("arduino:object_detection", {"EI_OBJ_DETECTION_MODEL": "/models/ootb/ei/other.eim"}),
            ModelBrickConfig("arduino:video_object_detection", {"EI_V_OBJ_DETECTION_MODEL": "/models/ootb/ei/yolo-x-nano-qnn.eim"}),
        ],
    )
    monkeypatch.setattr(model_module, "load_model_list", lambda: {"yolox-qnn-object-detection": entry})
    assert VideoObjectDetection(camera=camera).model == "ootb/ei/yolo-x-nano-qnn"


def test_missing_model_configuration_is_an_error(camera, monkeypatch):
    monkeypatch.delenv("EI_V_OBJ_DETECTION_MODEL", raising=False)
    monkeypatch.setattr(model_module, "get_brick_config", lambda cls: None)
    with pytest.raises(RuntimeError, match="EI_V_OBJ_DETECTION_MODEL"):
        VideoObjectDetection(camera=camera)


def test_bundled_model_from_the_configured_path(camera, monkeypatch):
    monkeypatch.setenv("EI_V_OBJ_DETECTION_MODEL", "/models/ootb/ei/yolo-x-nano.eim")
    assert VideoObjectDetection(camera=camera).model == "ootb/ei/yolo-x-nano"


def test_model_outside_the_models_directories_is_an_error(camera, monkeypatch):
    monkeypatch.setenv("EI_V_OBJ_DETECTION_MODEL", "/home/arduino/model.eim")
    with pytest.raises(RuntimeError, match="inference service"):
        VideoObjectDetection(camera=camera)


def test_callbacks_must_be_functions(camera):
    detector = VideoObjectDetection(camera=camera)
    with pytest.raises(TypeError):
        detector.on_detect("cat", "not_a_function")
    with pytest.raises(TypeError):
        detector.on_detect_all(42)


def test_override_threshold_validates_the_value(camera):
    detector = VideoObjectDetection(camera=camera)
    with pytest.raises(TypeError):
        detector.override_threshold("high")
    detector.override_threshold(0.8)
    assert detector.confidence == 0.8


# ---------------------------------------------------------------- detections


def test_frames_reach_the_service_and_boxes_come_back_in_frame_coordinates(running, service):
    received = queue.Queue()
    runner = running()
    runner.detector.on_detect("cat", lambda details: received.put(details))
    runner.camera.push()

    details = received.get(timeout=TIMEOUT)
    assert details["confidence"] == pytest.approx(0.9)
    assert details["bounding_box_xyxy"] == (16, 24, 64, 72)
    assert service.frames[0][2].shape == (120, 160, 3), "the camera frame reaches the service as it is"


def test_handler_without_parameters_is_invoked(running):
    fired = threading.Event()
    runner = running()
    runner.detector.on_detect("cat", lambda: fired.set())
    runner.camera.push()
    assert fired.wait(TIMEOUT)


def test_detections_below_the_confidence_threshold_are_dropped(running):
    seen = queue.Queue()
    runner = running(confidence=0.3)
    runner.detector.on_detect("dog", lambda details: seen.put(("dog", details)))
    runner.detector.on_detect_all(lambda detections: seen.put(("all", detections)))
    runner.camera.push()

    kind, detections = seen.get(timeout=TIMEOUT)
    assert kind == "all" and set(detections) == {"cat"}, "the dog scores 0.2, below the threshold"
    assert len(detections["cat"]) == 1
    with pytest.raises(queue.Empty):
        seen.get(timeout=0.3)


def test_override_threshold_applies_to_the_next_frames(running):
    labels = queue.Queue()
    runner = running(confidence=0.3)
    runner.detector.on_detect_all(lambda detections: labels.put(set(detections)))
    runner.camera.push()
    assert labels.get(timeout=TIMEOUT) == {"cat"}
    runner.detector.override_threshold(0.1)
    runner.camera.push()
    assert labels.get(timeout=TIMEOUT) == {"cat", "dog"}


def test_no_callback_when_nothing_passes_the_threshold(running, service):
    service.models["det"]["boxes"] = [DOG]
    called = threading.Event()
    runner = running()
    runner.detector.on_detect_all(lambda detections: called.set())
    runner.camera.push(3)
    assert not called.wait(0.5)


def test_frame_kwarg_is_none_without_camera_preview(running):
    frames = queue.Queue()
    runner = running()
    runner.detector.on_detect("cat", lambda details, frame=None: frames.put(frame))
    runner.camera.push()
    assert frames.get(timeout=TIMEOUT) is None


def test_frame_kwarg_is_the_jpeg_frame_with_camera_preview(running):
    frames = queue.Queue()
    runner = running(camera_preview=True)
    runner.detector.on_detect_all(lambda detections, frame: frames.put(frame))
    runner.camera.push()
    frame = frames.get(timeout=TIMEOUT)
    assert isinstance(frame, bytes) and frame[:2] == b"\xff\xd8"


def test_debounce_suppresses_rapid_repeats_and_allows_later_ones(running):
    calls = queue.Queue()
    runner = running(debounce_sec=0.5)
    runner.detector.on_detect("cat", lambda: calls.put(time.monotonic()))
    runner.camera.push(3)
    first = calls.get(timeout=TIMEOUT)
    with pytest.raises(queue.Empty):
        calls.get(timeout=0.3)
    time.sleep(0.3)
    runner.camera.push()
    assert calls.get(timeout=TIMEOUT) - first >= 0.5


def test_slow_handler_discards_the_detections_arriving_meanwhile(running):
    calls = []
    release = threading.Event()

    def slow():
        calls.append(time.monotonic())
        release.wait(TIMEOUT)

    runner = running()
    runner.detector.on_detect("cat", slow)
    runner.camera.push(4)
    time.sleep(0.5)
    release.set()
    time.sleep(0.2)
    assert len(calls) == 1


def test_frame_errors_do_not_stop_the_loop(running, service):
    replies = iter([{"code": "internal", "error": "boom"}])
    service.models["det"]["boxes"] = lambda seq, image: next(replies, [CAT])
    fired = threading.Event()
    runner = running()
    runner.detector.on_detect("cat", lambda: fired.set())
    for _ in range(20):  # frames captured while one is in flight are skipped, so pace them
        runner.camera.push()
        if fired.wait(0.1):
            break
    assert fired.is_set()


def test_frames_are_captured_only_when_the_model_is_free(running, service):
    """A slow model paces the capture: no frame is captured, resized or sent while one is in flight."""
    service.reply_delay = 0.3
    results = queue.Queue()
    runner = running()
    runner.detector.on_detect_all(lambda detections: results.put(detections))
    runner.camera.push(6)
    assert results.get(timeout=TIMEOUT)
    time.sleep(0.45)
    assert 1 <= len(service.frames) <= 3, "one frame per inference, the others still wait in the camera"
    assert runner.camera.frames.qsize() >= 3, "nothing was captured while the model was busy"


# ---------------------------------------------------------------- video stream


def read_part(response):
    """The next JPEG of a multipart/x-mixed-replace response."""
    length = None
    while True:
        line = response.readline()
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
        if line == b"\r\n" and length is not None:
            return response.read(length)


def test_video_stream_shows_the_frames_with_the_boxes(running):
    runner = running()
    first_result(runner)
    connection, response = stream_viewer(runner)
    runner.camera.push()
    first = cv2.imdecode(np.frombuffer(read_part(response), np.uint8), cv2.IMREAD_COLOR)
    runner.camera.push()
    second = cv2.imdecode(np.frombuffer(read_part(response), np.uint8), cv2.IMREAD_COLOR)
    connection.close()
    assert first.shape == (120, 160, 3)
    # The cat box is (16, 24)-(64, 72) in frame coordinates: its border is colored, nothing is drawn far from it
    border = first[50, 16].astype(int)
    assert border.max() > 100 and tuple(first[110, 150]) == (0, 0, 0)
    assert np.abs(second[50, 16].astype(int) - border).max() < 40, "the label keeps its color across frames"


def has_cat_box(jpeg):
    """True when the cat box border (16, 24)-(64, 72) is drawn on the frame."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return image[50, 16].astype(int).max() > 100


def stream_viewer(runner):
    connection = http.client.HTTPConnection("127.0.0.1", runner.detector.stream_port, timeout=5)
    connection.request("GET", "/")
    response = connection.getresponse()
    assert response.status == 200
    return connection, response


def first_result(runner):
    """Push one frame and wait for its detections, so the brick knows a set of boxes."""
    results = queue.Queue()
    runner.detector.on_detect_all(lambda detections: results.put(detections))
    runner.camera.push()
    assert results.get(timeout=TIMEOUT)


def test_video_stream_keeps_the_camera_rate_with_a_slow_model(running, service):
    """With a viewer every camera frame is streamed, drawn with the boxes of the latest result."""
    service.reply_delay = 0.3
    runner = running()
    first_result(runner)
    connection, response = stream_viewer(runner)
    frames = []
    for _ in range(10):  # one streamed frame per camera frame
        runner.camera.push()
        frames.append(read_part(response))
    connection.close()
    assert len(service.frames) <= 3, "the model saw one frame per inference"
    assert all(has_cat_box(frame) for frame in frames), "every frame carries the latest boxes"


def test_boxes_disappear_when_the_model_stops_answering(running, service):
    runner = running()
    first_result(runner)
    service.reply_delay = 30.0  # the next frame never gets its result within the test
    connection, response = stream_viewer(runner)
    runner.camera.push()
    assert has_cat_box(read_part(response)), "the boxes are still fresh"
    deadline = time.monotonic() + 3.0
    boxes_gone = False
    while time.monotonic() < deadline and not boxes_gone:
        time.sleep(0.05)
        runner.camera.push()
        boxes_gone = not has_cat_box(read_part(response))
    connection.close()
    assert boxes_gone, "the stale boxes were dropped"


def test_video_stream_can_be_disabled(service, camera):
    detector = VideoObjectDetection(camera=camera, stream_port=None)
    assert detector.stream_port is None
    detector.start()
    detector.stop()


# ---------------------------------------------------------------- service lifecycle


def test_waits_for_the_service_and_reconnects_when_it_restarts(service, camera, monkeypatch):
    monkeypatch.setattr(VideoObjectDetection, "_RETRY_SEC", 0.1)
    fired = queue.Queue()
    service.stop()
    detector = VideoObjectDetection(camera=camera, stream_port=0)
    detector.on_detect("cat", lambda: fired.put(True))
    runner = RunningDetector(detector)
    try:
        runner.camera.push()
        with pytest.raises(queue.Empty):
            fired.get(timeout=0.3)
        service.start()
        runner.camera.push()
        assert fired.get(timeout=TIMEOUT)
        # The service goes away and comes back: the brick reconnects
        service.stop()
        service.start()
        assert service.closed.wait(TIMEOUT)
        for _ in range(5):
            runner.camera.push()
            try:
                assert fired.get(timeout=1.0)
                break
            except queue.Empty:
                continue
        else:
            pytest.fail("no detection after the service restarted")
    finally:
        runner.stop()


def test_stop_closes_the_connection_and_the_camera(running, service):
    runner = running()
    runner.camera.push()
    assert service.opened.wait(TIMEOUT)
    runner.stop()
    assert service.closed.wait(TIMEOUT)
    assert not runner.camera.started
    assert not runner.thread.is_alive()
