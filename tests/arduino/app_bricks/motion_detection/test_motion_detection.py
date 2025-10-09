# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import pytest
import random
import time
from arduino.app_bricks.motion_detection import MotionDetection
from arduino.app_utils import HttpClient

import arduino.app_utils.app as app
from arduino.app_utils import AppController


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Mock out docker-compose lookups and image helpers."""
    fake_compose = {"services": {"models-runner": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"]}}}
    monkeypatch.setattr("arduino.app_internal.core.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda x: [(None, None), (None, "8200")])

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "project": {
                    "deploy_version": 163,
                    "id": 412593,
                    "impulse_id": 1,
                    "impulse_name": "Impulse #1",
                    "name": "Tutorial: Continuous motion recognition",
                    "owner": "Edge Impulse Inc.",
                },
                "modelParameters": {
                    "has_visual_anomaly_detection": False,
                    "axis_count": 3,
                    "frequency": 62.5,
                    "has_anomaly": 1,
                    "has_object_tracking": False,
                    "image_channel_count": 0,
                    "image_input_frames": 0,
                    "image_input_height": 0,
                    "image_input_width": 0,
                    "image_resize_mode": "none",
                    "inferencing_engine": 4,
                    "input_features_count": 375,
                    "interval_ms": 16,
                    "label_count": 4,
                    "labels": ["idle", "snake", "updown", "wave"],
                    "model_type": "classification",
                    "sensor": 2,
                    "slice_size": 31,
                    "thresholds": [],
                    "use_continuous_mode": False,
                    "sensorType": "accelerometer",
                },
            }

    def fake_get(
        self,
        url: str,
        method: str = "GET",
        data: dict | str = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = 5,
    ):
        return FakeResp()

    # Mock the requests.get method to return a fake response
    monkeypatch.setattr(HttpClient, "request_with_retry", fake_get)


def test_classify_success(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"updown": 0.8, "wave": 0.1, "snake": 0.003, "idle": 0.0}}}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = MotionDetection()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    updown_called = False

    def callback_for_updown():
        nonlocal updown_called
        updown_called = True

    ## Initialize the classifier
    classifier.on_movement_detection("updown", callback_for_updown)

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    for i in range(0, 375):
        sensX = random.uniform(0, 1)
        sensY = random.uniform(0, 1)
        sensZ = random.uniform(9.5, 10)
        classifier.accumulate_samples((sensX, sensY, sensZ))

    time.sleep(0.2)  # Allow some time for the classifier to process the samples

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert captured["url"].endswith("/api/features")
    assert updown_called


def test_classify_success_with_arguments(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"updown": 0.8, "wave": 0.1, "snake": 0.003, "idle": 0.0}}}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = MotionDetection()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    updown_called = False
    all_classification = None

    def callback_for_updown(classification: dict):
        nonlocal updown_called
        nonlocal all_classification
        updown_called = True
        all_classification = classification

    ## Initialize the classifier
    classifier.on_movement_detection("updown", callback_for_updown)

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    for i in range(0, 375):
        sensX = random.uniform(0, 1)
        sensY = random.uniform(0, 1)
        sensZ = random.uniform(9.5, 10)
        classifier.accumulate_samples((sensX, sensY, sensZ))

    time.sleep(0.2)  # Allow some time for the classifier to process the samples

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert captured["url"].endswith("/api/features")
    assert updown_called
    assert all_classification is not None
    assert "updown" in all_classification
    assert "idle" in all_classification
    assert "snake" in all_classification
    assert "wave" in all_classification
    assert float(all_classification["updown"]) == 80.0


def test_classify_success_with_more_arguments(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"updown": 0.8, "wave": 0.1, "snake": 0.003, "idle": 0.0}}}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = MotionDetection()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    updown_called = False
    all_classification = None

    def callback_for_updown(classification: dict, intermediate_results: dict = None, other_param: str = "test"):
        nonlocal updown_called
        nonlocal all_classification
        updown_called = True
        all_classification = classification

    ## Initialize the classifier
    classifier.on_movement_detection("updown", callback_for_updown)

    app_thread = threading.Thread(target=app_instance.run, daemon=True)
    app_thread.start()
    time.sleep(0.1)

    for i in range(0, 375):
        sensX = random.uniform(0, 1)
        sensY = random.uniform(0, 1)
        sensZ = random.uniform(9.5, 10)
        classifier.accumulate_samples((sensX, sensY, sensZ))

    time.sleep(0.2)  # Allow some time for the classifier to process the samples

    app_instance._stop_all_bricks()
    app_thread.join(timeout=1)

    assert captured["url"].endswith("/api/features")
    assert updown_called
    assert all_classification is not None
    assert "updown" in all_classification
    assert "idle" in all_classification
    assert "snake" in all_classification
    assert "wave" in all_classification
    assert float(all_classification["updown"]) == 80.0
