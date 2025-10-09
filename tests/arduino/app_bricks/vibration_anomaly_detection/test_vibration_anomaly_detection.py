# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import threading
import pytest
import random
import time
from arduino.app_bricks.vibration_anomaly_detection import VibrationAnomalyDetection
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
                    "deploy_version": 11,
                    "id": 774707,
                    "impulse_id": 1,
                    "impulse_name": "Time series data, Spectral Analysis, Classification (Keras), Anomaly Detection (K-means)",
                    "name": "Fan Monitoring - Advanced Anomaly Detection",
                    "owner": "Arduino",
                },
                "modelParameters": {
                    "has_visual_anomaly_detection": False,
                    "axis_count": 3,
                    "frequency": 100,
                    "has_anomaly": 1,
                    "has_object_tracking": False,
                    "has_performance_calibration": False,
                    "image_channel_count": 0,
                    "image_input_frames": 0,
                    "image_input_height": 0,
                    "image_input_width": 0,
                    "image_resize_mode": "none",
                    "inferencing_engine": 4,
                    "input_features_count": 600,
                    "interval_ms": 10,
                    "label_count": 2,
                    "labels": ["nominal", "off"],
                    "model_type": "classification",
                    "sensor": 2,
                    "slice_size": 50,
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


def test_classify_no_anomaly(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "result": {
                    "anomaly": -0.25,
                    "classification": {"nominal": 0.8522483706474304, "off": 0.14775167405605316},
                }
            }

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = VibrationAnomalyDetection(anomaly_detection_threshold=1.5)

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    anomaly_trigger_called = False

    def callback_for_anomaly():
        nonlocal anomaly_trigger_called
        anomaly_trigger_called = True

    ## Initialize the classifier
    classifier.on_anomaly(callback_for_anomaly)

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
    assert not anomaly_trigger_called


def test_classify_success(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "result": {
                    "anomaly": 2.2945172786712646,
                    "classification": {"nominal": 0.8522483706474304, "off": 0.14775167405605316},
                }
            }

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = VibrationAnomalyDetection(anomaly_detection_threshold=1.5)

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    anomaly_trigger_called = False

    def callback_for_anomaly():
        nonlocal anomaly_trigger_called
        anomaly_trigger_called = True

    ## Initialize the classifier
    classifier.on_anomaly(callback_for_anomaly)

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
    assert anomaly_trigger_called


def test_classify_success_with_arguments(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "result": {
                    "anomaly": 2.2945172786712646,
                    "classification": {"nominal": 0.8522483706474304, "off": 0.14775167405605316},
                }
            }

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = VibrationAnomalyDetection()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    anomaly_trigger_called = False
    score = None

    def callback_for_anomaly(anomaly_score: float):
        nonlocal anomaly_trigger_called
        nonlocal score
        anomaly_trigger_called = True
        score = anomaly_score

    ## Initialize the classifier
    classifier.on_anomaly(callback_for_anomaly)

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
    assert anomaly_trigger_called
    assert score is not None
    assert float(score) == 2.2945172786712646


def test_classify_success_with_more_arguments(app_instance: AppController, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "result": {
                    "anomaly": 3.56,
                    "classification": {"nominal": 0.8522483706474304, "off": 0.14775167405605316},
                }
            }

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    classifier = VibrationAnomalyDetection()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    anomaly_trigger_called = False
    score = None
    all_classification = None

    def callback_for_anomaly(anomaly_score: float, classification: dict):
        nonlocal anomaly_trigger_called
        nonlocal score
        nonlocal all_classification
        anomaly_trigger_called = True
        score = anomaly_score
        all_classification = classification

    ## Initialize the classifier
    classifier.on_anomaly(callback_for_anomaly)

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
    assert anomaly_trigger_called
    assert all_classification is not None
    assert "classification" in all_classification
    classifications = all_classification["classification"]
    assert len(classifications) == 2
    detected_classes = 0
    for key in classifications:
        if key["class_name"] in ["nominal", "off"]:
            detected_classes += 1
    assert detected_classes == 2
