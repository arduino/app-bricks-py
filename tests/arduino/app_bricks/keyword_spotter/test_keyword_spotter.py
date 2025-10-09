# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from arduino.app_bricks.keyword_spotting import KeywordSpotting
from arduino.app_utils import HttpClient


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
                    "deploy_version": 84,
                    "id": 412592,
                    "impulse_id": 1,
                    "impulse_name": "Time series data, Audio (MFCC), Neural Network (Keras) #1",
                    "name": "Tutorial: Responding to your voice",
                    "owner": "Edge Impulse Inc.",
                },
                "modelParameters": {
                    "has_visual_anomaly_detection": False,
                    "axis_count": 1,
                    "frequency": 16000,
                    "has_anomaly": 0,
                    "has_object_tracking": False,
                    "image_channel_count": 0,
                    "image_input_frames": 0,
                    "image_input_height": 0,
                    "image_input_width": 0,
                    "image_resize_mode": "none",
                    "inferencing_engine": 4,
                    "input_features_count": 15488,
                    "interval_ms": 0.0625,
                    "label_count": 3,
                    "labels": ["helloworld", "noise", "unknown"],
                    "model_type": "classification",
                    "sensor": 1,
                    "slice_size": 3872,
                    "thresholds": [],
                    "use_continuous_mode": True,
                    "sensorType": "microphone",
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


@pytest.fixture
def classifier():
    """Fixture to create an instance of KeywordSpotting.

    Returns:
        KeywordSpotting: An instance of the KeywordSpotting class.
    """
    return KeywordSpotting()
