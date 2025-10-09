# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from pathlib import Path
from arduino.app_internal.core.ei import EdgeImpulseRunnerFacade
from arduino.app_utils import HttpClient


@pytest.fixture(autouse=True)
def mock_infra(monkeypatch: pytest.MonkeyPatch):
    """Mock the infrastructure for Edge Impulse tests.

    This fixture sets up a fake docker-compose configuration to avoid
    real docker-compose lookups and network calls during the tests.

    It also mocks the `get_image_bytes` function to return the input
    bytes unchanged.
    """
    # avoid real docker-compose lookups
    fake = {"services": {"ei-inference": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-1337}:1337"]}}}
    monkeypatch.setattr("arduino.app_internal.core.ei.load_brick_compose_file", lambda cls: fake)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda h: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda s: [(None, None), (None, "1337")])
    # identity for get_image_bytes
    monkeypatch.setattr("arduino.app_utils.get_image_bytes", lambda b: b)


@pytest.fixture
def facade():
    """Fixture for the EdgeImpulseRunnerFacade class."""
    return EdgeImpulseRunnerFacade()


def test_infer_from_file_empty(tmp_path: Path, facade: EdgeImpulseRunnerFacade):
    """Test the infer_from_file method with an empty file path.

    Args:
        tmp_path (Path): A temporary directory path.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    assert facade.infer_from_file("") is None


def test_infer_from_file_delegates(tmp_path: Path, facade: EdgeImpulseRunnerFacade, monkeypatch: pytest.MonkeyPatch):
    """Test the infer_from_file method with a valid file path.

    Args:
        tmp_path (Path): A temporary directory path.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """
    f = tmp_path / "x.jpg"
    f.write_bytes(b"data")
    monkeypatch.setattr(facade, "infer_from_image", lambda image_bytes, image_type: {"ok": True})
    out = facade.infer_from_file(str(f))
    assert out == {"ok": True}


def test_infer_invalid_inputs(facade: EdgeImpulseRunnerFacade):
    """Test the infer method with invalid inputs.

    This test checks the behavior of the infer method when provided with
    invalid image data or image type. It ensures that the method returns
    None for invalid inputs, such as empty byte strings or unsupported
    image types.

    Args:
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    assert facade.infer_from_image(b"", "jpg") is None
    assert facade.infer_from_image(b"data", "") is None
    assert facade.infer_from_image(b"data", "bmp") is None


def test_infer_success_and_error(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the infer method for success and error cases.

    This test checks the behavior of the infer method when provided with
    valid image data and image type. It also tests the handling of HTTP
    errors and exceptions during the request.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    # success 200
    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"foo": 1}

    def fake_post(url, files=None):  # noqa
        seen["url"] = url
        seen["files"] = files
        return Resp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    out = facade.infer_from_image(b"data", "jpg")
    assert out == {"foo": 1}
    assert seen["url"].endswith(":1337/api/image")

    # http error
    class Bad:
        status_code = 500
        text = "err"

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: Bad())
    assert facade.infer_from_image(b"data", "png") is None
    # exception
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert facade.infer_from_image(b"data", "png") is None


def test_process_various(facade: EdgeImpulseRunnerFacade, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the process method with various input types.

    Args:
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
        tmp_path (Path): A temporary directory path.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """
    # string path
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    monkeypatch.setattr(facade, "infer_from_image", lambda b, t: {"ok": 1})
    assert facade.process(str(f)) == {"ok": 1}
    # dict with image
    item = {"image": b"x", "image_type": "png"}
    monkeypatch.setattr(facade, "infer_from_image", lambda b, t: {"y": 2})
    assert facade.process(item) == {"y": 2}
    # dict missing image => passthrough
    junk = {"foo": "bar"}
    assert facade.process(junk) == junk
    # other types => passthrough
    assert facade.process(123) == 123


def test_get_model_info(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the get_model_info method of EdgeImpulseRunnerFacade.

    This test checks if the method correctly constructs the URL and retrieves
    model information from the Edge Impulse service.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "project": {
                    "deploy_version": 19,
                    "id": 2251,
                    "impulse_id": 1,
                    "name": "test_model",
                    "version": "1.0.0",
                    "description": "Test model for Edge Impulse",
                    "owner": "Jan Inc.",
                },
                "modelParameters": {
                    "frequency": 16000,
                    "input_features_count": 128,
                    "label_count": 3,
                    "image_input_height": 320,
                    "image_input_width": 320,
                    "labels": ["label1", "label2", "label3"],
                    "model_type": "object_detection",
                    "sensor": 3,
                    "slice_size": 25600,
                    "thresholds": [{"id": 6, "min_score": 0.4000000059604645, "type": "object_detection"}],
                },
            }

    captured = {}

    def fake_get(
        self,
        url: str,
        method: str = "GET",
        data: dict | str = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = 5,
    ):
        captured["url"] = url
        return FakeResp()

    # Mock the requests.get method to return a fake response
    monkeypatch.setattr(HttpClient, "request_with_retry", fake_get)

    info = facade.get_model_info()
    assert captured["url"].endswith("/api/info")
    assert info.name == "test_model"
    assert info.input_features_count == 128
    assert info.label_count == 3
    assert info.frequency == 16000
    assert info.labels == ["label1", "label2", "label3"]
    assert info.model_type == "object_detection"
    assert info.thresholds is not None
    assert isinstance(info.thresholds, list)
    assert info.thresholds[0]["id"] == 6 and info.thresholds[0]["min_score"] == 0.4000000059604645


def test_infer_from_features(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the infer_from_features method of EdgeImpulseRunnerFacade.

    This test checks if the method correctly sends features to the Edge Impulse service
    and retrieves the inference results.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    features = [0.1, 0.2, 0.3]
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": "success"}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    result = facade.infer_from_features(features)
    assert captured["url"].endswith("/api/features")
    assert captured["json"] == {"features": features}
    assert result == {"result": "success"}
