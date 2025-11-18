# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from pathlib import Path
import io
from PIL import Image
from arduino.app_bricks.object_detection import ObjectDetection


class ModelInfo:
    def __init__(self, model_type: str):
        self.model_type = model_type


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Mock external dependencies in __init__.

    This is needed to avoid network calls and other side effects.
    """
    fake_compose = {"services": {"models-runner": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"]}}}
    monkeypatch.setattr("arduino.app_internal.core.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda x: [(None, None), (None, "8100")])
    monkeypatch.setattr("arduino.app_bricks.object_detection.ObjectDetection.get_model_info", lambda self: ModelInfo("object-detection"))


@pytest.fixture
def detector():
    """Fixture to create an instance of ObjectDetection.

    Returns:
        ObjectDetection: An instance of the ObjectDetection class.
    """
    return ObjectDetection()


def test_detect_invalid_inputs(detector: ObjectDetection):
    """Test invalid inputs for the detect method.

    1. No bytes: should return None
    2. No type: should return None
    3. Unsupported type: should return None

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
    """
    # no bytes
    assert detector.detect(b"", "jpg") is None
    # no type
    assert detector.detect(b"abc", "") is None
    # unsupported type
    assert detector.detect(b"abc", "bmp") is None


def test_detect_success(detector: ObjectDetection, monkeypatch: pytest.MonkeyPatch):
    """Test the detect method with valid inputs.

    This test mocks the requests.post method to avoid actual network calls.

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"bounding_boxes": [{"label": "C", "value": 0.5, "x": 1, "y": 2, "width": 3, "height": 4}]}}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    result = detector.detect(b"bytes", "jpg", confidence=0.25)
    assert result["detection"] == [{"class_name": "C", "confidence": "50.00", "bounding_box_xyxy": [1.0, 2.0, 4.0, 6.0]}]


def test_detect_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, detector: ObjectDetection):
    """Test the detect_from_file method with a valid file path.

    This test creates a temporary file and checks if the detection works as expected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
        tmp_path (Path): The temporary path fixture to create temporary files.
        detector (ObjectDetection): An instance of the ObjectDetection class.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"bounding_boxes": [{"label": "C", "value": 0.5, "x": 1, "y": 2, "width": 3, "height": 4}]}}

    captured = {}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    p = tmp_path / "img.jpg"
    p.write_bytes(b"123")
    result = detector.detect_from_file(str(p))
    assert result["detection"] == [{"class_name": "C", "confidence": "50.00", "bounding_box_xyxy": [1.0, 2.0, 4.0, 6.0]}]


def test_draw_bounding_boxes(detector: ObjectDetection):
    """Test the draw_bounding_boxes method with a valid image and detection.

    This test checks if the method returns a PIL Image object.

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
    """
    img = Image.new("RGB", (20, 20), color="white")
    det = {"detection": [{"class_name": "X", "bounding_box_xyxy": [2, 2, 10, 10], "confidence": "50.0"}]}

    out = detector.draw_bounding_boxes(img, det)
    assert isinstance(out, Image.Image)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    out2 = detector.draw_bounding_boxes(raw, det)
    assert isinstance(out2, Image.Image)


def test_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, detector: ObjectDetection):
    """Test the process method with a valid file path.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        tmp_path (Path): A temporary directory path.
        detector (ObjectDetection): An instance of the ObjectDetection class.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"bounding_boxes": [{"label": "C", "value": 0.5, "x": 1, "y": 2, "width": 3, "height": 4}]}}

    captured = {}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    p = tmp_path / "img.jpg"
    p.write_bytes(b"123")
    result = detector.process(str(p))
    assert result["detection"] == [{"class_name": "C", "confidence": "50.00", "bounding_box_xyxy": [1.0, 2.0, 4.0, 6.0]}]


def test_detect_http_error_status(detector: ObjectDetection, monkeypatch: pytest.MonkeyPatch):
    """Test the detect method with an HTTP error status.

    This test checks if the method returns None when the status code is not 200.

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    class FakeResp:
        status_code = 500
        text = "Server error"

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda url, files=None, headers=None: FakeResp())
    assert detector.detect(b"bytes", "jpg") is None


def test_detect_status_not_ok(detector: ObjectDetection, monkeypatch: pytest.MonkeyPatch):
    """Test the detect method with a non-OK status in the response.

    This test checks if the method returns None when the status is not "OK".

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {"status": "FAIL", "message": "oops"}

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda url, files=None, headers=None: FakeResp())
    assert detector.detect(b"bytes", "jpg") is None


def test_detect_doesnt_propagate_exception(detector: ObjectDetection, monkeypatch: pytest.MonkeyPatch):
    """Test the detect method when an exception occurs.

    This test checks if the method returns None when an exception is raised during detection.

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """
    monkeypatch.setattr(detector, "detect_from_file", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network down")))
    result = detector.detect(b"bytes", "jpg")
    assert result is None, "Expected detect to return None when an exception occurs"


def test_detect_from_file_doesnt_propagate_exception(tmp_path: Path, detector: ObjectDetection, monkeypatch: pytest.MonkeyPatch):
    """Test the detect_from_file method when detect raises an exception.

    This test checks if the method returns None when the detect method raises an exception.

    Args:
        tmp_path (Path): The temporary path fixture to create temporary files.
        detector (ObjectDetection): An instance of the ObjectDetection class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """
    p = tmp_path / "img.jpg"
    p.write_bytes(b"123")
    monkeypatch.setattr(detector, "detect", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad")))
    result = detector.detect_from_file(str(p))
    assert result is None, "Expected detect_from_file to return None when detect raises an error"


def test_process_file_not_found_returns_none(detector: ObjectDetection):
    """Test the process method with a non-existent file path.

    This test checks if the method returns None when the file does not exist (FileNotFoundError).

    Args:
        detector (ObjectDetection): An instance of the ObjectDetection class.
    """
    assert detector.process("no_such_file.jpg") is None
