# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from pathlib import Path
from arduino.app_bricks.image_classification import ImageClassification


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Mock out docker-compose lookups and image helpers."""
    fake_compose = {"services": {"models-runner": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"]}}}
    monkeypatch.setattr("arduino.app_internal.core.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda x: [(None, None), (None, "8200")])
    # make get_image_bytes a no-op for raw bytes
    monkeypatch.setattr(
        "arduino.app_utils.get_image_bytes",
        lambda x: x if isinstance(x, (bytes, bytearray)) else None,
    )


@pytest.fixture
def classifier():
    """Fixture to create an instance of ImageClassification.

    Returns:
        ImageClassification: An instance of the ImageClassification class.
    """
    return ImageClassification()


def test_classify_invalid_inputs(classifier: ImageClassification):
    """Test classify method with invalid inputs.

    1. Test with empty bytes and empty type.
    2. Test with non-empty bytes and empty type.
    3. Test with non-empty bytes and unsupported type.

    Args:
        classifier (ImageClassification): An instance of the ImageClassification class.
    """
    # no data
    assert classifier.classify(b"", "jpg") is None
    # no type
    assert classifier.classify(b"abc", "") is None
    # unsupported type
    assert classifier.classify(b"abc", "bmp") is None


def test_classify_success(classifier: ImageClassification, monkeypatch: pytest.MonkeyPatch):
    """Test classify method with valid inputs.

    Args:
        classifier (ImageClassification): An instance of the ImageClassification class.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock dependencies.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"church": 0.5}}}

    captured = {}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    # call with explicit confidence
    out = classifier.classify(b"bytes", "jpg", confidence=0.33)
    assert out == {"classification": [{"class_name": "church", "confidence": "50.00"}]}

    # call with a confidence that is too high
    out = classifier.classify(b"bytes", "jpg", confidence=0.60)
    assert out == {"classification": []}


def test_classify_http_error_and_status_not_ok(classifier: ImageClassification, monkeypatch: pytest.MonkeyPatch):
    """Test classify method with HTTP error and status not OK.

    1. Simulate a non-200 HTTP response.
    2. Simulate a 200 HTTP response with status not equal to 'OK'.

    Args:
        classifier (ImageClassification): _description_
        monkeypatch (pytest.MonkeyPatch): _description_
    """

    # status_code!=200
    class Bad1:
        status_code = 500
        text = "oops"

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: Bad1())
    assert classifier.classify(b"xyz", "png") is None

    # status_code==200 but status!='OK'
    class Bad2:
        status_code = 200

        def json(self):
            return {"status": "FAIL", "message": "err"}

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: Bad2())
    assert classifier.classify(b"xyz", "png") is None


def test_classify_exception_swallowed(classifier: ImageClassification, monkeypatch: pytest.MonkeyPatch):
    """Test classify method with exception during request.

    1. Simulate an exception during the request.
    2. Ensure that the exception is caught and None is returned.

    Args:
        classifier (ImageClassification): An instance of the ImageClassification class.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock dependencies.
    """
    monkeypatch.setattr(
        "arduino.app_internal.core.ei.requests.post",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # exception inside request -> None
    assert classifier.classify(b"data", "jpg") is None


def test_classify_from_file(tmp_path: Path, classifier: ImageClassification, monkeypatch: pytest.MonkeyPatch):
    """Test classify_from_file method with various inputs.

    1. Test with an empty path.
    2. Test with a valid file path and simulate an exception in classify method.
    3. Test with a valid file path and ensure classify method is called correctly.

    Args:
        tmp_path (Path): Pytest fixture to create temporary files.
        classifier (ImageClassification): An instance of the ImageClassification class.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock dependencies.
    """
    # empty path => None
    assert classifier.classify_from_file("") is None

    # classify raises -> returns None
    f = tmp_path / "img.png"
    f.write_bytes(b"data")
    monkeypatch.setattr(classifier, "classify", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    assert classifier.classify_from_file(str(f)) is None

    # classify called with correct args and returns expected result
    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"church": 0.5}}}

    captured = {}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    out = classifier.classify_from_file(str(f), confidence=0.33)
    assert out == {"classification": [{"class_name": "church", "confidence": "50.00"}]}


def test_process(tmp_path: Path, classifier: ImageClassification, monkeypatch: pytest.MonkeyPatch):
    """Test process method with various inputs.

    Args:
        tmp_path (Path): Pytest fixture to create temporary files.
        classifier (ImageClassification): An instance of the ImageClassification class.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock dependencies.
    """
    item = {"result": {"classification": {"church": 0.5}}}
    assert classifier.process(item) == {"classification": [{"class_name": "church", "confidence": "50.00"}]}

    # empty item => None
    assert classifier.process("") is None

    # empty dict => None
    assert classifier.process({}) is None

    # empty image => None
    assert classifier.process({"image": b""}) is None

    # empty image type => None
    assert classifier.process({"image": b"data", "image_type": ""}) is None

    # unsupported image type => None
    assert classifier.process({"image": b"data", "image_type": "bmp"}) is None

    # valid image bytes and type
    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": {"classification": {"church": 0.5}}}

    captured = {}

    def fake_post(
        url: str,
        files: dict = None,
    ):
        captured["url"] = url
        captured["files"] = files
        return FakeResp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    out = classifier.process({"image": b"data", "image_type": "jpg"})
    assert out == {"classification": [{"class_name": "church", "confidence": "50.00"}]}

    # nonexistent file => None
    assert classifier.process("no_file.png") is None
