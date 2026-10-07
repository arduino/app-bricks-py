# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL.Image import Image

from arduino.app_bricks.camera_code_detection import CameraCodeDetection, Detection


@pytest.fixture
def camera() -> MagicMock:
    camera = MagicMock()
    camera.capture.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
    return camera


@pytest.fixture(autouse=True)
def decoded_codes(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make every scanned frame contain two codes."""
    decode = MagicMock(
        return_value=[
            SimpleNamespace(data=b"first", type="QRCODE", rect=(1, 2, 10, 10)),
            SimpleNamespace(data=b"second", type="EAN13", rect=(20, 2, 30, 10)),
        ]
    )
    monkeypatch.setattr("arduino.app_bricks.camera_code_detection.detection.decode", decode)
    return decode


def test_default_camera_is_created_when_none_is_given(monkeypatch: pytest.MonkeyPatch):
    default_camera = MagicMock()
    monkeypatch.setattr("arduino.app_bricks.camera_code_detection.detection.Camera", default_camera)

    CameraCodeDetection(camera=None).start()

    default_camera.return_value.start.assert_called_once()


def test_list_callback_receives_all_detections_at_once(camera: MagicMock):
    received: list[list[Detection]] = []

    def on_codes(frame: Image, detections: list[Detection]) -> None:
        received.append(detections)

    detector = CameraCodeDetection(camera)
    detector.on_detect(on_codes)
    detector.loop()

    assert [[d.content for d in detections] for detections in received] == [["first", "second"]]


def test_single_callback_receives_one_detection_per_call(camera: MagicMock):
    received: list[Detection] = []

    def on_code(frame: Image, detection: Detection) -> None:
        received.append(detection)

    detector = CameraCodeDetection(camera)
    detector.on_detect(on_code)
    detector.loop()

    assert [d.content for d in received] == ["first", "second"]
    assert [d.type for d in received] == ["QRCODE", "EAN13"]
    assert received[0].coords.tolist() == [[1, 2], [11, 2], [11, 12], [1, 12]]


def test_unannotated_callback_receives_one_detection_per_call(camera: MagicMock):
    received = []

    detector = CameraCodeDetection(camera)
    detector.on_detect(lambda frame, detection: received.append(detection))
    detector.loop()

    assert [d.content for d in received] == ["first", "second"]


def test_registering_a_callback_replaces_the_previous_one(camera: MagicMock):
    lists = MagicMock()
    singles: list[Detection] = []

    def on_codes(frame: Image, detections: list[Detection]) -> None:
        lists(detections)

    def on_code(frame: Image, detection: Detection) -> None:
        singles.append(detection)

    detector = CameraCodeDetection(camera)
    detector.on_detect(on_codes)
    detector.on_detect(on_code)
    detector.loop()

    lists.assert_not_called()
    assert len(singles) == 2


@pytest.mark.parametrize("annotation", ["list", "single"])
def test_none_removes_the_callback(camera: MagicMock, annotation: str):
    callback = MagicMock()

    def on_codes(frame: Image, detections: list[Detection]) -> None:
        callback(detections)

    def on_code(frame: Image, detection: Detection) -> None:
        callback(detection)

    detector = CameraCodeDetection(camera)
    detector.on_detect(on_codes if annotation == "list" else on_code)
    detector.on_detect(None)
    detector.loop()

    callback.assert_not_called()


def test_failing_callback_is_reported_to_on_error(camera: MagicMock):
    failure = RuntimeError("boom")
    errors: list[Exception] = []

    def on_code(frame: Image, detection: Detection) -> None:
        raise failure

    detector = CameraCodeDetection(camera)
    detector.on_detect(on_code)
    detector.on_error(errors.append)
    detector.loop()

    assert errors == [failure]


def test_disabling_both_code_kinds_is_rejected(camera: MagicMock):
    with pytest.raises(ValueError, match="At least one of 'detect_qr' or 'detect_barcode' must be True."):
        CameraCodeDetection(camera, detect_qr=False, detect_barcode=False)
