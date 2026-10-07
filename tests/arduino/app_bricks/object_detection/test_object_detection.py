# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from arduino.app_bricks.object_detection import ObjectDetection


def _detector(confidence: float = 0.3, model_type: str = "object_detection") -> ObjectDetection:
    # Bypass __init__: no Edge Impulse runner is needed to post-process its results
    detector = ObjectDetection.__new__(ObjectDetection)
    detector.confidence = confidence
    detector._model_info = SimpleNamespace(model_type=model_type)
    return detector


def _inference(*boxes: tuple[str, float]) -> dict[str, Any]:
    return {"result": {"bounding_boxes": [{"label": label, "value": value, "x": 1, "y": 2, "width": 3, "height": 4} for label, value in boxes]}}


def test_detections_below_the_module_confidence_are_dropped():
    out = _detector(confidence=0.5)._extract_detection(_inference(("cat", 0.9), ("dog", 0.2)))
    assert out == {"detection": [{"class_name": "cat", "confidence": "90.00", "bounding_box_xyxy": [1.0, 2.0, 4.0, 6.0]}]}


def test_the_call_confidence_overrides_the_module_one():
    out = _detector(confidence=0.5)._extract_detection(_inference(("dog", 0.2)), confidence=0.1)
    assert out is not None and [d["class_name"] for d in out["detection"]] == ["dog"]


def test_no_result_is_none():
    assert _detector()._extract_detection(None) is None
    assert _detector()._extract_detection({"result": {}}) is None


@pytest.mark.parametrize("model_type", ["object_detection", "constrained_object_detection"])
def test_draw_bounding_boxes_returns_an_image(model_type: str):
    detector = _detector(model_type=model_type)
    detections = detector._extract_detection(_inference(("cat", 0.9)))
    assert isinstance(detector.draw_bounding_boxes(Image.new("RGB", (32, 32)), detections), Image.Image)
    assert detector.draw_bounding_boxes(None, detections) is None
