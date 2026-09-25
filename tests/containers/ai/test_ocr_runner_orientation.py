# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Rotated-text reading in the ocr-runner (the `rotations` setting): angle parsing, image
rotation, mapping detections back to the original image, and the choice of the
orientation that reads most confidently.

The module is loaded standalone by file path (it only depends on numpy), like the other
ocr-runner tests, so the runner's `utils` package name cannot clash with other runners'.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "containers" / "ai" / "ocr-runner" / "utils" / "orientation.py"
_spec = importlib.util.spec_from_file_location("ocr_orientation", _MODULE_PATH)
orientation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orientation)


# --- parse_rotations -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ([], []),
        (90, [90]),
        ([90, 180, 270], [90, 180, 270]),
        ([270, 90], [270, 90]),  # order is kept
        ([90, 90, 180], [90, 180]),  # duplicates dropped
        ([0, 360, 90], [90]),  # upright is always read: 0 is a no-op
        ([-90], [270]),  # normalized modulo 360
        (["90", 180.0], [90, 180]),  # JSON may carry strings or floats
    ],
)
def test_parse_rotations_normalizes(value, expected):
    assert orientation.parse_rotations(value) == expected


@pytest.mark.parametrize("value", [[45], [30, 90], ["ninety"], [None], [1.5]])
def test_parse_rotations_rejects_non_quarter_turns(value):
    with pytest.raises(ValueError):
        orientation.parse_rotations(value)


# --- rotate_image / points_to_original ----------------------------------------------------


def test_rotate_image_turns_counter_clockwise():
    image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    assert np.array_equal(orientation.rotate_image(image, 90), np.rot90(image))
    assert np.array_equal(orientation.rotate_image(image, 180), image[::-1, ::-1])
    assert orientation.rotate_image(image, 90).shape == (6, 4)
    assert orientation.rotate_image(image, 90).flags["C_CONTIGUOUS"]
    assert orientation.rotate_image(image, 0) is image
    colour = np.zeros((4, 6, 3), dtype=np.uint8)
    assert orientation.rotate_image(colour, 270).shape == (6, 4, 3)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_points_to_original_inverts_rotate_image(angle):
    """A marked pixel found at (x, y) in the rotated image maps back onto the same pixel."""
    height, width = 5, 8
    for y, x in [(0, 0), (0, 7), (4, 0), (2, 3)]:
        image = np.zeros((height, width), dtype=np.uint8)
        image[y, x] = 255
        ry, rx = np.argwhere(orientation.rotate_image(image, angle) == 255)[0]
        # Pixel (rx, ry) covers [rx, rx + 1) x [ry, ry + 1): map its centre.
        mapped = orientation.points_to_original(np.array([rx + 0.5, ry + 0.5]), angle, image.shape)
        assert mapped.tolist() == pytest.approx([x + 0.5, y + 0.5])


def test_detections_to_original_moves_boxes_and_keeps_text_order():
    """Text running top to bottom in a 100x40 (WxH) frame, read on the frame turned by 90
    degrees (40x100): the polygon comes back in frame coordinates, still starting at the
    text's own top-left corner, and the bounding box encloses it."""
    detections = [{"text": "abc", "confidence": 0.9, "bounding_box_xyxy": [10, 20, 30, 90], "polygon": [[10, 20], [30, 20], [30, 90], [10, 90]]}]
    orientation.detections_to_original(detections, 90, (40, 100, 3))
    (det,) = detections
    assert det["polygon"] == [[80, 10], [80, 30], [10, 30], [10, 10]]
    assert det["bounding_box_xyxy"] == [10, 10, 80, 30]
    untouched = [{"text": "a", "confidence": 1.0, "bounding_box_xyxy": [1, 2, 3, 4], "polygon": [[1, 2], [3, 2], [3, 4], [1, 4]]}]
    assert orientation.detections_to_original(untouched, 0, (40, 100)) == untouched


def test_detections_to_original_clips_to_the_frame():
    detections = [{"text": "a", "confidence": 1.0, "bounding_box_xyxy": [0, 0, 5, 5], "polygon": [[-3, -3], [5, -3], [5, 5], [-3, 5]]}]
    orientation.detections_to_original(detections, 180, (10, 10))
    assert all(0 <= v <= 10 for point in detections[0]["polygon"] for v in point)


# --- reading_score / select_orientation --------------------------------------------------


def test_reading_score_weights_by_text_length():
    assert orientation.reading_score([]) == 0.0
    assert orientation.reading_score([("", 0.9)]) == 0.0
    assert orientation.reading_score([("Hello", 1.0), ("2", 0.5)]) == pytest.approx((5 * 1.0 + 0.5) / 6)


def test_select_orientation_keeps_upright_unless_a_rotation_clearly_wins():
    """Measured on rendered texts: the one wrong pick among 168 images was an upright image
    scoring 0.57 upright and 0.60 turned; genuinely rotated text scores far apart (0.50
    upright against 0.89 turned on the reported image)."""
    margin = orientation.ROTATION_MARGIN
    assert orientation.select_orientation({0: 0.57, 90: 0.60}) == 0
    assert orientation.select_orientation({0: 0.50, 90: 0.89}) == 90
    assert orientation.select_orientation({0: 0.5, 90: 0.5 + margin + 0.01}) == 90
    assert orientation.select_orientation({0: 0.9}) == 0


def test_select_orientation_picks_the_best_rotation_earlier_on_ties():
    assert orientation.select_orientation({0: 0.2, 90: 0.6, 180: 0.9, 270: 0.7}) == 180
    assert orientation.select_orientation({0: 0.2, 270: 0.8, 90: 0.8}) == 270


def test_select_orientation_requires_the_upright_score():
    with pytest.raises(ValueError):
        orientation.select_orientation({90: 0.9})
