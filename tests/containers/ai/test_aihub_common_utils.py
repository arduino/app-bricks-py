# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Behavior of the shared runner helpers in the base image's aihub package.

These functions were deduplicated out of the gesture-recognition, pose-estimation
and ocr runner containers, which all rely on the exact semantics below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the shared aihub package importable; it lives in its own source-only image, which
# both aihub-litert-models-runner and aihub-onnx-models-runner copy it from.
FRAMEWORK_DIR = Path(__file__).resolve().parents[3] / "containers" / "base" / "aihub-framework"
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

from aihub.draw import draw_box_from_xyxy, draw_connections, draw_points  # noqa: E402
from aihub.image_processing import denormalize_coordinates, resize_pad  # noqa: E402
from aihub.model_io_processing import dequantize, quantize  # noqa: E402


def test_resize_pad_centers_by_default():
    image = np.full((10, 20), 200, dtype=np.uint8)

    padded, scale, (pad_left, pad_top) = resize_pad(image, (16, 16))

    assert padded.shape == (16, 16)
    assert scale == 0.8  # min(16/10, 16/20)
    assert (pad_left, pad_top) == (0, 4)
    # Vertical padding is split evenly around the resized image
    assert not padded[:4].any() and not padded[-4:].any()
    assert padded[4:-4].all()


def test_resize_pad_floats_left_with_pad_value():
    image = np.full((4, 4), 100, dtype=np.float32)

    padded, scale, (pad_left, pad_top) = resize_pad(image, (4, 8), pad_value=7.0, horizontal_float="left")

    assert padded.shape == (4, 8)
    assert scale == 1.0
    assert (pad_left, pad_top) == (0, 0)
    assert (padded[:, :4] == 100).all()
    assert (padded[:, 4:] == 7.0).all()


def test_denormalize_coordinates_in_place():
    coords = np.array([[0.5, 0.5]], dtype=np.float32)

    denormalize_coordinates(coords, (100, 200), scale=2.0, pad=(10, 20))

    # (0.5 * size - pad) / scale, per axis
    assert coords.tolist() == [[20.0, 40.0]]


def test_quantize_dequantize_round_trip():
    values = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    zero_points = np.array([0], dtype=np.int32)
    scales = np.array([1.0 / 255.0], dtype=np.float64)

    quantized = quantize(values, zero_points, scales)
    assert quantized.dtype == np.uint8
    assert quantized.tolist() == [0, 64, 128, 255]

    restored = dequantize(quantized, zero_points, scales)
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, values, atol=1.0 / 255.0)


def test_quantize_clips_to_dtype_range():
    values = np.array([-10.0, 10.0], dtype=np.float32)
    quantized = quantize(values, np.array([0]), np.array([1.0]), dtype=np.int8)

    assert quantized.dtype == np.int8
    assert quantized.tolist() == [-10, 10]

    clipped = quantize(np.array([1000.0], dtype=np.float32), np.array([0]), np.array([1.0]), dtype=np.int8)
    assert clipped.tolist() == [127]


def test_dequantize_passes_float_tensors_through():
    values = np.array([1.5, -2.0], dtype=np.float64)

    restored = dequantize(values, np.array([]), np.array([]))

    assert restored.dtype == np.float32
    assert restored.tolist() == [1.5, -2.0]


def test_draw_helpers_modify_frame_in_place():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    draw_box_from_xyxy(frame, (4, 4), (28, 28), color=(255, 0, 0), size=1)
    assert frame.any()

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    # Array corners (with .item()) are accepted too, as used by the gesture runner
    draw_box_from_xyxy(frame, np.array([4, 4]), np.array([28, 28]), color=(0, 255, 0), size=1, text="hi")
    assert frame.any()

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    draw_points(frame, np.array([[16, 16]], dtype=np.float32), color=(255, 255, 255), size=4)
    assert frame.any()

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    draw_connections(frame, np.array([[4, 4], [28, 28]], dtype=np.float32), [(0, 1)], color=(255, 255, 255), size=2)
    assert frame.any()
