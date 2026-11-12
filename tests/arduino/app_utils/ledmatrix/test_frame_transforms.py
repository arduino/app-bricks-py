# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.ledmatrix import MODULINO_MATRIX, Frame, FrameDesigner


def _ramp(width: int = 12, height: int = 8, levels: int = 256) -> Frame:
    arr = (np.arange(width * height).reshape(height, width) % levels).astype(int)
    return Frame(arr, brightness_levels=levels)


def test_transformations_are_in_place_and_fluent():
    f = _ramp()
    orig = f.arr.copy()
    assert f.flip_horizontally() is f
    assert np.array_equal(f.arr, np.fliplr(orig))
    assert f.flip_horizontally().flip_vertically() is f
    assert np.array_equal(f.arr, np.flipud(orig))
    f.flip_vertically().rotate180()
    assert np.array_equal(f.arr, np.rot90(orig, k=2))


def test_invert_and_invert_not_null():
    f = Frame(np.array([[0, 1, 7]]), brightness_levels=8)
    assert f.copy().invert().arr.tolist() == [[7, 6, 0]]
    assert f.invert_not_null().arr.tolist() == [[0, 6, 0]]


def test_quantize_returns_a_new_frame():
    f = Frame(np.array([[0, 128, 255]]), brightness_levels=256)
    q = f.quantize(8)
    assert q is not f
    assert q.brightness_levels == 8
    assert q.arr.tolist() == [[0, 4, 7]]
    assert f.arr.tolist() == [[0, 128, 255]]
    assert q.quantize(256).arr.tolist() == [[0, 146, 255]]
    with pytest.raises(ValueError):
        f.quantize(1)


def test_designer_is_bound_to_a_geometry():
    designer = FrameDesigner(*MODULINO_MATRIX, brightness_levels=8)
    assert (designer.width, designer.height) == (12, 8)
    empty = designer.empty()
    assert empty.shape == (8, 12)
    assert empty.brightness_levels == 8
    f = designer.from_rows([[1] * 12 for _ in range(8)])
    assert f.brightness_levels == 8
    with pytest.raises(ValueError):
        designer.from_rows([[1] * 13 for _ in range(8)])
    with pytest.raises(ValueError):
        designer.from_rows([[1] * 12 for _ in range(7)])


def test_designer_defaults_to_the_uno_q_matrix():
    designer = FrameDesigner()
    assert (designer.width, designer.height, designer.brightness_levels) == (13, 8, 256)
    assert designer.empty().shape == (8, 13)


def test_designer_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        FrameDesigner(0, 8)
    with pytest.raises(ValueError):
        FrameDesigner(13, 8, brightness_levels=1)


def test_designer_transformations_delegate_to_the_frame():
    designer = FrameDesigner(*MODULINO_MATRIX)
    f = _ramp()
    expected = _ramp().flip_horizontally().rotate180().invert()
    assert designer.invert(designer.rotate180(designer.flip_horizontally(f))) is f
    assert np.array_equal(f.arr, expected.arr)
