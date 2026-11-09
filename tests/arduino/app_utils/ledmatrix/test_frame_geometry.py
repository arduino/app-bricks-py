# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.ledmatrix import MODULINO_MATRIX, UNO_Q_MATRIX, Frame


def test_geometry_is_inferred_from_the_array():
    f = Frame(np.zeros((8, 12), dtype=int))
    assert (f.width, f.height, f.shape) == (12, 8, (8, 12))
    f = Frame(np.zeros((8, 13), dtype=int))
    assert (f.width, f.height) == UNO_Q_MATRIX


def test_explicit_geometry_must_match_the_array():
    Frame(np.zeros((8, 12), dtype=int), width=12, height=8)
    with pytest.raises(ValueError, match=r"array must have shape \(8, 13\)"):
        Frame(np.zeros((8, 12), dtype=int), width=13)
    with pytest.raises(ValueError, match=r"array must have shape \(7, 12\)"):
        Frame(np.zeros((8, 12), dtype=int), height=7)


def test_constructor_rejects_invalid_arrays():
    with pytest.raises(TypeError, match="array is not set"):
        Frame(None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="2-dimensional"):
        Frame(np.zeros((8,), dtype=int))
    with pytest.raises(ValueError, match="at least one row"):
        Frame(np.zeros((0, 13), dtype=int))
    with pytest.raises(TypeError, match="integer dtype"):
        Frame(np.zeros((8, 13), dtype=float))
    with pytest.raises(ValueError, match="brightness_levels must be an integer in 2..256"):
        Frame(np.zeros((8, 13), dtype=int), brightness_levels=1)


def test_geometry_properties_are_read_only():
    f = Frame(np.zeros((8, 13), dtype=int))
    with pytest.raises(AttributeError):
        f.width = 12  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        f.height = 7  # pyright: ignore[reportAttributeAccessIssue]


def test_brightness_levels_setter_validates_and_rechecks_the_range():
    f = Frame(np.full((8, 13), 5, dtype=int), brightness_levels=8)
    f.brightness_levels = 6
    assert f.brightness_levels == 6
    with pytest.raises(ValueError, match="brightness_levels must be an integer in 2..256"):
        f.brightness_levels = 300
    with pytest.raises(ValueError, match=r"array values out of range 0..3 \(found min=5, max=5\)"):
        f.brightness_levels = 4
    # a rejected value leaves the frame unchanged
    assert f.brightness_levels == 6


def test_set_array_keeps_the_frame_shape():
    f = Frame(np.zeros((8, 12), dtype=int))
    with pytest.raises(ValueError, match=r"array must have shape \(8, 12\)"):
        f.set_array(np.zeros((8, 13), dtype=int))
    f.set_array(np.ones((8, 12), dtype=int))
    assert f.get_value(7, 11) == 1
    with pytest.raises(ValueError, match=r"column index out of range \(0-11\)"):
        f.set_value(0, 12, 1)


def test_from_rows_infers_the_geometry():
    f = Frame.from_rows([[0] * 12 for _ in range(8)])
    assert (f.width, f.height) == MODULINO_MATRIX
    f = Frame.from_rows(["0,1,0", "1,0,1"], brightness_levels=2)
    assert f.shape == (2, 3)


def test_from_rows_validates_the_geometry():
    with pytest.raises(ValueError, match="rows must be a list of 8 rows"):
        Frame.from_rows([[0] * 13 for _ in range(7)], width=13, height=8)
    with pytest.raises(ValueError, match="row 1 must be a list of 12 values"):
        Frame.from_rows([[0] * 12, [0] * 13])
    with pytest.raises(ValueError, match="row 0 must be a list of 13 values"):
        Frame.from_rows([[0] * 12 for _ in range(8)], width=13)
    with pytest.raises(ValueError, match="row 1 must contain 3 comma-separated values"):
        Frame.from_rows(["0,0,0", "0,0"])
    with pytest.raises(ValueError, match="rows must not be empty"):
        Frame.from_rows([])


def test_empty_and_copy():
    f = Frame.empty(*MODULINO_MATRIX, brightness_levels=8)
    assert f.shape == (8, 12)
    assert f.brightness_levels == 8
    assert not f.arr.any()
    c = f.copy()
    c.set_value(0, 0, 7)
    assert f.get_value(0, 0) == 0
    assert c.brightness_levels == 8


class _PlainSubclass(Frame):
    pass


class _MetadataSubclass(Frame):
    def __init__(self, name: str, arr: np.ndarray, brightness_levels: int = 256) -> None:
        super().__init__(arr, brightness_levels)
        self.name = name


def test_factories_return_the_subclass_when_its_constructor_allows_it():
    assert type(_PlainSubclass.from_rows([[0, 1]], brightness_levels=2)) is _PlainSubclass
    assert type(_PlainSubclass.empty(2, 2)) is _PlainSubclass
    assert type(_PlainSubclass.empty(2, 2).copy()) is _PlainSubclass


def test_factories_fall_back_to_frame_for_subclasses_with_extra_arguments():
    f = _MetadataSubclass.from_rows([[0, 1]], brightness_levels=2)
    assert type(f) is Frame
    assert type(_MetadataSubclass("x", np.zeros((2, 2), dtype=int)).copy()) is Frame
