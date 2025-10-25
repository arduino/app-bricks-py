# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.ledmatrix import Frame


def test_from_rows_list_and_csv():
    """Test Frame.from_rows with list of lists and CSV strings."""
    rows_list = [[0] * 13 for _ in range(8)]
    f1 = Frame.from_rows(rows_list, brightness_levels=8)
    assert f1.arr.shape == (8, 13)
    assert f1.brightness_levels == 8

    rows_csv = [",".join(str(i % 8) for i in range(13)) for _ in range(8)]
    f2 = Frame.from_rows(rows_csv, brightness_levels=8)
    assert f2.arr.shape == (8, 13)
    assert f2.brightness_levels == 8


def test_rescale_and_board_bytes():
    """Test Frame rescale_quantized_frame and to_board_bytes methods."""
    arr = np.zeros((8, 13), dtype=int)
    arr[0, 0] = 7
    f = Frame(arr, brightness_levels=8)
    scaled = f.rescale_quantized_frame(scale_max=255)
    assert scaled.dtype == np.uint8
    assert int(scaled.max()) <= 255
    b = f.to_board_bytes()
    assert isinstance(b, (bytes, bytearray))
    assert len(b) == 8 * 13


def test_re_set_array_valid():
    """Test that re-setting a valid array with arr property fails."""
    f = Frame(np.zeros((8, 13), dtype=int), brightness_levels=8)
    new_arr = np.full((8, 13), 5, dtype=int)
    # Direct assignment must raise AttributeError
    with pytest.raises(AttributeError):
        f.arr = new_arr
    # The returned view must be non-writable
    assert f.arr.flags.writeable is False
    # in-place modification must fail
    with pytest.raises(ValueError):
        f.arr[0, 0] = 1
    # View should not be alias writable of _arr
    orig = f._arr.copy()
    arr_view = f.arr
    with pytest.raises(ValueError):
        arr_view[0, 1] = 2
    assert np.array_equal(f._arr, orig)


def test_modify_arry_in_place():
    """Test that modifying the array in place should fail."""
    f = Frame(np.zeros((8, 13), dtype=int), brightness_levels=8)
    # Modifica in-place deve fallire
    with pytest.raises(ValueError):
        f.arr[0, 0] = 5
    # The view must be updated after set_array
    arr2 = np.ones((8, 13), dtype=int)
    f.set_array(arr2)
    assert np.array_equal(f.arr, arr2)
    # The view must be updated after set_value
    f.set_value(0, 0, 7)
    assert f.arr[0, 0] == 7


def test_set_array_valid_and_invalid():
    f = Frame(np.zeros((8, 13), dtype=int), brightness_levels=8)
    arr_valid = np.full((8, 13), 3, dtype=int)
    f.set_array(arr_valid)
    assert np.array_equal(f.arr, arr_valid)
    # set_array with invalid array must not modify _arr
    arr_invalid = np.full((8, 13), 99, dtype=int)
    arr_before = f._arr.copy()
    with pytest.raises(ValueError):
        f.set_array(arr_invalid)
    assert np.array_equal(f._arr, arr_before)


def test_set_value_and_arr_sync():
    f = Frame(np.zeros((8, 13), dtype=int), brightness_levels=8)
    f.set_value(2, 3, 5)
    assert f.arr[2, 3] == 5
    # set_value out of range
    with pytest.raises(ValueError):
        f.set_value(0, 0, 99)


def test_set_array_invalid():
    """Test that setting invalid arrays raises appropriate exceptions."""
    f = Frame(np.zeros((8, 13), dtype=int), brightness_levels=8)

    # Test setting to non-2D array
    with pytest.raises(ValueError):
        f.set_array(np.zeros((8,), dtype=int))

    # Test setting to wrong shape
    with pytest.raises(ValueError):
        f.set_array(np.zeros((7, 13), dtype=int))

    # Test setting to non-integer dtype
    with pytest.raises(TypeError):
        f.set_array(np.zeros((8, 13), dtype=float))
    # Test setting values out of range
    with pytest.raises(ValueError):
        f.set_array(np.full((8, 13), 10, dtype=int))  # brightness_levels=8 -> max valid value is 7


def test_to_board_bytes():
    """Test that to_board_bytes produces correct output for the test Frame."""
    arr = np.zeros((8, 13), dtype=int)
    arr[0, 0] = 7
    f = Frame(arr, brightness_levels=8)
    b = f.to_board_bytes()
    assert isinstance(b, (bytes, bytearray))
    assert len(b) == 8 * 13
    # Check that the first byte corresponds to the first pixel set to 255 (rescaled value for 7 when 8 levels)
    assert b[0] == 255
