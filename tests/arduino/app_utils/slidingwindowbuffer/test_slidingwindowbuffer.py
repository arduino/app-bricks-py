# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils.slidingwindowbuffer import SlidingWindowBuffer
import numpy as np


def test_push_pull():
    buf = SlidingWindowBuffer(window_size=5, slide_amount=1)

    data = np.array([1, 2, 3, 4, 5])
    buf.push(data)
    assert np.array_equal(buf.pull(), data)

    new_data = np.array([6, 7, 8, 9, 10])
    buf.push(new_data)
    assert np.array_equal(buf.pull(), np.array([2, 3, 4, 5, 6]))
    assert np.array_equal(buf.pull(), np.array([3, 4, 5, 6, 7]))
    assert np.array_equal(buf.pull(), np.array([4, 5, 6, 7, 8]))
    assert np.array_equal(buf.pull(), np.array([5, 6, 7, 8, 9]))
    assert np.array_equal(buf.pull(), np.array([6, 7, 8, 9, 10]))

    new_data = np.array([1, 2, 3, 4, 5])
    buf.push(new_data)
    buf.flush()
    assert not buf.has_data()
    assert len(buf.pull(timeout=0.1)) == 0

    new_data = np.array([11, 12, 13, 14, 15])
    buf.push(new_data)
    new_data = np.array([16, 17, 18, 19, 20])
    buf.push(new_data)
    assert np.array_equal(buf.pull(), np.array([11, 12, 13, 14, 15]))
    assert np.array_equal(buf.pull(), np.array([12, 13, 14, 15, 16]))
    assert np.array_equal(buf.pull(), np.array([13, 14, 15, 16, 17]))
    assert np.array_equal(buf.pull(), np.array([14, 15, 16, 17, 18]))
    assert np.array_equal(buf.pull(), np.array([15, 16, 17, 18, 19]))
    assert np.array_equal(buf.pull(), np.array([16, 17, 18, 19, 20]))


def test_pull_with_array():
    buf = SlidingWindowBuffer(window_size=12, slide_amount=1)

    buf.push(np.array([1, 2, 3]))
    buf.push(np.array([4, 5, 6]))
    buf.push(np.array([7, 8, 9]))
    buf.push(np.array([10, 11, 12]))

    assert buf.has_data()

    pulled_data = buf.pull(timeout=0)
    assert np.array_equal(pulled_data, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))


def test_3d_array():
    buf = SlidingWindowBuffer(window_size=5, slide_amount=1)

    data = np.arange(12).reshape(4, 3)
    buf.push(data)
    assert not buf.has_data()

    buf.push(np.array([[12, 13, 14]]))

    assert buf.has_data()
    assert np.array_equal(buf.pull(), np.arange(15).reshape(5, 3))
