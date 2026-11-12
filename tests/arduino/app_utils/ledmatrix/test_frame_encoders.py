# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.ledmatrix import Frame

# X shape of app-bricks-examples (02-led-matrix-frame), 13x8, 7 is on
X_SHAPE = [
    [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
    [0, 0, 7, 7, 0, 0, 0, 0, 0, 7, 7, 0, 0],
    [0, 0, 0, 0, 7, 7, 0, 7, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7, 0, 7, 7, 0, 0, 0, 0],
    [0, 0, 7, 7, 0, 0, 0, 0, 0, 7, 7, 0, 0],
    [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
]

# The heart published for the UNO R4 WiFi matrix, {0x3184a444, 0x44042081, 0x100a0040}
R4_HEART = [
    "..##...##...",
    ".#..#.#..#..",
    ".#...#...#..",
    ".#.......#..",
    "..#.....#...",
    "...#...#....",
    "....#.#.....",
    ".....#......",
]

# Another 12x8 heart, drawn row-major
HEART = [
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


def _decode_words(words: list[int], width: int, height: int) -> list[list[int]]:
    """Unpack MSB-first 32-bit words into rows of 0/1, the inverse of Frame.to_words."""
    bits = "".join(f"{w:032b}" for w in words)
    return [[int(b) for b in bits[r * width : (r + 1) * width]] for r in range(height)]


def test_to_words_x_shape_matches_the_uno_q_sketch():
    """Same words as the X shape in 03-led-matrix-animation-mcu/sketch/frames.h."""
    frame = Frame(np.array(X_SHAPE, dtype=np.uint8))
    assert frame.to_words() == [0xC0198303, 0x600E0070, 0x06C0C198, 0x03000000]


def test_to_words_12x8_matches_the_uno_r4_heart():
    rows = [[1 if c == "#" else 0 for c in row] for row in R4_HEART]
    frame = Frame.from_rows(rows, brightness_levels=2)
    assert frame.to_words() == [0x3184A444, 0x44042081, 0x100A0040]


def test_to_words_12x8_round_trip():
    frame = Frame.from_rows(HEART, brightness_levels=2)
    words = frame.to_words()
    assert words == [0x3184A444, 0x42081100, 0xA0040000]
    assert _decode_words(words, 12, 8) == HEART


def test_to_words_any_value_above_zero_is_on_and_pads_the_last_word():
    frame = Frame(np.array([[0, 1, 255]]), brightness_levels=256)
    assert frame.to_words() == [0b011 << 29]
    assert len(Frame.empty(33, 1).to_words()) == 2


def test_to_bytes_values_as_stored_or_quantized():
    frame = Frame(np.array(X_SHAPE, dtype=np.uint8), brightness_levels=8)
    assert frame.to_bytes() == bytes(v for row in X_SHAPE for v in row)
    assert frame.to_bytes(256) == bytes(255 if v else 0 for row in X_SHAPE for v in row)
    assert frame.to_bytes(8) == frame.to_bytes()
    with pytest.raises(ValueError):
        frame.to_bytes(257)


def test_to_board_bytes_is_deprecated_and_keeps_scaling_to_255():
    frame = Frame(np.array(X_SHAPE, dtype=np.uint8), brightness_levels=8)
    with pytest.warns(DeprecationWarning, match="to_bytes"):
        assert frame.to_board_bytes() == frame.to_bytes(256)
    # with 256 levels the values are sent as stored, as the 02-led-matrix-frame example relies on
    frame = Frame(np.array(X_SHAPE, dtype=np.uint8))
    with pytest.warns(DeprecationWarning):
        assert frame.to_board_bytes() == frame.to_bytes()


def test_rescale_quantized_frame_is_deprecated_and_matches_quantize():
    frame = Frame(np.arange(8).reshape(1, 8), brightness_levels=8)
    with pytest.warns(DeprecationWarning, match="quantize"):
        expected = frame.rescale_quantized_frame(scale_max=255)
    assert np.array_equal(frame.quantize(256).arr, expected)
