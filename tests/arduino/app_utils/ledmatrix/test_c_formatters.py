# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from arduino.app_utils.ledmatrix import Frame, c_identifier, sequence_to_c, to_c_array


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("My Frame", "my_frame"),
        ("  Héllo -- World!  ", "h_llo_world"),
        ("3 hearts", "f_3_hearts"),
        ("__x__", "x"),
        ("", "frame"),
        ("!!!", "frame"),
        (None, "frame"),
    ],
)
def test_c_identifier(name, expected):
    assert c_identifier(name) == expected


def test_c_identifier_fallback():
    assert c_identifier("", fallback="animation") == "animation"


def test_to_c_array():
    frame = Frame(np.array([[0, 1, 2], [3, 4, 5]]), brightness_levels=8)
    assert to_c_array(frame, "My Frame") == "uint8_t my_frame[] = {\n  0, 1, 2,\n  3, 4, 5\n};\n"


def test_sequence_to_c():
    on = Frame(np.ones((8, 13), dtype=int), brightness_levels=2)
    off = Frame.empty(13, 8, brightness_levels=2)
    text = sequence_to_c([(on, 500, "all on"), (off, 1000)], name="Blink")
    assert text == (
        "const uint32_t blink[][5] = {\n"
        "    {0xffffffff, 0xffffffff, 0xffffffff, 0xff000000, 500},  // all on\n"
        "    {0x00000000, 0x00000000, 0x00000000, 0x00000000, 1000},\n"
        "};\n"
    )


def test_sequence_to_c_any_size():
    frame = Frame.empty(12, 8)
    assert sequence_to_c([(frame, 100)]).startswith("const uint32_t animation[][4] = {")


def test_sequence_to_c_rejects_empty_or_mixed_sizes():
    with pytest.raises(ValueError):
        sequence_to_c([])
    with pytest.raises(ValueError):
        sequence_to_c([(Frame.empty(13, 8), 100), (Frame.empty(12, 8), 100)])
