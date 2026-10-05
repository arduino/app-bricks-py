# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from PIL import Image

from arduino.app_bricks.camera_code_detection import Detection, draw_bounding_box

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def test_draw_bounding_box_outlines_the_code():
    frame = Image.new("RGB", (200, 150))
    corners = np.array([[20, 40], [120, 40], [120, 110], [20, 110]], dtype=int)

    annotated = draw_bounding_box(frame, Detection("hello", "QRCODE", corners))

    assert annotated is frame
    assert annotated.getpixel((70, 40)) == GREEN
    assert annotated.getpixel((120, 75)) == GREEN
    assert annotated.getpixel((70, 110)) == GREEN
    assert annotated.getpixel((20, 75)) == GREEN
    assert annotated.getpixel((70, 75)) == BLACK


@pytest.mark.parametrize(
    "coords",
    [None, np.zeros((3, 2), dtype=int), [[20, 40], [120, 40], [120, 110], [20, 110]]],
    ids=["missing", "wrong-shape", "not-an-array"],
)
def test_draw_bounding_box_returns_the_frame_untouched_on_invalid_coordinates(coords, capsys):
    frame = Image.new("RGB", (200, 150))

    annotated = draw_bounding_box(frame, Detection("hello", "QRCODE", coords))

    assert annotated is frame
    assert annotated.getbbox() is None
    assert "Invalid or missing coordinates" in capsys.readouterr().out
