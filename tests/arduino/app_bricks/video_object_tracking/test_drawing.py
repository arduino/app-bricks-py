# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from arduino.app_bricks.video_object_tracking.drawing import DASH_COLOR, EDGE_COLOR, draw_area, draw_crossing_line

GRAY = (128, 128, 128)


def _frame(height: int = 480, width: int = 640) -> np.ndarray:
    """A gray frame, on which both the white dashes and their black edge show."""
    return np.full((height, width, 3), 128, np.uint8)


def _colors(pixels: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(c) for c in pixel) for pixel in pixels}


def test_the_crossing_line_is_drawn_in_dashes_across_the_whole_frame():
    frame = _frame()
    drawn = draw_crossing_line(frame, (10, 240, 50, 240))

    assert drawn is frame, "drawn in place"
    assert tuple(frame[240, 0]) == DASH_COLOR and tuple(frame[240, 639]) == DASH_COLOR, "a dash at both edges, beyond the two points"
    assert _colors(frame[240]) == {DASH_COLOR, EDGE_COLOR, GRAY}, "dashes, their black edge and the gaps between them"
    assert _colors(frame[238]) == {EDGE_COLOR, GRAY}, "the edge above the dashes"
    assert _colors(frame[230]) == {GRAY} and _colors(frame[250]) == {GRAY}, "rows away from the line stay untouched"


def test_a_diagonal_crossing_line_is_extended_to_the_frame_borders():
    frame = _frame()
    draw_crossing_line(frame, (40, 40, 60, 60))

    assert DASH_COLOR in _colors(frame[0:8, 0:8].reshape(-1, 3)), "the line y = x starts at the corner, beyond its two points"
    assert DASH_COLOR in _colors(frame[472:480, 472:480].reshape(-1, 3)), "and reaches the bottom of the frame"
    assert tuple(frame[100, 20]) == GRAY, "off the line nothing is drawn"


def test_a_crossing_line_outside_the_frame_draws_nothing():
    frame = _frame()
    draw_crossing_line(frame, (0, 900, 160, 900))
    draw_crossing_line(frame, (50, 50, 50, 50))
    assert (frame == 128).all()


def test_the_area_is_drawn_as_a_closed_outline_of_dashes_with_dashed_corners():
    frame = _frame()
    drawn = draw_area(frame, np.array([(100, 100), (300, 100), (300, 250), (100, 250)]))

    assert drawn is frame, "drawn in place"
    for row, column in ((100, 100), (100, 300), (250, 300), (250, 100)):
        assert tuple(frame[row, column]) == DASH_COLOR, ("a dash at every corner", row, column)
    assert tuple(frame[98, 98]) == EDGE_COLOR, "the black edge goes around the corner"
    assert _colors(frame[100, 101:300]) == {DASH_COLOR, EDGE_COLOR, GRAY}, "the top side, with gaps left as they were"
    assert _colors(frame[101:250, 100]) == {DASH_COLOR, EDGE_COLOR, GRAY}, "the closing side too"
    assert tuple(frame[175, 200]) == GRAY, "the inside is not filled"


def test_the_dashes_follow_the_frame_height():
    for height, rows in ((480, 2), (960, 4)):
        frame = _frame(height, height * 4 // 3)
        draw_area(frame, np.array([(100, 100), (300, 100), (300, 250), (100, 250)]))
        white_rows = [row for row in range(90, 111) if (frame[row, 110:290] == 255).all(axis=1).any()]
        assert len(white_rows) == rows, (height, white_rows)
