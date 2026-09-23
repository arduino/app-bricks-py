# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from arduino.app_bricks.video_object_tracking.drawing import draw_crossing_line


def test_the_crossing_line_is_drawn_across_the_whole_frame():
    frame = np.zeros((120, 160, 3), np.uint8)
    drawn = draw_crossing_line(frame, (10, 40, 50, 40))
    assert drawn is frame, "drawn in place"
    assert (frame[40] == (255, 0, 255)).all(), "the horizontal magenta line reaches both edges, beyond its two points"
    assert not frame[10].any() and not frame[100].any(), "rows away from the line stay untouched"


def test_a_diagonal_crossing_line_is_extended_to_the_frame_borders():
    frame = np.zeros((120, 160, 3), np.uint8)
    draw_crossing_line(frame, (40, 40, 60, 60))
    assert frame[0, 0].any() and frame[100, 100].any(), "the line y = x runs from the corner, beyond its two points"
    assert not frame[100, 20].any(), "off the line nothing is drawn"


def test_a_crossing_line_outside_the_frame_draws_nothing():
    frame = np.zeros((120, 160, 3), np.uint8)
    draw_crossing_line(frame, (0, 300, 160, 300))
    draw_crossing_line(frame, (50, 50, 50, 50))
    assert not frame.any()
