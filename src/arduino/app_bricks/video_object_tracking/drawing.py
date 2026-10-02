# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Drawing of the crossing line and of the watched area on the frames of the video stream."""

import cv2
import numpy as np

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2
AREA_COLOR = (0, 255, 255)


def draw_crossing_line(frame: np.ndarray, line: tuple[int, int, int, int]) -> np.ndarray:
    """The frame with the straight line through the two points of `line` drawn from edge to edge, in place.

    Args:
        frame (np.ndarray): HxWx3 BGR frame.
        line (tuple[int, int, int, int]): Two points of the line, x1, y1, x2, y2, in frame coordinates.

    Returns:
        np.ndarray: The same frame, drawn on.
    """
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return frame
    height, width = frame.shape[:2]
    beyond_frame = (width + height) / max(abs(dx), abs(dy))
    start = (round(x1 - dx * beyond_frame), round(y1 - dy * beyond_frame))
    end = (round(x1 + dx * beyond_frame), round(y1 + dy * beyond_frame))
    inside, start, end = cv2.clipLine((0, 0, width, height), start, end)
    if inside:
        cv2.line(frame, start, end, LINE_COLOR, LINE_THICKNESS)
    return frame


def draw_area(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """The frame with the closed outline of the polygon drawn on it, in place.

    Args:
        frame (np.ndarray): HxWx3 BGR frame.
        polygon (np.ndarray): The vertices, x and y in frame coordinates, one per row.

    Returns:
        np.ndarray: The same frame, drawn on.
    """
    cv2.polylines(frame, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], True, AREA_COLOR, LINE_THICKNESS)
    return frame
