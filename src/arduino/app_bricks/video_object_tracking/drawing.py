# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Drawing of the crossing line and of the watched area on the frames of the video stream, in white dashes with a black
edge so that they show on any background."""

import math

import cv2
import numpy as np

DASH_COLOR = (255, 255, 255)
EDGE_COLOR = (0, 0, 0)
DASH = 8  # pixels of a dash on a 480-row frame, all the sizes follow the frame height
GAP = 6
DASH_THICKNESS = 2
EDGE_THICKNESS = 4

type Point = tuple[float, float]


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
        _draw_dashed(frame, [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))], closed=False)
    return frame


def draw_area(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """The frame with the closed outline of the polygon drawn on it, in place.

    Args:
        frame (np.ndarray): HxWx3 BGR frame.
        polygon (np.ndarray): The vertices, x and y in frame coordinates, one per row.

    Returns:
        np.ndarray: The same frame, drawn on.
    """
    _draw_dashed(frame, [(float(x), float(y)) for x, y in np.asarray(polygon).reshape(-1, 2)], closed=True)
    return frame


def _draw_dashed(frame: np.ndarray, points: list[Point], closed: bool) -> None:
    """Draw the path through the points in dashes, every side starting and ending with one so that the corners show."""
    scale = frame.shape[0] / 480
    dash_thickness = max(1, round(DASH_THICKNESS * scale))
    edge_thickness = max(dash_thickness + 2, round(EDGE_THICKNESS * scale))
    sides = zip(points, points[1:] + points[:1]) if closed else zip(points[:-1], points[1:])
    # A stroke reaches half its thickness past both ends of its dash, so on the path the dashes are shorter and the gaps longer
    dash, gap = max(1.0, DASH * scale - dash_thickness), GAP * scale + dash_thickness
    dashes = [piece for a, b in sides for piece in _dashes(a, b, dash, gap)]
    for thickness, color in ((edge_thickness, EDGE_COLOR), (dash_thickness, DASH_COLOR)):
        for a, b in dashes:
            cv2.fillConvexPoly(frame, _stroke(a, b, thickness), color)


def _dashes(a: Point, b: Point, dash: float, gap: float) -> list[tuple[Point, Point]]:
    """The dashes of the side from a to b, as (start, end) pairs: as many as fit, spread to start at a and end at b."""
    length = math.dist(a, b)
    if length == 0:
        return []
    count = max(1, round((length + gap) / (dash + gap)))
    while count > 1 and count * dash > length:
        count -= 1
    if count == 1:
        return [(a, b)]
    step = (length - dash) / (count - 1)
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    return [((a[0] + ux * i * step, a[1] + uy * i * step), (a[0] + ux * (i * step + dash), a[1] + uy * (i * step + dash))) for i in range(count)]


def _stroke(a: Point, b: Point, thickness: int) -> np.ndarray:
    """The corners of the stroke from a to b, `thickness` pixels across and reaching half of it past both ends."""
    length = math.dist(a, b)
    ux, uy = ((b[0] - a[0]) / length, (b[1] - a[1]) / length) if length else (1.0, 0.0)
    nx, ny = -uy, ux
    before, after = thickness / 2, thickness / 2 - 1  # a span of `thickness` pixels, both end pixels included
    x1, y1 = a[0] - ux * before, a[1] - uy * before
    x2, y2 = b[0] + ux * after, b[1] + uy * after
    corners = [
        (x1 - nx * before, y1 - ny * before),
        (x1 + nx * after, y1 + ny * after),
        (x2 + nx * after, y2 + ny * after),
        (x2 - nx * before, y2 - ny * before),
    ]
    return np.floor(np.array(corners) + 0.5).astype(np.int32)
