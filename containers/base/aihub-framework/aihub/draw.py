# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Generic overlay-drawing helpers shared by the AI runner containers."""

from __future__ import annotations

import cv2
import numpy as np


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int | list[int] = 10,
    outline_color: tuple[int, int, int] | None = None,
):
    """
    Draw the given points on the frame.

    Parameters
    ----------
        frame: np.ndarray
            np array (H W C x uint8, RGB)

        points: np.ndarray
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        color: tuple[int, int, int]
            Color of drawn points (RGB)

        size: int
            Size of drawn points

        outline_color: tuple[int, int, int] | None
            Color of the thin outer circle (RGB). If None, no outline is drawn.

    Returns
    -------
        None; modifies frame in place.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    assert isinstance(size, int) or len(size) == len(points)

    # Pre-compute whether size is scalar to avoid repeated checks
    size_is_scalar = isinstance(size, int)

    # Draw outline first if specified, then filled circles
    if outline_color is not None:
        for i, (x, y) in enumerate(points):
            curr_size = size if size_is_scalar else size[i]
            radius = int(curr_size / 2)
            center = (int(x), int(y))
            cv2.circle(frame, center, radius + 1, outline_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(frame, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        for i, (x, y) in enumerate(points):
            curr_size = size if size_is_scalar else size[i]
            radius = int(curr_size / 2)
            cv2.circle(frame, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def draw_connections(
    frame: np.ndarray,
    points: np.ndarray,
    connections: list[tuple[int, int]] | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 1,
):
    """
    Draw connecting lines between the given points on the frame.

    Parameters
    ----------
        frame:
            np array (H W C x uint8, RGB)

        points:
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...
            or
            array (N, 2, 2) where layout is
                [
                  [ # connection 1
                    [ x1, y1 ]
                    [ x2, y2 ]
                  ],
                  [ # connection 2
                    [ x1, y1 ]
                    [ x2, y2 ]
                  ],
                  ...
                ]
                (in this case, connections is unused and can be None)

        connections:
            List of points that should be connected by a line.
            Format is [(src point index, dst point index), ...]

            Unused if points is of shape (N, 2, 2).

        color:
            Color of drawn points (RGB)

        size: int
            Size of drawn connection lines

    Returns
    -------
        None; modifies frame in place.
    """
    point_pairs: list[tuple[tuple[int, int], tuple[int, int]]] | np.ndarray
    if len(points.shape) == 3:
        point_pairs = points
    else:
        assert connections is not None
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)
        point_pairs = [
            (
                (int(points[i][0]), int(points[i][1])),
                (int(points[j][0]), int(points[j][1])),
            )
            for (i, j) in connections
        ]
    cv2.polylines(
        frame,
        np.asarray(point_pairs, dtype=np.int64),
        isClosed=False,
        color=color,
        thickness=size,  # type: ignore[call-overload]
        lineType=cv2.LINE_AA,
    )


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | tuple[int, int],
    bottom_right: np.ndarray | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 2,
    text: str | None = None,
):
    """
    Draw a rectangle using the provided top left / bottom right corners.

    Parameters
    ----------
        frame: np.ndarray
            np array (H W C x uint8, RGB)

        top_left: np.ndarray | tuple[int, int]
            (x, y) coordinates of the top left corner

        bottom_right: np.ndarray | tuple[int, int]
            (x, y) coordinates of the bottom right corner

        color: tuple[int, int, int]
            Color of the rectangle lines (RGB)

        size: int
            Thickness of the rectangle lines

        text: None | str
            Overlay text at the top of the box.

    Returns
    -------
        None; modifies frame in place.
    """
    if not isinstance(top_left, tuple):
        top_left = (int(top_left[0].item()), int(top_left[1].item()))
    if not isinstance(bottom_right, tuple):
        bottom_right = (int(bottom_right[0].item()), int(bottom_right[1].item()))
    cv2.rectangle(frame, top_left, bottom_right, color, size)
    if text is not None:
        cv2.putText(
            frame,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            size,
        )
