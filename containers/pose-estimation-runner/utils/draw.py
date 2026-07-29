# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np

from utils.constants import (
    KEYPOINT_NAMES,
    MIN_PART_SCORE,
    SKELETON_CONNECTION_INDICES,
)


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

        points: np.ndarray | torch.Tensor
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


def draw_poses(
    frame: np.ndarray,
    pose_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_coords_xy: np.ndarray,
) -> dict:
    """
    Draw the skeleton overlay for each detected pose and build the detection metadata.

    Parameters
    ----------
    frame
        Image array (H, W, C) in RGB, modified in place.
    pose_scores
        Pose confidence scores, shape (max_detections,). Poses are filled in
        order, so the first zero score marks the end of the detections.
    keypoint_scores
        Keypoint confidence scores, shape (max_detections, 17).
    keypoint_coords_xy
        Keypoint coordinates in (x, y) format mapped to the frame space,
        shape (max_detections, 17, 2).

    Returns
    -------
    dict
        Dictionary with a 'poses' key containing one dict per detected person:
        - 'score': float, pose confidence
        - 'keypoints': list of 17 dicts with 'name', 'x', 'y', 'score'
        - 'bounding_box_xyxy': [x1, y1, x2, y2] enclosing the confident keypoints
    """
    height, width = frame.shape[:2]
    poses_metadata = []

    for pose_score, kp_scores, kp_coords in zip(pose_scores, keypoint_scores, keypoint_coords_xy, strict=False):
        if pose_score == 0.0:
            break

        confident = kp_scores >= MIN_PART_SCORE

        edges = [(a, b) for a, b in SKELETON_CONNECTION_INDICES if confident[a] and confident[b]]
        if edges:
            draw_connections(frame, kp_coords, edges, (255, 255, 255), 2)
        if confident.any():
            draw_points(frame, kp_coords[confident], (90, 250, 34), 7, (255, 255, 255))

        # Compute the bounding box from the confident keypoints, clipped to the frame
        bbox_coords = kp_coords[confident] if confident.any() else kp_coords
        x_min, y_min = bbox_coords.min(axis=0)
        x_max, y_max = bbox_coords.max(axis=0)

        poses_metadata.append({
            "score": float(pose_score),
            "keypoints": [
                {
                    "name": KEYPOINT_NAMES[i],
                    "x": int(kp_coords[i][0]),
                    "y": int(kp_coords[i][1]),
                    "score": float(kp_scores[i]),
                }
                for i in range(len(KEYPOINT_NAMES))
            ],
            "bounding_box_xyxy": [
                int(np.clip(x_min, 0, width - 1)),
                int(np.clip(y_min, 0, height - 1)),
                int(np.clip(x_max, 0, width - 1)),
                int(np.clip(y_max, 0, height - 1)),
            ],
        })

    return {"poses": poses_metadata}
