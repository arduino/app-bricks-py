# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from utils.constants import (
    KEYPOINT_NAMES,
    MIN_KEYPOINT_SCORE,
    SKELETON_CONNECTION_INDICES,
)

from aihub.draw import draw_box_from_xyxy, draw_connections, draw_points


def draw_persons(
    frame: np.ndarray,
    person_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_coords_xy: np.ndarray,
    draw_low_confidence_points: bool = True,
    draw_bboxes: bool = False,
    bbox_padding_top: float = 0.0,
    bbox_padding_right: float = 0.0,
    bbox_padding_bottom: float = 0.0,
    bbox_padding_left: float = 0.0,
) -> dict:
    """
    Draw the skeleton overlay for each detected pose and build the detection metadata.

    Parameters
    ----------
    frame
        Image array (H, W, C) in RGB, modified in place.
    person_scores
        Pose confidence scores, shape (max_detections,). Poses are filled in
        order, so the first zero score marks the end of the detections.
    keypoint_scores
        Keypoint confidence scores, shape (max_detections, 17).
    keypoint_coords_xy
        Keypoint coordinates in (x, y) format mapped to the frame space,
        shape (max_detections, 17, 2).
    draw_low_confidence_points
        Mark keypoints below MIN_KEYPOINT_SCORE too, as small hollow dots
        joined by darker lines. Set it to False for an overlay that
        shows only the confident keypoints and connections.
    draw_bboxes
        Draw each person's bounding box (the same one reported in the
        metadata) as a yellow rectangle. Off by default.
    bbox_padding_top
        Expand each person's bounding box upwards by this fraction of its
        height. Applied to the reported metadata and the drawn rectangle alike.
    bbox_padding_right
        Expand each person's bounding box to the right by this fraction of its width.
    bbox_padding_bottom
        Expand each person's bounding box downwards by this fraction of its height.
    bbox_padding_left
        Expand each person's bounding box to the left by this fraction of its width.

    Returns
    -------
    dict
        Dictionary with a 'persons' key containing one dict per detected person:
        - 'score': float, pose confidence
        - 'keypoints': list of 17 dicts with 'name', 'x', 'y', 'score'
        - 'bounding_box_xyxy': [x1, y1, x2, y2] enclosing the confident keypoints
    """
    height, width = frame.shape[:2]
    persons_metadata = []

    for person_score, kp_scores, kp_coords in zip(person_scores, keypoint_scores, keypoint_coords_xy, strict=False):
        if person_score == 0.0:
            break

        confident = kp_scores >= MIN_KEYPOINT_SCORE

        # Compute the bounding box from the confident keypoints, expanded by the
        # configured padding and clipped to the frame
        bbox_coords = kp_coords[confident] if confident.any() else kp_coords
        x_min, y_min = bbox_coords.min(axis=0)
        x_max, y_max = bbox_coords.max(axis=0)
        box_w, box_h = x_max - x_min, y_max - y_min
        y_min -= bbox_padding_top * box_h
        y_max += bbox_padding_bottom * box_h
        x_min -= bbox_padding_left * box_w
        x_max += bbox_padding_right * box_w
        bbox_xyxy = [
            int(np.clip(x_min, 0, width - 1)),
            int(np.clip(y_min, 0, height - 1)),
            int(np.clip(x_max, 0, width - 1)),
            int(np.clip(y_max, 0, height - 1)),
        ]

        if draw_bboxes:
            draw_box_from_xyxy(frame, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (255, 255, 0), 2)

        edges = [(a, b) for a, b in SKELETON_CONNECTION_INDICES if confident[a] and confident[b]]
        if edges:
            draw_connections(frame, kp_coords, edges, (255, 255, 255), 2)

        if draw_low_confidence_points and not confident.all():
            uncertain_edges = [(a, b) for a, b in SKELETON_CONNECTION_INDICES if not (confident[a] and confident[b])]
            if uncertain_edges:
                draw_connections(frame, kp_coords, uncertain_edges, (120, 120, 255), 1)
            draw_points(frame, kp_coords[~confident], (60, 60, 200), 3, (200, 200, 255))

        if confident.any():
            draw_points(frame, kp_coords[confident], (90, 250, 34), 7, (255, 255, 255))

        persons_metadata.append({
            "score": float(person_score),
            "keypoints": [
                {
                    "name": KEYPOINT_NAMES[i],
                    "x": int(kp_coords[i][0]),
                    "y": int(kp_coords[i][1]),
                    "score": float(kp_scores[i]),
                }
                for i in range(len(KEYPOINT_NAMES))
            ],
            "bounding_box_xyxy": bbox_xyxy,
        })

    return {"persons": persons_metadata}
