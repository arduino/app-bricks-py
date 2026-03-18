# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

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


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | tuple[int, int],
    bottom_right: np.ndarray | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: str | None = None,
):
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters
    ----------
        frame: np.ndarray
            np array (H W C x uint8, RGB)

        box: np.ndarray | torch.Tensor
            array (4), where layout is
                [xc, yc, h, w]

        color: tuple[int, int, int]
            Color of drawn points and connection lines (RGB)

        size: int
            Size of drawn points and connection lines RGB channel layout

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


def draw_predictions(
    NHWC_int_numpy_frames: list[np.ndarray],
    batched_selected_landmarks: list[np.ndarray],
    batched_is_right_hand: list[list[bool]],
    batched_gesture_labels: list[list[str]],
    batched_gesture_confidences: list[list[float]],
    landmark_connections: list[tuple[int, int]] | None = None,
) -> dict:
    """
    Override of mediapipe::app.py::MediaPipeApp::draw_outputs
    Also draws whether the detection is a right or left hand.

    Parameters
    ----------
    NHWC_int_numpy_frames
        List of image frames.
    batched_selected_landmarks
        List of landmark arrays for each batch.
    batched_is_right_hand
        List of lists indicating if each hand is right (True) or left (False).
    batched_gesture_labels
        List of lists of gesture labels for each hand.
    batched_gesture_confidences
        List of lists of confidence scores for each hand.
    landmark_connections
        Optional list of landmark connection pairs.

    Returns
    -------
    dict
        Dictionary with 'hands' key containing list of hand metadata dicts.
        Each hand dict contains: hand, gesture_label, confidence, bounding_box_xyxy, landmarks.
    """
    all_hands = []

    for batch_idx in range(len(NHWC_int_numpy_frames)):
        image = NHWC_int_numpy_frames[batch_idx]
        ld = batched_selected_landmarks[batch_idx]
        irh = batched_is_right_hand[batch_idx]
        gestures = batched_gesture_labels[batch_idx]
        confidences = batched_gesture_confidences[batch_idx]
        if ld.size != 0 and len(irh) != 0:
            hands_metadata = draw_landmarks_gesture_label(image, ld, irh, gestures, confidences, landmark_connections=landmark_connections)
            all_hands.extend(hands_metadata)

    return {"hands": all_hands}


def draw_landmarks_gesture_label(
    NHWC_int_numpy_frame: np.ndarray,
    landmarks: np.ndarray,
    is_right_hand: list[bool],
    gesture_labels: list[str],
    gesture_confidences: list[float],
    coords_normalized: bool = False,
    landmark_connections: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Draw landmarks, overlay 'Left/Right: <gesture>' and gesture label near each hand on the image.

    Parameters
    ----------
    NHWC_int_numpy_frame
        Image array (H, W, C) in BGR (OpenCV).
    landmarks
        torch.Tensor of shape (B, N, D) where columns 0,1 are x,y.
    is_right_hand
        list[bool] of length B.
    gesture_labels
        list[str] of length B with resolved labels per hand.
    gesture_confidences
        list[float] of length B with confidence scores per hand.
    coords_normalized
        If True, x,y are in [0,1] and will be converted to pixel coordinates.

    Returns
    -------
    list[dict]
        List of hand metadata dicts, each containing:
        - 'hand': str, either 'right' or 'left'
        - 'gesture': str
        - 'confidence': float, gesture classification confidence
        - 'landmarks': list of shape [21, 3] with x, y, z coordinates
        - 'bounding_box_xyxy': [x1, y1, x2, y2]
    """
    H, W = NHWC_int_numpy_frame.shape[:2]
    hands_metadata = []

    for ldm, irh, gest, conf in zip(landmarks, is_right_hand, gesture_labels, gesture_confidences, strict=False):
        # Convert landmarks to numpy
        xy = ldm[:, [0, 1]]
        # xy = (
        #     xy.detach().cpu().numpy()
        #     if isinstance(xy, torch.Tensor)
        #     else np.asarray(xy)
        # )

        # Convert normalized coords to pixel coords if needed
        xy_px = np.column_stack([xy[:, 0] * W, xy[:, 1] * H]) if coords_normalized else xy

        # Draw landmark points and connections
        if landmark_connections:
            draw_connections(
                NHWC_int_numpy_frame,
                xy_px,
                landmark_connections,
                (255, 255, 255),
                2,
            )
        draw_points(NHWC_int_numpy_frame, xy_px, (90, 250, 34), 7, (255, 255, 255))

        # Compute bounding box from landmarks
        x_min, y_min = xy_px.min(axis=0).astype(int)
        x_max, y_max = xy_px.max(axis=0).astype(int)

        # Prepare label text
        # handedness = "Right" if irh else "Left"
        # label_text = f"{handedness}: {gest}"

        # Use helper for box + text overlay
        # draw_box_from_xyxy(
        #     NHWC_int_numpy_frame,
        #     top_left=(x_min - 20, y_min - 20),
        #     bottom_right=(x_max + 20, y_max + 20),
        #     color=(0, 255, 0),
        #     size=1,
        #     text=label_text,
        # )

        # Collect metadata for this hand
        hands_metadata.append({
            "hand": "right" if irh else "left",
            "gesture": gest,
            "confidence": conf,
            "landmarks": ldm.tolist(),  # Original landmarks (21, 3) with x, y, z as list
            "bounding_box_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
        })

    return hands_metadata
