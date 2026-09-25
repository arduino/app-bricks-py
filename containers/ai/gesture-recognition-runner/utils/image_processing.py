# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np


def apply_batched_affines_to_frame(frame: np.ndarray, affines: list[np.ndarray], output_image_size: tuple[int, int]) -> np.ndarray:
    """
    Generate one image per affine applied to the given frame.
    I/O is numpy since this uses cv2 APIs under the hood.

    Inputs:
        frame: np.ndarray
            Frame on which to apply the affine. Shape is [ H W C ], dtype must be np.byte.
        affines: list[np.ndarray]
            List of 2x3 affine matrices to apply to the frame.
        output_image_size: torch.Tensor
            Size of each output frame.

    Outputs:
        images: np.ndarray
            Computed images. Shape is [B H W C]
    """
    assert (
        frame.dtype == np.byte or frame.dtype == np.uint8  # noqa: PLR1714 Using a set for comparison is not equivalent to using == on both of these individually.
    )  # cv2 does not work correctly otherwise. Don't remove this assertion.

    imgs = []
    for affine in affines:
        img = cv2.warpAffine(frame, affine, output_image_size)
        imgs.append(img)
    return np.stack(imgs)


def apply_affine_to_coordinates(coordinates: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply the given affine matrix to the given coordinates.

    Inputs:
        coordinates: torch.Tensor
            Coordinates on which to apply the affine. Shape is [ ..., 2 ], where 2 == [X, Y]
        affines: torch.Tensor
            Affine matrix to apply to the coordinates.

    Outputs:
        Transformed coordinates. Shape is [ ..., 2 ], where 2 == [X, Y]
    """
    return (affine[:, :2] @ coordinates.T + affine[:, 2:]).T


def compute_vector_rotation(
    vec_start: np.ndarray,
    vec_end: np.ndarray,
    offset_rads: float | np.ndarray = 0,
) -> np.ndarray:
    """
    From the given vector, compute the rotation angle of the vector with an added offset.

    Parameters
    ----------
    vec_start : np.ndarray
        Starting point of the vector. Shape [B, 2] (x, y).
    vec_end : np.ndarray
        Ending point of the vector. Shape [B, 2] (x, y).
    offset_rads : float or np.ndarray
        Offset (in radians) to subtract from the computed rotation.
        Can be a scalar or array broadcastable to shape [B].

    Returns
    -------
    theta : np.ndarray
        Rotation angle in radians. Shape [B].
    """
    # Compute dy, dx
    dy = vec_start[..., 1] - vec_end[..., 1]
    dx = vec_start[..., 0] - vec_end[..., 0]

    # atan2(dy, dx)
    theta = np.arctan2(dy, dx) - offset_rads
    return theta
