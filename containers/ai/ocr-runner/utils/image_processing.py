# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Numpy/OpenCV ports of the EasyOCR image helpers.

The generic helpers (resize_pad, denormalize_coordinates) live in the base
image's `aihub.image_processing` module.
"""

from __future__ import annotations

import cv2
import numpy as np


def four_point_transform(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """
    Warp the parallelogram `rect` out of `image` into an axis-aligned crop.

    Port of `easyocr.utils.four_point_transform`.

    Parameters
    ----------
    image
        Source image.
    rect
        [4, 2] float32 corners, ordered top-left, top-right, bottom-right, bottom-left.

    Returns
    -------
    warped : np.ndarray
        The rectified crop.
    """
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def adjust_contrast(image: np.ndarray, contrast_factor: float) -> np.ndarray:
    """
    Blend an image towards its own mean, matching `torchvision.transforms.functional.adjust_contrast`.

    Parameters
    ----------
    image
        Single-channel float image in the range [0, 1].
    contrast_factor
        > 1 increases contrast, < 1 flattens it.

    Returns
    -------
    adjusted : np.ndarray
        Image of the same shape, clipped back to [0, 1].
    """
    mean = float(np.mean(image, dtype=np.float64))
    blended = contrast_factor * image.astype(np.float32) + (1.0 - contrast_factor) * mean
    return np.clip(blended, 0.0, 1.0)
