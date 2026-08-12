# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import math

import cv2
import numpy as np


def resize_pad(
    image: np.ndarray,
    dst_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize and pad image to shape (dst_size[0], dst_size[1]) while preserving aspect ratio.

    Parameters
    ----------
    image
        Input image with shape (H, W) or (H, W, C). dtype can be uint8, float32, etc.
    dst_size
        Desired (height, width).

    Returns
    -------
    rescaled_padded_image : np.ndarray
        Output image with shape (dst_h, dst_w) or (dst_h, dst_w, C).
    scale : float
        Scale factor applied to the original image (same for H and W).
    padding : (int, int)
        (pad_left, pad_top) applied to the resized image.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    src_h, src_w = image.shape[:2]
    dst_h, dst_w = int(dst_size[0]), int(dst_size[1])

    # Compute uniform scale to fit within dst while preserving aspect ratio
    h_ratio = dst_h / src_h
    w_ratio = dst_w / src_w
    scale = min(h_ratio, w_ratio)

    new_h = max(1, math.floor(src_h * scale))
    new_w = max(1, math.floor(src_w * scale))

    interp = cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Compute padding amounts
    pad_total_h = dst_h - new_h
    pad_total_w = dst_w - new_w

    pad_top, pad_bottom = (pad_total_h // 2, pad_total_h - pad_total_h // 2)
    pad_left, pad_right = (pad_total_w // 2, pad_total_w - pad_total_w // 2)

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0.0)

    return padded, scale, (pad_left, pad_top)
