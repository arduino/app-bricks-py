# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model tensor (de)quantization shared by the AI runner containers."""

from __future__ import annotations

import numpy as np


def dequantize(values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Map an integer model tensor back to float32 using its quantization parameters.

    Tensors without quantization parameters (float models) pass through unchanged,
    only cast to float32.
    """
    zero_points = np.asarray(zero_points)
    scales = np.asarray(scales)
    if zero_points.size == 0 or scales.size == 0:
        return values.astype(np.float32)

    return ((values - np.int32(zero_points)) * np.float64(scales)).astype(np.float32)


def quantize(values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Map a float array onto an integer model tensor using its quantization parameters.

    The result is clipped to the value range of `dtype`.
    """
    quantized = np.round(np.asarray(values, dtype=np.float32) / np.float64(scales)) + np.int32(zero_points)

    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)
