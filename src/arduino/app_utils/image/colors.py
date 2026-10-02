# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Differences between colors as the eye perceives them."""

import math

import cv2
import numpy as np


def color_difference(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> float:
    """The CIEDE2000 difference between two colors: around 1 is the smallest difference the eye notices, and the
    larger it is, the more different the two colors look.

    Args:
        color_a (tuple[int, int, int]): A BGR color.
        color_b (tuple[int, int, int]): Another BGR color.

    Returns:
        float: The difference, 0 for the same color.
    """
    return _ciede2000(_lab(color_a), _lab(color_b))


def _lab(color: tuple[int, int, int]) -> tuple[float, float, float]:
    """The CIELAB coordinates of a BGR color."""
    l, a, b = cv2.cvtColor(np.array([[color]], np.float32) / 255.0, cv2.COLOR_BGR2LAB)[0, 0]
    return float(l), float(a), float(b)


def _ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """The CIEDE2000 difference between two CIELAB colors, as Sharma, Wu and Dalal give it (2005)."""
    (l1, a1, b1), (l2, a2, b2) = lab1, lab2
    c_mean = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(c_mean**7 / (c_mean**7 + 25**7)))
    a1, a2 = (1 + g) * a1, (1 + g) * a2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1 = math.degrees(math.atan2(b1, a1)) % 360 if c1 else 0.0
    h2 = math.degrees(math.atan2(b2, a2)) % 360 if c2 else 0.0
    if c1 * c2 == 0:
        dh = 0.0
    elif abs(h2 - h1) <= 180:
        dh = h2 - h1
    else:
        dh = h2 - h1 - 360 if h2 > h1 else h2 - h1 + 360
    d_l, d_c = l2 - l1, c2 - c1
    d_h = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(dh / 2))
    l_mean, c_mean = (l1 + l2) / 2, (c1 + c2) / 2
    if c1 * c2 == 0:
        h_mean = h1 + h2
    elif abs(h1 - h2) <= 180:
        h_mean = (h1 + h2) / 2
    else:
        h_mean = (h1 + h2 + 360) / 2 if h1 + h2 < 360 else (h1 + h2 - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(h_mean - 30))
        + 0.24 * math.cos(math.radians(2 * h_mean))
        + 0.32 * math.cos(math.radians(3 * h_mean + 6))
        - 0.20 * math.cos(math.radians(4 * h_mean - 63))
    )
    rotation = -math.sin(math.radians(60 * math.exp(-(((h_mean - 275) / 25) ** 2)))) * 2 * math.sqrt(c_mean**7 / (c_mean**7 + 25**7))
    s_l = 1 + 0.015 * (l_mean - 50) ** 2 / math.sqrt(20 + (l_mean - 50) ** 2)
    s_c = 1 + 0.045 * c_mean
    s_h = 1 + 0.015 * c_mean * t
    return math.sqrt((d_l / s_l) ** 2 + (d_c / s_c) ** 2 + (d_h / s_h) ** 2 + rotation * (d_c / s_c) * (d_h / s_h))
