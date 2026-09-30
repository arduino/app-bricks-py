# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import colorsys

import pytest

from arduino.app_utils.image.colors import _ciede2000, color_difference

YELLOW = (0, 255, 255)


def _light(hue_degrees: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue_degrees / 360, 0.47, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


@pytest.mark.parametrize(
    "lab1, lab2, difference",
    [
        ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
        ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
        ((50, 0, 0), (50, -1, 2), 2.3669),
        ((50, 2.5, 0), (73, 25, -18), 27.1492),
        ((50, 2.5, 0), (50, 3.1736, 0.5854), 1.0),
    ],
)
def test_the_color_difference_matches_the_published_ciede2000_values(lab1, lab2, difference):
    assert _ciede2000(lab1, lab2) == pytest.approx(difference, abs=1e-4)


def test_a_color_has_no_difference_from_itself():
    assert color_difference((30, 140, 220), (30, 140, 220)) == pytest.approx(0.0, abs=1e-6)


def test_the_difference_does_not_depend_on_the_order():
    assert color_difference(YELLOW, _light(95)) == pytest.approx(color_difference(_light(95), YELLOW))


def test_a_light_green_looks_closer_to_yellow_than_a_light_orange_at_the_same_hue_distance():
    assert color_difference(YELLOW, _light(60 + 35)) < color_difference(YELLOW, _light(60 - 35))
