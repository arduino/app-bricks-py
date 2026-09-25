# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Reading rotated text: run the whole pipeline on the image turned by each requested
angle, keep the orientation that reads most confidently (`rotation` setting).

The CRNN recognizer only reads horizontal, left-to-right text. EasyOCR's `rotation_info`
deals with that per box: detection runs once on the upright image and every cutout is
also recognized rotated. That was this runner's first implementation, and it reads
vertical text badly. The detector only merges characters into lines horizontally, so on
a page turned by 90 degrees every word becomes a box of its own; short words ("to",
"Nice") give nearly square cutouts whose upright reading is a confident single glyph
("3" at 0.68, "2" at 0.93) that the rotated reading cannot beat; and reading order is
computed in the upright frame, so the words of different lines interleave. Measured on
rendered two-line texts (7 fonts, 2 sizes, 84 images, turned 90 degrees clockwise):
50% of the words read correctly per box, 56% with the best per-box selection rule found.

Turning the image instead gives the detector upright text: lines merge, cutouts are
ordinary strips and reading order comes out right. On the same images 84% of the words
read correctly - what the upright originals score - and the right orientation was
picked for 167 of 168 images. The cost is one full pass (detection + recognition) per
extra angle.

Choosing: every orientation is scored with the character-weighted mean confidence of
what it read, and a rotated orientation replaces the upright one only when it scores at
least ROTATION_MARGIN more. The one wrong pick in the measurement above was an upright
image scoring 0.57 upright and 0.60 turned; genuinely rotated text scores far apart
(0.50 upright against 0.89 turned on tests/containers/ai/ocr_runner_images/hey-arduino-cw90.png).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

# The recognizer input is a horizontal strip, so only quarter turns make sense.
VALID_ROTATIONS = (90, 180, 270)

# How much higher a rotated orientation has to score to replace the upright one.
ROTATION_MARGIN = 0.1


def parse_rotations(value: object) -> list[int]:
    """
    Normalize a `rotations` configuration value into a list of angles.

    Parameters
    ----------
    value
        None, an int, or an iterable of ints (degrees). Angles are taken modulo 360;
        0 means the upright reading, which is always performed, and is dropped.

    Returns
    -------
    rotations : list[int]
        Distinct angles among 90, 180, 270, in the order given.

    Raises
    ------
    ValueError
        If an angle is not a multiple of 90, or the value is not a number/iterable.
    """
    if value is None or value == "":
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        value = [value]

    rotations: list[int] = []
    for item in value:
        try:
            angle = int(item) % 360
        except (TypeError, ValueError):
            raise ValueError(f"rotation {item!r} is not an integer number of degrees") from None
        if angle == 0:
            continue
        if angle not in VALID_ROTATIONS:
            raise ValueError(f"rotation {item!r} is not a multiple of 90 degrees (allowed: {', '.join(map(str, VALID_ROTATIONS))})")
        if angle not in rotations:
            rotations.append(angle)
    return rotations


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an [H, W] or [H, W, C] image counter-clockwise by a multiple of 90 degrees (0 returns it unchanged)."""
    if angle % 360 == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=(angle // 90) % 4))


def points_to_original(points: np.ndarray, angle: int, original_shape: Sequence[int]) -> np.ndarray:
    """
    Map (x, y) points from an image rotated with `rotate_image` back to the original image.

    Parameters
    ----------
    points
        [..., 2] coordinates in the rotated image.
    angle
        The angle the image was rotated by (counter-clockwise, multiple of 90).
    original_shape
        Shape of the original, unrotated image ([H, W] or [H, W, C]).

    Returns
    -------
    mapped : np.ndarray
        [..., 2] float coordinates in the original image.
    """
    height, width = original_shape[0], original_shape[1]
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[..., 0], pts[..., 1]
    quarter_turns = (angle // 90) % 4
    if quarter_turns == 0:
        mapped = (x, y)
    elif quarter_turns == 1:  # the original's right edge became the top
        mapped = (width - y, x)
    elif quarter_turns == 2:
        mapped = (width - x, height - y)
    else:  # the original's left edge became the top
        mapped = (y, height - x)
    return np.stack(mapped, axis=-1)


def detections_to_original(detections: list[dict], angle: int, original_shape: Sequence[int]) -> list[dict]:
    """
    Move detection dicts (as built by `utils.metadata.build_metadata`) read on a rotated
    image back into the coordinates of the original image, in place.

    The polygon keeps its vertex order, so it still starts at the top-left corner *of the
    text* as read: for text that runs top to bottom in the original that is its top-right
    corner. `bounding_box_xyxy` is recomputed from the mapped polygon. Coordinates are
    clipped to the original image.
    """
    if angle % 360 == 0:
        return detections
    height, width = original_shape[0], original_shape[1]
    for det in detections:
        points = points_to_original(np.asarray(det["polygon"], dtype=np.float64), angle, original_shape)
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        polygon = np.rint(points).astype(int)
        det["polygon"] = polygon.tolist()
        det["bounding_box_xyxy"] = [
            int(polygon[:, 0].min()),
            int(polygon[:, 1].min()),
            int(polygon[:, 0].max()),
            int(polygon[:, 1].max()),
        ]
    return detections


def reading_score(texts_and_confidences: Iterable[tuple[str, float]]) -> float:
    """Character-weighted mean confidence of a set of readings; 0.0 when nothing was read."""
    chars = 0
    weighted = 0.0
    for text, confidence in texts_and_confidences:
        chars += len(text)
        weighted += len(text) * float(confidence)
    return weighted / chars if chars else 0.0


def select_orientation(scores: dict[int, float]) -> int:
    """
    Pick the orientation to report among the scored ones.

    Parameters
    ----------
    scores
        `reading_score` per angle; must contain 0 (upright).

    Returns
    -------
    angle : int
        0 unless a rotated orientation scores more than ROTATION_MARGIN above upright;
        then the highest-scoring rotated one (the earlier one on ties).
    """
    if 0 not in scores:
        raise ValueError("the upright orientation (0) must always be scored")
    best_rotated = max((angle for angle in scores if angle), key=lambda angle: scores[angle], default=None)
    if best_rotated is not None and scores[best_rotated] > scores[0] + ROTATION_MARGIN:
        return best_rotated
    return 0
