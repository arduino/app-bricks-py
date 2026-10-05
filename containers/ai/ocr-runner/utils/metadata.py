# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""OCR result metadata assembly."""

from __future__ import annotations

import numpy as np

from utils.bbox_processing import box_4corners, box_xx_yy


def build_metadata(
    results_horizontal: list[tuple[box_xx_yy, str, float]],
    results_free: list[tuple[box_4corners, str, float]],
) -> dict:
    """
    Assemble the OCR result metadata.

    Parameters
    ----------
    results_horizontal
        (box, text, confidence) per axis-aligned detection, box as (xmin, xmax, ymin, ymax).
    results_free
        (box, text, confidence) per slanted detection, box as 4 (x, y) corners.

    Returns
    -------
    metadata : dict
        - 'text': str, every detected string joined by newlines, in reading order.
        - 'detections': list of dicts, each containing:
            - 'text': str
            - 'confidence': float
            - 'bounding_box_xyxy': list [x1, y1, x2, y2] in frame coordinates
            - 'polygon': list of 4 [x, y] vertices in frame coordinates, ordered
              top-left, top-right, bottom-right, bottom-left
            - 'type': str ('horizontal' or 'free')
    """
    detections: list[dict] = []

    for box, text, confidence in results_horizontal:
        x_min, x_max, y_min, y_max = (int(v) for v in box)
        detections.append({
            "text": text,
            "confidence": float(confidence),
            "bounding_box_xyxy": [x_min, y_min, x_max, y_max],
            "polygon": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            "type": "horizontal",
        })

    for corners, text, confidence in results_free:
        points = np.asarray(corners, dtype=np.int32).reshape(4, 2)
        detections.append({
            "text": text,
            "confidence": float(confidence),
            "bounding_box_xyxy": [
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()),
                int(points[:, 1].max()),
            ],
            "polygon": points.tolist(),
            "type": "free",
        })

    detections = sort_reading_order(detections)

    return {
        "text": "\n".join(d["text"] for d in detections),
        "detections": detections,
    }


def sort_reading_order(detections: list[dict]) -> list[dict]:
    """
    Put detections into reading order: top to bottom, then left to right within a line.

    Horizontal and free boxes arrive in two separate lists, and sorting on the box top
    alone scrambles skewed text (on a line slanting upwards the rightmost word has the
    smallest y). So boxes are first grouped into lines by vertical-centre proximity,
    then ordered by x inside each line.

    Parameters
    ----------
    detections
        Detection dicts carrying a 'bounding_box_xyxy' key.

    Returns
    -------
    ordered : list[dict]
        The same dicts, reordered.
    """
    if not detections:
        return detections

    def y_center(det: dict) -> float:
        _, y1, _, y2 = det["bounding_box_xyxy"]
        return (y1 + y2) / 2

    def height(det: dict) -> float:
        _, y1, _, y2 = det["bounding_box_xyxy"]
        return y2 - y1

    # A box belongs to the current line while its centre stays within half a
    # median box height of the line's running centre.
    tolerance = 0.5 * float(np.median([height(d) for d in detections]))

    ordered: list[dict] = []
    line: list[dict] = []
    line_center = 0.0
    for det in sorted(detections, key=y_center):
        if line and abs(y_center(det) - line_center) > tolerance:
            ordered.extend(sorted(line, key=lambda d: d["bounding_box_xyxy"][0]))
            line = []
        line.append(det)
        line_center = float(np.mean([y_center(d) for d in line]))
    ordered.extend(sorted(line, key=lambda d: d["bounding_box_xyxy"][0]))

    return ordered
