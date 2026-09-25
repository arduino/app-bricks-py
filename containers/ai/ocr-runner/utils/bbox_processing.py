# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""CRAFT score-map -> text box decoding.

Numpy/OpenCV ports of `easyocr.craft_utils.getDetBoxes_core` and
`easyocr.utils.group_text_box`. Both originals are already torch-free; they are
reproduced here so the runner has no dependency on the easyocr package.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# (xmin, xmax, ymin, ymax)
box_xx_yy = tuple[int, int, int, int]

# ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
box_4corners = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def diff(input_list: list[float]) -> float:
    """Spread of a list of values. Port of `easyocr.utils.diff`."""
    return max(input_list) - min(input_list)


def get_det_boxes(
    textmap: np.ndarray,
    linkmap: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
) -> list[np.ndarray]:
    """
    Turn the CRAFT region/affinity score maps into quadrilateral text boxes.

    Port of `easyocr.craft_utils.getDetBoxes_core`. Polygon refinement is omitted
    because the ai-hub-models EasyOCR app runs with poly=False, which makes every
    returned polygon None anyway.

    Parameters
    ----------
    textmap
        [H, W] float region score map.
    linkmap
        [H, W] float affinity score map.
    text_threshold
        Minimum peak region score for a connected component to be kept.
    link_threshold
        Threshold applied to the affinity map when merging characters.
    low_text
        Threshold applied to the region map when segmenting characters.

    Returns
    -------
    boxes : list[np.ndarray]
        [4, 2] float32 corners per detection, in clockwise order starting top-left.
        Coordinates are in score-map space (half the network input resolution).
    """
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # Labeling
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det: list[np.ndarray] = []
    for k in range(1, n_labels):
        # Size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # Make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # Boundary check
        sx = max(sx, 0)
        sy = max(sy, 0)
        ex = min(ex, img_w)
        ey = min(ey, img_h)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # Make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # Align diamond-shape
        box_w, box_h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            left, right = min(np_contours[:, 0]), max(np_contours[:, 0])
            top, bottom = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)

        # Make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)

        det.append(np.array(box))

    return det


def group_text_box(
    polys: list[np.ndarray],
    slope_ths: float = 0.1,
    ycenter_ths: float = 0.5,
    height_ths: float = 0.5,
    width_ths: float = 1.0,
    add_margin: float = 0.05,
    sort_output: bool = True,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """
    Merge per-character boxes into text lines.

    Port of `easyocr.utils.group_text_box`. Boxes whose slant is below slope_ths are
    reinterpreted as axis-aligned boxes and merged line by line; the rest stay as
    free-form parallelograms.

    Parameters
    ----------
    polys
        List of flat 8-element arrays (x1, y1, x2, y2, x3, y3, x4, y4), ordered
        top-left, top-right, bottom-right, bottom-left.
    slope_ths
        Maximum slant for a box to be treated as horizontal.
    ycenter_ths
        Maximum vertical-centre distance (relative to box height) for boxes on one line.
    height_ths
        Maximum relative height difference for boxes merged into one line.
    width_ths
        Maximum horizontal gap (relative to box height) between merged boxes.
    add_margin
        Fractional margin added around each merged box.
    sort_output
        Sort horizontal boxes top to bottom before merging.

    Returns
    -------
    horizontal_list : list[list[float]]
        Merged axis-aligned boxes as [x_min, x_max, y_min, y_max].
    free_list : list[list[list[float]]]
        Slanted boxes as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))

            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # Combine boxes that sit on the same line
    new_box: list[list[float]] = []
    b_height: list[float] = []
    b_ycenter: list[float] = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        elif abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
            b_height.append(poly[5])
            b_ycenter.append(poly[4])
            new_box.append(poly)
        else:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            combined_list.append(new_box)
            new_box = [poly]
    combined_list.append(new_box)

    # Merge each line, left to right
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
            continue

        boxes = sorted(boxes, key=lambda item: item[0])

        merged_box, new_box = [], []
        x_max = 0.0
        for box in boxes:
            if len(new_box) == 0:
                b_height = [box[5]]
                x_max = box[1]
                new_box.append(box)
            elif (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and (
                (box[0] - x_max) < width_ths * (box[3] - box[2])
            ):  # merge boxes
                b_height.append(box[5])
                x_max = box[1]
                new_box.append(box)
            else:
                b_height = [box[5]]
                x_max = box[1]
                merged_box.append(new_box)
                new_box = [box]
        if len(new_box) > 0:
            merged_box.append(new_box)

        for mbox in merged_box:
            if len(mbox) != 1:  # adjacent boxes in the same line
                x_min = min(mbox, key=lambda x: x[0])[0]
                x_max = max(mbox, key=lambda x: x[1])[1]
                y_min = min(mbox, key=lambda x: x[2])[2]
                y_max = max(mbox, key=lambda x: x[3])[3]

                margin = int(add_margin * (min(x_max - x_min, y_max - y_min)))
                merged_list.append([x_min - margin, x_max + margin, y_min - margin, y_max + margin])
            else:  # non adjacent box in the same line
                box = mbox[0]
                margin = int(add_margin * (min(box[1] - box[0], box[3] - box[2])))
                merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])

    return merged_list, free_list
