# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Adapts a frame to the model input the way Edge Impulse Studio does, and maps the boxes back.

Resize modes (image_resize_mode of the model): squash resizes ignoring the aspect ratio, fit-shortest
fills the input and center-crops the excess, fit-longest fits the whole frame and adds black bars.
OpenCV resizes with the area interpolation the Edge Impulse SDK uses. The channel order of the frame
is left as it is, the feature encoder reads it from the frame header.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("ei.preprocess")

RESIZE_MODES = {
    "squash": "squash",
    "fit-shortest": "crop",
    "fit-short": "crop",
    "crop": "crop",
    "fit-longest": "letterbox",
    "fit-long": "letterbox",
}
INTERPOLATION = cv2.INTER_AREA
LETTERBOX_COLOR = 0
RGB, BGR = 0, 1  # the color codes of the frame header


@dataclass(frozen=True)
class Transform:
    """Maps between source and model coordinates: model = source * scale + offset."""

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    source_w: int
    source_h: int

    def to_source(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        """Model box (x, y, w, h) to a box in the source frame, clipped to its borders."""
        x0 = min(max((x - self.offset_x) / self.scale_x, 0.0), self.source_w)
        y0 = min(max((y - self.offset_y) / self.scale_y, 0.0), self.source_h)
        x1 = min(max((x + w - self.offset_x) / self.scale_x, 0.0), self.source_w)
        y1 = min(max((y + h - self.offset_y) / self.scale_y, 0.0), self.source_h)
        return x0, y0, x1 - x0, y1 - y0


class Preprocessor:
    """Resizes frames to one model input size in the model's resize mode, reusing one output buffer."""

    def __init__(self, model_w: int, model_h: int, resize_mode: str) -> None:
        self.model_w, self.model_h = model_w, model_h
        self.mode = RESIZE_MODES.get(str(resize_mode).lower())
        if self.mode is None:
            log.warning("Unknown resize mode '%s': using 'squash'", resize_mode)
            self.mode = "squash"
        self._canvas = np.empty((model_h, model_w, 3), np.uint8)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, Transform]:
        """Return the model input of an HxWx3 uint8 frame, same channel order, and its Transform.

        The returned array is a buffer of this preprocessor, valid until the next call.
        """
        h, w = frame.shape[:2]
        resizers = {"squash": self._squash, "crop": self._crop, "letterbox": self._letterbox}
        return resizers[self.mode](frame, w, h)

    def _squash(self, frame: np.ndarray, w: int, h: int) -> tuple[np.ndarray, Transform]:
        mw, mh = self.model_w, self.model_h
        cv2.resize(frame, (mw, mh), dst=self._canvas, interpolation=INTERPOLATION)
        return self._canvas, Transform(mw / w, mh / h, 0.0, 0.0, w, h)

    def _crop(self, frame: np.ndarray, w: int, h: int) -> tuple[np.ndarray, Transform]:
        mw, mh = self.model_w, self.model_h
        scale = max(mw / w, mh / h)
        # Cropping the source first leaves fewer pixels to resize
        crop_w, crop_h = min(w, round(mw / scale)), min(h, round(mh / scale))
        x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
        cv2.resize(frame[y0 : y0 + crop_h, x0 : x0 + crop_w], (mw, mh), dst=self._canvas, interpolation=INTERPOLATION)
        sx, sy = mw / crop_w, mh / crop_h
        return self._canvas, Transform(sx, sy, -x0 * sx, -y0 * sy, w, h)

    def _letterbox(self, frame: np.ndarray, w: int, h: int) -> tuple[np.ndarray, Transform]:
        mw, mh = self.model_w, self.model_h
        scale = min(mw / w, mh / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        x0, y0 = (mw - new_w) // 2, (mh - new_h) // 2
        self._canvas.fill(LETTERBOX_COLOR)
        target = self._canvas[y0 : y0 + new_h, x0 : x0 + new_w]
        if new_w == mw:
            cv2.resize(frame, (new_w, new_h), dst=target, interpolation=INTERPOLATION)  # full rows, the slice is contiguous
        else:
            target[:] = cv2.resize(frame, (new_w, new_h), interpolation=INTERPOLATION)
        return self._canvas, Transform(scale, scale, float(x0), float(y0), w, h)
