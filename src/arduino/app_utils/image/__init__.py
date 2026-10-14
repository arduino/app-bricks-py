# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from .image import *
from .pipeable import PipeableFunction

if TYPE_CHECKING:
    from .adjustments import *

__all__ = [
    "get_image_type",
    "get_image_bytes",
    "draw_bounding_boxes",
    "draw_anomaly_markers",
    "letterbox",
    "resize",
    "flip_h",
    "flip_v",
    "crop",
    "crop_to_aspect_ratio",
    "rotate",
    "adjust",
    "split_channels",
    "greyscale",
    "compress_to_jpeg",
    "compress_to_png",
    "numpy_to_pil",
    "pil_to_numpy",
    "letterboxed",
    "resized",
    "flipped_h",
    "flipped_v",
    "cropped",
    "cropped_to_aspect_ratio",
    "rotated",
    "adjusted",
    "greyscaled",
    "compressed_to_jpeg",
    "compressed_to_png",
    "PipeableFunction",
]

# The adjustments need OpenCV, about 0.5 s to import on the board: they load on first access, so the code that only
# draws or encodes images with PIL does not pay for it. A star import of this package still loads them.
if not TYPE_CHECKING:  # Type checkers resolve the names through the import above, and still reject unknown ones

    def __getattr__(name: str) -> object:
        if name not in __all__:  # Every name of __all__ not bound above comes from the adjustments
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        from . import adjustments

        value = getattr(adjustments, name)
        globals()[name] = value  # Later accesses skip __getattr__
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *__all__})
