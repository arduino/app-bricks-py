# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # numpy is imported where it is used: arduino.app_utils imports this module, a module-level import would slow down every app start
    import numpy as np

__all__ = [
    "Frame",
    "FrameDesigner",
    "MODULINO_MATRIX",
    "UNO_Q_MATRIX",
]

# Convenience geometries, as (width, height) tuples, of some Arduino LED matrices. Frame and
# FrameDesigner work with any size: these only save typing, e.g. Frame.empty(*MODULINO_MATRIX).
UNO_Q_MATRIX: tuple[int, int] = (13, 8)
"""(width, height) of the LED matrix built into Arduino UNO Q and VENTUNO Q."""

MODULINO_MATRIX: tuple[int, int] = (12, 8)
"""(width, height) of the Arduino Modulino Matrix, same layout as the UNO R4 WiFi LED matrix."""


class Frame:
    """Represents a brightness matrix for an LED matrix of any size.

    Internally stores a 2-D numpy array of shape (height, width) with integer
    brightness levels in range [0, brightness_levels-1]. The geometry is
    taken from the array: a (8, 13) array is a frame for the UNO Q matrix,
    a (8, 12) array one for the Modulino Matrix.

    Application metadata (a name, a duration, a database id...) is best added
    by composition, wrapping a Frame in your own dataclass, rather than by
    subclassing. Subclassing keeps working: the factories (`from_rows`,
    `empty`, `copy`) return an instance of the subclass when its constructor
    accepts Frame's arguments, a plain Frame otherwise; a subclass whose
    constructor takes extra arguments overrides them to keep its own type.
    """

    def __init__(
        self,
        arr: np.ndarray,
        brightness_levels: int = 256,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Create a Frame from a numpy array.

        Args:
            arr (numpy.ndarray): 2-D numpy array of shape (height, width) with integer values.
            brightness_levels (int): number of brightness levels (default 256).
            width (int | None): expected width; the array must match it when given.
            height (int | None): expected height; the array must match it when given.

        Raises:
            ValueError: on a geometry mismatch or out-of-range values.
            TypeError: on a non integer array.
        """
        import numpy as np

        self.brightness_levels = brightness_levels
        if arr is None:  # pyright: ignore[reportUnnecessaryComparison]  # runtime guard for untyped callers
            raise TypeError("array is not set")
        np_arr = np.asarray(arr)
        self._validate_array_input(np_arr)
        actual_height, actual_width = np_arr.shape
        expected = (
            actual_height if height is None else int(height),
            actual_width if width is None else int(width),
        )
        if np_arr.shape != expected:
            raise ValueError(f"array must have shape ({expected[0]}, {expected[1]})")
        self._assert_array_in_range(np_arr, self.brightness_levels)
        self._arr = np_arr

    def __repr__(self) -> str:
        """Return the array as representation of the Frame."""
        return self.arr.__repr__()

    # -- geometry and state ---------------------------------------------
    @property
    def width(self) -> int:
        """Number of columns of the frame."""
        return int(self._arr.shape[1])

    @property
    def height(self) -> int:
        """Number of rows of the frame."""
        return int(self._arr.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the (height, width) shape of the frame as a tuple of ints."""
        return self._arr.shape

    @property
    def brightness_levels(self) -> int:
        """Number of discrete brightness levels (2..256): values range in 0..brightness_levels-1."""
        return self._brightness_levels

    @brightness_levels.setter
    def brightness_levels(self, value: int) -> None:
        """Change the number of brightness levels, the current values must fit the new range.

        Raises:
            ValueError: if the value is not in 2..256 or the array has values outside the new range.
        """
        levels = int(value)
        if not 2 <= levels <= 256:
            raise ValueError("brightness_levels must be an integer in 2..256")
        arr = getattr(self, "_arr", None)
        if arr is not None:
            self._assert_array_in_range(arr, levels)
        self._brightness_levels = levels

    @property
    def arr(self) -> np.ndarray:
        """Public read-only view of the internal array.

        Returns a numpy ndarray view with the writeable flag turned off so
        callers cannot mutate the internal storage in-place. Use
        `set_array` to replace the whole array.
        """
        v = self._arr.view()
        try:
            v.flags.writeable = False
        except Exception:
            return self._arr.copy()
        return v

    # -- factory methods ----------------------------------------------
    @classmethod
    def _from_array(cls, arr: np.ndarray, brightness_levels: int) -> Frame:
        """Build a frame of this class from an already validated array (internal constructor path).

        A subclass whose constructor does not accept ``(arr, brightness_levels)``
        gets a plain Frame: it overrides this method, or the public factories,
        to build instances of its own type.
        """
        import inspect

        try:
            inspect.signature(cls).bind(arr, brightness_levels=brightness_levels)
        except (TypeError, ValueError):
            return Frame(arr, brightness_levels=brightness_levels)
        return cls(arr, brightness_levels=brightness_levels)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[list[int]] | Sequence[str] | None,
        brightness_levels: int = 256,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> Frame:
        """Create a Frame from frontend rows.

        The geometry is inferred from the rows (number of rows and length of
        the first row) unless width and height are given, in which case the
        rows must match them.

        Args:
            rows (Sequence[list[int]] | Sequence[str]): Either a list or tuple of rows, each a list of ints, or of
                CSV strings of numeric values. Every row must have the same number of values.
            brightness_levels (int): Number of discrete brightness levels for the
                resulting Frame (2..256).
            width (int | None): expected number of values per row.
            height (int | None): expected number of rows.

        Returns:
            frame: A validated frame, an instance of the class it is called on (see the class docstring).

        Raises:
            ValueError: on missing or malformed rows, or out-of-range values.
        """
        import numpy as np

        brightness_levels = int(brightness_levels)
        if not (2 <= brightness_levels <= 256):
            raise ValueError("brightness_levels must be in 2..256")

        if rows is None:
            raise ValueError("rows missing")
        if height is not None and len(rows) != height:
            raise ValueError(f"rows must be a list of {height} rows")
        if len(rows) == 0:
            raise ValueError("rows must not be empty")

        # Case: comma-separated numeric strings
        if isinstance(rows[0], str):
            parsed: list[list[int]] = []
            expected_width = width
            for i, row in enumerate(rows):
                if not isinstance(row, str):
                    raise ValueError(f"row {i} is not a string")
                parts = [p.strip() for p in row.split(",")]
                if expected_width is None:
                    expected_width = len(parts)
                if len(parts) != expected_width:
                    raise ValueError(f"row {i} must contain {expected_width} comma-separated values")
                try:
                    nums = [int(p) for p in parts]
                except Exception as e:
                    raise ValueError(f"row {i} contains non-integer value: {e}")
                parsed.append(nums)
            np_arr = np.asarray(parsed, dtype=int)

        # Case: list of lists
        else:
            expected_width = width
            # ensure every row is a list, all of the same length
            for i, row in enumerate(rows):
                if not isinstance(row, (list, tuple)):
                    raise ValueError(f"row {i} must be a list of values")
                if expected_width is None:
                    expected_width = len(row)
                if len(row) != expected_width:
                    raise ValueError(f"row {i} must be a list of {expected_width} values")
            np_arr = np.asarray(rows, dtype=int)
            # Validate values are within declared brightness range
            if np.any(np_arr < 0) or np.any(np_arr >= brightness_levels):
                raise ValueError(f"row values must be in 0..{brightness_levels - 1}")

        return cls._from_array(np_arr, brightness_levels)

    @classmethod
    def empty(cls, width: int, height: int, brightness_levels: int = 256) -> Frame:
        """Create a frame with every LED off.

        Args:
            width (int): number of columns.
            height (int): number of rows.
            brightness_levels (int): number of brightness levels (default 256).

        Returns:
            Frame: a blank frame, an instance of the class it is called on (see the class docstring).
        """
        import numpy as np

        return cls._from_array(np.zeros((int(height), int(width)), dtype=np.uint8), brightness_levels)

    def copy(self) -> Frame:
        """Return an independent copy of this frame.

        Returns:
            Frame: a frame of the same class with a copy of the array (see the class docstring).
        """
        return self._from_array(self._arr.copy(), self.brightness_levels)

    def set_value(self, row: int, col: int, value: int) -> None:
        """Set a specific value in the frame array.

        Args:
            row (int): Row index (0-(height-1)).
            col (int): Column index (0-(width-1)).
            value (int): Brightness value to set (0 to brightness_levels-1).
        """
        if not (0 <= row < self.height):
            raise ValueError(f"row index out of range (0-{self.height - 1})")
        if not (0 <= col < self.width):
            raise ValueError(f"column index out of range (0-{self.width - 1})")
        if not (0 <= value < self.brightness_levels):
            raise ValueError(f"value out of range (0-{self.brightness_levels - 1})")
        self._arr[row, col] = value

    def get_value(self, row: int, col: int) -> int:
        """Get a specific value from the frame array.

        Args:
            row (int): Row index (0-(height-1)).
            col (int): Column index (0-(width-1)).
        Returns:
            int: The brightness value at the specified position.
        """
        if not (0 <= row < self.height):
            raise ValueError(f"row index out of range (0-{self.height - 1})")
        if not (0 <= col < self.width):
            raise ValueError(f"column index out of range (0-{self.width - 1})")
        return int(self._arr[row, col])

    def set_array(self, arr: np.ndarray) -> Frame:
        """Set the internal array to a new numpy array in-place.

        The new array must have the shape of the frame; on invalid input the
        frame is left unchanged.

        Args:
            arr (numpy.ndarray): numpy array of shape (height, width) with integer values
        Returns:
            Frame: the same Frame instance after modification.
        """
        import numpy as np

        np_arr = np.asarray(arr).copy()
        self._validate_array_input(np_arr)
        if np_arr.shape != self.shape:
            raise ValueError(f"array must have shape ({self.height}, {self.width})")
        self._assert_array_in_range(np_arr, self.brightness_levels)
        self._arr = np_arr
        return self

    # -- transformations (in-place, fluent) ------------------------------
    def invert(self) -> Frame:
        """Invert brightness values in-place.

        Returns:
            Frame: the same Frame instance after modification.
        """
        maxv = self.brightness_levels - 1
        return self.set_array(maxv - self._arr.astype(int))

    def invert_not_null(self) -> Frame:
        """Invert non-zero brightness values in-place, LEDs that are off stay off.

        Returns:
            Frame: the same Frame instance after modification.
        """
        maxv = self.brightness_levels - 1
        arr = self._arr.astype(int)
        mask = arr > 0
        arr[mask] = maxv - arr[mask]
        return self.set_array(arr)

    def rotate180(self) -> Frame:
        """Rotate the frame by 180 degrees in-place.

        Returns:
            Frame: the same Frame instance after modification.
        """
        import numpy as np

        return self.set_array(np.rot90(self._arr, k=2))

    def flip_horizontally(self) -> Frame:
        """Flip the frame horizontally (left to right) in-place.

        Returns:
            Frame: the same Frame instance after modification.
        """
        import numpy as np

        return self.set_array(np.fliplr(self._arr))

    def flip_vertically(self) -> Frame:
        """Flip the frame vertically (top to bottom) in-place.

        Returns:
            Frame: the same Frame instance after modification.
        """
        import numpy as np

        return self.set_array(np.flipud(self._arr))

    def quantize(self, levels: int) -> Frame:
        """Return a new frame with the values rescaled to another number of brightness levels.

        Values are mapped from [0, brightness_levels-1] to [0, levels-1] and
        rounded to the nearest integer; this frame is not modified.

        Args:
            levels (int): number of brightness levels of the new frame (2..256).

        Returns:
            Frame: a new frame (see the class docstring for its class).

        Raises:
            ValueError: if levels is not in 2..256.
        """
        levels = int(levels)
        if not 2 <= levels <= 256:
            raise ValueError("brightness_levels must be an integer in 2..256")
        return self._from_array(self._scaled(levels - 1), levels)

    def _scaled(self, scale_max: int) -> np.ndarray:
        """Return a new uint8 array with the values mapped from [0, brightness_levels-1] to [0, scale_max]."""
        import numpy as np

        src_max = max(1, self.brightness_levels - 1)
        if src_max == scale_max:
            return self._arr.astype(np.uint8)
        scale = float(scale_max) / float(src_max)
        return (self._arr.astype(float) * scale).round().astype(np.uint8)

    # -- export methods -------------------------------------------------
    # The encoders are named after the formats of the Arduino_LED_Matrix library.
    def to_bytes(self, levels: int | None = None) -> bytes:
        """Return one byte per pixel, row-major: the format of ``Arduino_LED_Matrix::draw()``.

        Args:
            levels (int | None): None to send the values as stored (0..brightness_levels-1),
                or a number of brightness levels (2..256) to quantize the values to first,
                e.g. 256 for a sketch expecting 0..255, 8 for one expecting 0..7.

        Returns:
            bytes: width*height bytes, row after row.

        Raises:
            ValueError: if levels is not None nor in 2..256.
        """
        if levels is None:
            values = self._arr
        else:
            levels = int(levels)
            if not 2 <= levels <= 256:
                raise ValueError("brightness_levels must be an integer in 2..256")
            values = self._scaled(levels - 1)
        return bytes(int(v) for v in values.flatten().tolist())

    def to_words(self) -> list[int]:
        """Return the frame as a binary bitmap packed in 32-bit words.

        A pixel is on when its value is greater than 0. Pixels are packed
        row-major, most significant bit first, in ceil(width*height/32) words,
        the last one zero padded. This is the format of
        ``Arduino_LED_Matrix::loadFrame()`` and ``loadSequence()``, of the
        Arduino LED Matrix online editor and of the Modulino Matrix in
        MonochromaticHorizontal mode: 4 words for the 13x8 UNO Q matrix,
        3 for a 12x8 matrix.

        Returns:
            list[int]: the words, each in 0..2**32-1.
        """
        bits = (self._arr > 0).flatten().tolist()
        words: list[int] = []
        for start in range(0, len(bits), 32):
            chunk = bits[start : start + 32]
            word = 0
            for bit in chunk:
                word = (word << 1) | int(bit)
            words.append(word << (32 - len(chunk)))
        return words

    def to_board_bytes(self) -> bytes:
        """Return the byte buffer (row-major) representing this frame, values scaled to 0..255.

        Deprecated: use `to_bytes()`. Note the difference: this method always
        rescales the values to 0..255 (it equals ``to_bytes(256)``), while
        ``to_bytes()`` sends the values as stored.

        Returns:
            Raw bytes (length height*width) suitable for the firmware.
        """
        warnings.warn(
            "Frame.to_board_bytes() is deprecated, use Frame.to_bytes() (to_bytes(256) for values scaled to 0..255)",
            DeprecationWarning,
            stacklevel=2,
        )
        return bytes(int(v) for v in self._scaled(255).flatten().tolist())

    # -- validation helpers ----------------------------------------------
    @staticmethod
    def _validate_array_input(arr: np.ndarray) -> None:
        """Validate an array: 2-D, at least one row and column, integer dtype.

        Raises:
            TypeError, ValueError on invalid input.
        """
        import numpy as np

        if arr.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if 0 in arr.shape:
            raise ValueError("array must have at least one row and one column")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("array must have integer dtype")

    @staticmethod
    def _assert_array_in_range(arr: np.ndarray, brightness_levels: int) -> None:
        """Assert that array values are within 0..brightness_levels-1.

        Raises:
            ValueError: if any value is out of the allowed range.
        """
        import numpy as np

        maxv = int(brightness_levels) - 1
        if np.any(arr < 0) or np.any(arr > maxv):
            a_min = int(np.min(arr))
            a_max = int(np.max(arr))
            raise ValueError(f"array values out of range 0..{maxv} (found min={a_min}, max={a_max})")

    # -- utility methods -------------------------------------------------
    def rescale_quantized_frame(self, scale_max: int | None = 255) -> np.ndarray:
        """Return a scaled numpy array with values mapped from [0, brightness_levels-1] -> [0, scale_max].

        This does not mutate self.arr; it returns a new numpy array of dtype
        uint8 suitable for sending to the board or for further formatting.
        With scale_max None no scaling is applied and the array itself is returned.

        Deprecated: use `quantize()`, ``quantize(scale_max + 1).arr`` holds the same values.
        """
        warnings.warn(
            "Frame.rescale_quantized_frame() is deprecated, use Frame.quantize()",
            DeprecationWarning,
            stacklevel=2,
        )
        # If no scaling requested, return integer copy
        if scale_max is None:
            return self.arr

        # Enforce board max: scale_max cannot exceed 255 (also min 1)
        if scale_max < 1 or scale_max > 255:
            raise ValueError("scale_max cannot be greater than 255 (board max) or less than 1")

        return self._scaled(scale_max)


class FrameDesigner:
    """Template to create and transform frames for an LED matrix of a given size.

    A designer is bound to a geometry and a number of brightness levels: its
    `empty` and `from_rows` create frames with them, so that every frame of an
    application matches its matrix. The default is the UNO Q matrix
    (`UNO_Q_MATRIX`); e.g. `FrameDesigner(*MODULINO_MATRIX)` designs frames for
    the Modulino Matrix.
    """

    def __init__(self, width: int = 13, height: int = 8, brightness_levels: int = 256) -> None:
        """Bind the designer to a matrix geometry and a number of brightness levels.

        Args:
            width (int): number of columns of the matrix (default 13).
            height (int): number of rows of the matrix (default 8).
            brightness_levels (int): number of brightness levels of the frames (2..256, default 256).

        Raises:
            ValueError: on a non positive size or brightness levels outside 2..256.
        """
        width, height, brightness_levels = int(width), int(height), int(brightness_levels)
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if not 2 <= brightness_levels <= 256:
            raise ValueError("brightness_levels must be an integer in 2..256")
        self.width = width  # led matrix width
        self.height = height  # led matrix height
        self.brightness_levels = brightness_levels

    # -- frame creation --------------------------------------------------
    def empty(self) -> Frame:
        """Create a frame of the designer geometry with every LED off.

        Returns:
            Frame: a blank frame.
        """
        return Frame.empty(self.width, self.height, self.brightness_levels)

    def from_rows(self, rows: Sequence[list[int]] | Sequence[str]) -> Frame:
        """Create a frame from rows, which must match the designer geometry.

        Args:
            rows (Sequence[list[int]] | Sequence[str]): rows as accepted by `Frame.from_rows`.

        Returns:
            Frame: a validated frame.

        Raises:
            ValueError: on rows not matching the geometry, malformed rows or out-of-range values.
        """
        return Frame.from_rows(rows, self.brightness_levels, width=self.width, height=self.height)

    # -- transformations (in-place) ------------------------------------
    # They delegate to the Frame methods of the same name, which can also be called directly.
    def invert(self, frame: Frame) -> Frame:
        """Invert brightness values in-place on a Frame, see `Frame.invert`.

        Args:
            frame (Frame): Frame instance to mutate.

        Returns:
            Frame: the same Frame instance after modification.
        """
        return frame.invert()

    def invert_not_null(self, frame: Frame) -> Frame:
        """Invert non-zero brightness values in-place on a Frame, see `Frame.invert_not_null`.

        Args:
            frame (Frame): Frame instance to mutate.

        Returns:
            Frame: the same Frame instance after modification.
        """
        return frame.invert_not_null()

    def rotate180(self, frame: Frame) -> Frame:
        """Rotate a Frame by 180 degrees in-place, see `Frame.rotate180`.

        Args:
            frame (Frame): Frame instance to mutate.

        Returns:
            Frame: the same Frame instance after modification.
        """
        return frame.rotate180()

    def flip_horizontally(self, frame: Frame) -> Frame:
        """Flip a Frame horizontally in-place, see `Frame.flip_horizontally`.

        Args:
            frame (Frame): Frame instance to mutate.

        Returns:
            Frame: the same Frame instance after modification.
        """
        return frame.flip_horizontally()

    def flip_vertically(self, frame: Frame) -> Frame:
        """Flip a Frame vertically in-place, see `Frame.flip_vertically`.

        Args:
            frame (Frame): Frame instance to mutate.

        Returns:
            Frame: the same Frame instance after modification.
        """
        return frame.flip_vertically()
