# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import numpy as np
from typing import Any


class Frame:
    """Represents a brightness matrix for the LED matrix.

    Internally stores a numpy array of shape (8, 13) with integer
    brightness levels in range [0, brightness_levels-1].
    """

    def __init__(self, arr: np.ndarray, brightness_levels: int = 256):
        """Create a Frame from a numpy array.

        Args:
            arr (numpy.ndarray): numpy array of shape (8, 13) with integer values
            brightness_levels (int): number of brightness levels (default 255)
        """
        self.height = 8
        self.width = 13
        self.brightness_levels = int(brightness_levels)
        self._arr = arr
        self._validate()

    def __repr__(self):
        """Return the array as representation of the Frame."""
        return self.arr.__repr__()

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept setting of certain attributes.

        Public `arr` is exposed as a read-only property; to replace the
        array please use `set_array(...)` which performs validation and assigns
        to private attribute `_arr`.
        """
        # allow direct assignment for internal storage
        if name == "_arr":
            super().__setattr__(name, value)
            self._validate_array_input()
            self._assert_array_in_range() if getattr(self, "brightness_levels", None) is not None else None
            return
        if name == "brightness_levels":
            super().__setattr__(name, int(value))
            self._validate_brightness_levels()
            if getattr(self, "_arr", None) is not None:
                self._assert_array_in_range()
            return

        super().__setattr__(name, value)

    @property
    def shape(self):
        """Return the (height, width) shape of the frame as a tuple of ints."""
        return self.arr.shape

    @property
    def arr(self) -> np.ndarray:
        """Public read-only view of the internal array.

        Returns a numpy ndarray view with the writeable flag turned off so
        callers cannot mutate the internal storage in-place. Use
        `set_array` to replace the whole array.
        """
        if getattr(self, "_arr", None) is None:
            return None
        v = self._arr.view()
        try:
            v.flags.writeable = False
        except Exception:
            return self._arr.copy()
        return v

    # -- factory methods ----------------------------------------------
    @classmethod
    def from_rows(cls, rows: list[list[int]] | list[str], brightness_levels: int = 256) -> "Frame":
        """Create a Frame from frontend rows.

        Args:
            rows (list[list[int]] | list[str]): Either a list of 8 lists each with 13 ints, or a list of 8
                CSV strings with 13 numeric values each.
            brightness_levels (int): Number of discrete brightness levels for the
                resulting Frame (2..256).

        Returns:
            frame: A validated `Frame` instance.

        Raises:
            ValueError: on malformed rows or out-of-range values.
        """
        brightness_levels = int(brightness_levels)
        if not (2 <= brightness_levels <= 256):
            raise ValueError("brightness_levels must be in 2..256")

        if rows is None:
            raise ValueError("rows missing")
        # Expect exactly 8 rows and 13 columns
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError("rows must be a list of 8 rows")

        # Case: comma-separated numeric strings
        if isinstance(rows[0], str):
            parsed = []
            for i, row in enumerate(rows):
                if not isinstance(row, str):
                    raise ValueError(f"row {i} is not a string")
                parts = [p.strip() for p in row.split(",")]
                if len(parts) != 13:
                    raise ValueError(f"row {i} must contain 13 comma-separated values")
                try:
                    nums = [int(p) for p in parts]
                except Exception as e:
                    raise ValueError(f"row {i} contains non-integer value: {e}")
                parsed.append(nums)
            np_arr = np.asarray(parsed, dtype=int)

        # Case: list of lists
        elif isinstance(rows[0], list):
            # ensure every row is a list of length 13
            for i, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 13:
                    raise ValueError(f"row {i} must be a list of 13 values")
            np_arr = np.asarray(rows, dtype=int)
            # Validate values are within declared brightness range
            if np.any(np_arr < 0) or np.any(np_arr >= brightness_levels):
                raise ValueError(f"row values must be in 0..{brightness_levels - 1}")
        else:
            raise ValueError("unsupported rows format")

        return Frame(arr=np_arr, brightness_levels=brightness_levels)

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
        Args:
            arr (numpy.ndarray): numpy array of shape (height, width) with integer values
        Returns:
            Frame: the same Frame instance after modification.
        """
        prev = self._arr
        try:
            np_arr = np.asarray(arr)
            self._arr = np_arr.copy()
            self._validate()
        except Exception:
            # rollback
            self._arr = prev
            raise
        return self

    # -- export methods -------------------------------------------------
    def to_board_bytes(self) -> bytes:
        """Return the byte buffer (row-major) representing this frame.

        Values are scaled to 0..255 for board consumption.

        Returns:
            Raw bytes (length height*width) suitable for the firmware.
        """
        scaled = self.rescale_quantized_frame(scale_max=255)
        flat = [int(x) for x in scaled.flatten().tolist()]
        return bytes(flat)

    # -- validation helpers ----------------------------------------------
    def _validate(self) -> None:
        """Validate the current Frame instance in-place (internal)."""
        self._validate_brightness_levels()
        self._validate_array_input()
        self._assert_array_in_range()

    def _validate_brightness_levels(self) -> None:
        """Ensure :attr:`brightness_levels` is an int in 2..256.

        Raises:
            ValueError: if the attribute is not a valid integer in range.
        """
        if not (isinstance(self.brightness_levels, int) and 2 <= self.brightness_levels <= 256):
            raise ValueError("brightness_levels must be an integer in 2..256")

    def _validate_array_input(self) -> None:
        """Validate an input array-like of shape (2-D) and integer dtype.

        This method performs validation in-place and does not return the
        provided array. It raises on invalid input.

        Raises:
            TypeError, ValueError on invalid input.
        """
        if getattr(self, "_arr", None) is None:
            raise TypeError("array is not set")
        if not isinstance(self._arr, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if self._arr.ndim != 2:
            raise ValueError("array must be 2-dimensional")
        if self._arr.shape != (self.height, self.width):
            raise ValueError(f"array must have shape ({self.height}, {self.width})")
        if not np.issubdtype(self._arr.dtype, np.integer):
            raise TypeError("array must have integer dtype")

    def _assert_array_in_range(self) -> None:
        """Assert that array values are within 0..brightness_levels-1.

        Raises:
            ValueError: if any value is out of the allowed range.
        """
        if getattr(self, "_arr", None) is None:
            raise TypeError("array is not set")
        maxv = int(self.brightness_levels) - 1
        if np.any(self._arr < 0) or np.any(self._arr > maxv):
            a_min = int(np.min(self._arr))
            a_max = int(np.max(self._arr))
            raise ValueError(f"array values out of range 0..{maxv} (found min={a_min}, max={a_max})")

    # -- utility methods -------------------------------------------------
    def rescale_quantized_frame(self, scale_max: int = 255) -> np.ndarray:
        """Return a scaled numpy array with values mapped from [0, brightness_levels-1] -> [0, scale_max].

        This does not mutate self.arr; it returns a new numpy array of dtype
        uint8 suitable for sending to the board or for further formatting.
        """
        # If no scaling requested, return integer copy
        if scale_max is None:
            return self.arr

        # Enforce board max: scale_max cannot exceed 255 (also min 1)
        if scale_max < 1 or scale_max > 255:
            raise ValueError("scale_max cannot be greater than 255 (board max) or less than 1")

        # Use brightness_levels to determine the input maximum value.
        # brightness_levels is the number of discrete levels (e.g. 256 -> 0..255)
        src_max = max(1, int(self.brightness_levels) - 1)

        # Fast path: if input already uses the target range, just cast
        if src_max == scale_max:
            out = self.arr
            return out.astype(np.uint8)

        # Compute scaling factor from [0..src_max] -> [0..scale_max]
        scale = float(scale_max) / float(src_max) if src_max > 0 else 0.0
        out = (self.arr.astype(float) * scale).round().astype(np.int32)
        return out.astype(np.uint8)


class FrameDesigner:
    """Utilities to create LED matrix frames for the target board.

    FrameDesigner centralizes the LED matrix target specification and
    provides helpers to make transformations of a `Frame` instance.
    """

    def __init__(self):
        """Initialize the FrameDesigner instance with board defaults.

        These attributes define brightness levels used by application helpers.
        """
        self.width = 13  # led matrix width
        self.height = 8  # led matrix height

    # -- transformations (in-place) ------------------------------------
    def invert(self, frame: "Frame") -> "Frame":
        """Invert brightness values in-place on a Frame.
        Args:
            frame (Frame): Frame instance to mutate.
        Returns:
            Frame: the same Frame instance after modification.
        """
        maxv = int(frame.brightness_levels) - 1
        new_arr = (maxv - frame.arr).astype(int)
        frame.set_array(new_arr)
        return frame

    def invert_not_null(self, frame: "Frame") -> "Frame":
        """Invert non-zero brightness values in-place on a Frame.
        Args:
            frame (Frame): Frame instance to mutate.
        Returns:
            Frame: the same Frame instance after modification.
        """
        maxv = int(frame.brightness_levels) - 1
        arr = frame.arr.copy()
        mask = arr > 0
        arr[mask] = (maxv - arr[mask]).astype(int)
        frame.set_array(arr)
        return frame

    def rotate180(self, frame: "Frame") -> "Frame":
        """Rotate a Frame by 180 degrees in-place.
        Args:
            frame (Frame): Frame instance to mutate.
        Returns:
            Frame: the same Frame instance after modification.
        """
        new_arr = np.rot90(frame.arr, k=2)
        frame.set_array(new_arr)
        return frame

    def flip_horizontally(self, frame: "Frame") -> "Frame":
        """Flip a Frame horizontally in-place.
        Args:
            frame (Frame): Frame instance to mutate.
        Returns:
            Frame: the same Frame instance after modification.
        """
        new_arr = np.fliplr(frame.arr)
        frame.set_array(new_arr)
        return frame

    def flip_vertically(self, frame: "Frame") -> "Frame":
        """Flip a Frame vertically in-place.
        Args:
            frame (Frame): Frame instance to mutate.
        Returns:
            Frame: the same Frame instance after modification.
        """
        new_arr = np.flipud(frame.arr)
        frame.set_array(new_arr)
        return frame
