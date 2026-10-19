# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

type FormatPlain = type | np.dtype | str
type FormatPacked = tuple[FormatPlain, bool]


def parse_format(format: FormatPlain | FormatPacked, error: type[Exception]) -> tuple[np.dtype, bool]:
    """
    Parse an audio format into its numpy dtype and whether its samples are packed (e.g. 24-bit audio).

    Args:
        format (FormatPlain | FormatPacked): Audio format as one of:
            - Type classes: np.int16, np.float32, np.uint8
            - dtype objects: np.dtype('<i2'), np.dtype('>f4')
            - Strings: 'int16', '<i2', '>f4', 'float32'
            - Tuple of (format, is_packed): to specify if the format is packed (e.g. 24-bit audio)
        error (type[Exception]): Exception raised when the format is not valid.

    Returns:
        tuple[np.dtype, bool]: The format dtype and whether it is packed.

    Raises:
        error: If the format is not valid.
    """
    match format:
        case tuple((plain, is_packed)):
            return _parse_dtype(plain, error), is_packed
        case type() | np.dtype() | str():
            return _parse_dtype(format, error), False
        case _:
            raise error(f"Invalid format: {format}. Expected a numpy dtype, type or string, or a (format, is_packed) tuple")


def _parse_dtype(format: FormatPlain, error: type[Exception]) -> np.dtype:
    """Convert a plain audio format to its numpy dtype."""
    if isinstance(format, str) and format.strip() == "":
        raise error("Format must be a non-empty string or a valid numpy dtype/type or a tuple")
    try:
        return np.dtype(format)
    except TypeError as e:
        raise error(f"Invalid format: {format}") from e
