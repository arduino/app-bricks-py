# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from ai_edge_litert.interpreter import Delegate, load_delegate
import ctypes
from ctypes.util import find_library


def load_qnn_delegate(delegate_lib_path: str = "libQnnTFLiteDelegate.so", delegate_options: dict | None = None) -> list[Delegate] | None:
    """
    Attempt to load the specified delegate library.
    If this fails, instruct to use CPU.
    """
    if _has_library(delegate_lib_path):
        options = delegate_options or {
            "backend_type": "htp",
            "htp_performance_mode": "2",
            "log_level": "1",
        }
        return [load_delegate(delegate_lib_path, options)]
    else:
        print(f"Delegate library '{delegate_lib_path}' not found. Falling back to CPU delegate.")
        return None


def _has_library(name: str) -> bool:
    # Step 1 — Try to locate a platform-specific filename
    lib = find_library(name)
    if lib is None:
        # Might already be a full filename/path — try loading directly
        lib = name

    # Step 2 — Try loading it
    try:
        ctypes.CDLL(lib)
        return True
    except OSError:
        return False
