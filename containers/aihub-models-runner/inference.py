# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    This is a dummy inference callback.
    It will be replaced with the actual implementation at boot time.
    """

    return rgb_frame, {}
