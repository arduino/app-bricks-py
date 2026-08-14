# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    This is a dummy inference callback.
    It will be replaced with the actual implementation at boot time.
    """

    return rgb_frame, {}


def apply_config(config: dict) -> None:
    """
    This is a dummy runtime-config handler, optional for a runner to define.
    It will be replaced with the actual implementation at boot time.
    """

    print(f"config ignored by this runner: {config}", flush=True)
