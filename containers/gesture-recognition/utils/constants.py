# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""Model and inference constants."""

import numpy as np


# Model input options
INPUT_HEIGHT = 256
INPUT_WIDTH = 256

# Detection thresholds
NMS_IOU_THRESHOLD = 0.3
MIN_DETECTOR_BOX_SCORE = 0.95
MIN_LANDMARK_SCORE = 0.5
DETECTOR_SCORE_CLIPPING_THRESHOLD = 20

# Processing parameters
KEYPOINT_ROTATION_VEC_START_IDX = 0
KEYPOINT_ROTATION_VEC_END_IDX = 2
ROTATION_OFFSET_RADS = np.pi / 2
DETECT_BOX_OFFSET_XY = 0.5
DETECT_BOX_SCALE = 2.5

# Gesture labels
GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

# Hand landmark connections for visualization
HAND_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
]
