# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model and inference constants."""

# Model input resolution (height, width). Posenet MobileNet expects 513x257 RGB.
INPUT_HEIGHT = 513
INPUT_WIDTH = 257

# The model emits heatmaps/offsets/displacements on a grid downsampled by this
# factor relative to the input resolution.
OUTPUT_STRIDE = 16

# Decode parameters
MAX_POSE_DETECTIONS = 10
SCORE_THRESHOLD = 0.25
NMS_RADIUS = 20
MIN_POSE_SCORE = 0.25
MIN_PART_SCORE = 0.1

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

KEYPOINT_IDS = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# Edges traversed when decoding a pose from a root keypoint (a minimum spanning
# tree over the keypoints).
POSE_CHAIN = [
    ("nose", "left_eye"),
    ("left_eye", "left_ear"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("nose", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("nose", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

PARENT_CHILD_TUPLES = [(KEYPOINT_IDS[parent], KEYPOINT_IDS[child]) for parent, child in POSE_CHAIN]

# Edges drawn as the skeleton overlay.
SKELETON_CONNECTIONS = [
    ("left_hip", "left_shoulder"),
    ("left_elbow", "left_shoulder"),
    ("left_elbow", "left_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_shoulder"),
    ("right_elbow", "right_shoulder"),
    ("right_elbow", "right_wrist"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]

SKELETON_CONNECTION_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b]) for a, b in SKELETON_CONNECTIONS]
