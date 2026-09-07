# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The data the pose estimation brick hands to an app: keypoints, people and pose events."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Keypoint:
    """One of the 17 body keypoints of a detected person.

    Attributes:
        name (str): Keypoint name, one of `KEYPOINT_NAMES`.
        x (int): Horizontal pixel coordinate in the camera frame.
        y (int): Vertical pixel coordinate in the camera frame.
        score (float): Confidence score in [0.0, 1.0] for this keypoint.
    """

    name: str
    x: int
    y: int
    score: float


@dataclass
class Person:
    """A person detected in a frame.

    Attributes:
        keypoints (dict[str, Keypoint]): The person's 17 keypoints, keyed by
            keypoint name (see `KEYPOINT_NAMES`). Low-confidence keypoints are
            included; filter by their score.
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) box
            enclosing the person's confident keypoints, expanded by the
            configured bbox padding (none by default), in frame coordinates.
    """

    keypoints: dict[str, Keypoint]
    bounding_box_xyxy: tuple[int, int, int, int]


@dataclass
class Pose:
    """A pose classification event for a single person.

    Delivered by `on_pose` callbacks when the tracked person assumes or leaves
    a built-in pose.

    Attributes:
        name (str): Built-in pose name, e.g. "sitting".
        event (Literal["enter", "exit"]): "enter" when the person assumes the
            pose, "exit" when they leave it.
        confidence (float): Classification confidence in [0.0, 1.0] at the event edge.
        keypoints (dict[str, Keypoint]): The person's 17 keypoints, keyed by
            keypoint name (see `KEYPOINT_NAMES`).
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) box
            enclosing the person's confident keypoints, expanded by the
            configured bbox padding (none by default), in frame coordinates.
    """

    name: str
    event: Literal["enter", "exit"]
    confidence: float
    keypoints: dict[str, Keypoint]
    bounding_box_xyxy: tuple[int, int, int, int]


def parse_people(metadata: dict, min_score: float) -> list[Person]:
    """The people of one runner result whose detection score reaches min_score."""
    people = []
    for entry in metadata.get("persons", []):
        if float(entry.get("score", 0.0)) < min_score:
            continue
        keypoints = {
            kp.get("name", ""): Keypoint(name=kp.get("name", ""), x=int(kp.get("x", 0)), y=int(kp.get("y", 0)), score=float(kp.get("score", 0.0)))
            for kp in entry.get("keypoints", [])
        }
        bbox = entry.get("bounding_box_xyxy", [0, 0, 0, 0])
        people.append(Person(keypoints=keypoints, bounding_box_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    return people
