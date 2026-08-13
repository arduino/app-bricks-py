# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Built-in pose classification: embedding, k-NN and the temporal layer.

One person's 17 keypoints in, per-class probabilities and stable enter/exit
events out. The reference database ships with the brick (assets/pose_classifier.npz)
together with the dials it was tuned with.
"""

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

"""Names of the 17 body keypoints detected for each person, in model output order."""
KEYPOINT_NAMES: tuple[str, ...] = (
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
)

IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Joint pairs whose signed (dx, dy) vectors form the embedding.
EMBEDDING_PAIRS = (
    ("left_shoulder", "left_wrist"),
    ("right_shoulder", "right_wrist"),
    ("nose", "left_wrist"),
    ("nose", "right_wrist"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_hip", "left_wrist"),
    ("right_hip", "right_wrist"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_hip", "left_ankle"),
    ("right_hip", "right_ankle"),
    ("left_wrist", "right_wrist"),
    ("left_ankle", "right_ankle"),
)

EMBEDDING_SIZE = len(EMBEDDING_PAIRS) * 2 + 2  # pairs (dx, dy) + shoulder-center vector

EMBEDDING_JOINTS = tuple(sorted({name for pair in EMBEDDING_PAIRS for name in pair}))

# A joint reported beyond the frame bounds (plus this tolerance of the frame
# size) is extrapolated, not observed: the pose model's offset refinement can
# place occluded limbs far outside the image.
OUT_OF_FRAME_TOLERANCE = 0.1

# Live-frame gate on the normalization anchors only: weak non-anchor joints
# are still usable evidence, while a discarded frame stalls the temporal layer.
ANCHOR_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
MIN_ANCHOR_SCORE = 0.2

_METRICS = ("euclidean", "cosine", "manhattan", "seuclidean")
_VOTE_WEIGHTINGS = ("uniform", "distance")


def normalize_pose(xy: np.ndarray) -> np.ndarray | None:
    """Translate the skeleton to the hip center and scale it by the torso size.

    Args:
        xy: array (17, 2) of pixel coordinates in KEYPOINT_NAMES order.

    Returns:
        Normalized copy (hip center at origin, torso length == 1), or None if
        the pose is degenerate (torso collapsed, e.g. bad detection).
    """
    xy = np.asarray(xy, dtype=np.float32)
    hip_center = (xy[IDX["left_hip"]] + xy[IDX["right_hip"]]) / 2.0
    shoulder_center = (xy[IDX["left_shoulder"]] + xy[IDX["right_shoulder"]]) / 2.0
    torso = float(np.linalg.norm(shoulder_center - hip_center))
    if torso < 1e-3:
        return None
    return (xy - hip_center) / torso


def embed(xy_norm: np.ndarray) -> np.ndarray:
    """Build the feature vector from a normalized skeleton.

    Args:
        xy_norm: array (17, 2) as returned by normalize_pose().

    Returns:
        Feature vector of shape (EMBEDDING_SIZE,): signed (dx, dy) per pair,
        plus the shoulder-center vector (torso orientation; hips are at origin).
    """
    feats = np.empty(EMBEDDING_SIZE, dtype=np.float32)
    i = 0
    for a, b in EMBEDDING_PAIRS:
        feats[i : i + 2] = xy_norm[IDX[b]] - xy_norm[IDX[a]]
        i += 2
    feats[i : i + 2] = (xy_norm[IDX["left_shoulder"]] + xy_norm[IDX["right_shoulder"]]) / 2.0
    return feats


class PoseKNN:
    """k-NN pose classifier with distance-based rejection ("unknown").

    metric: "euclidean", "cosine", "manhattan" or "seuclidean" (euclidean with
    per-feature std scaling over the calibration rows). vote_weighting:
    "uniform" or "distance" (votes scale with 1/distance). The rejection
    threshold is calibrated in the chosen metric's space.

    Constructor defaults mirror the shipped database's tuned dials; every
    production path still passes the dials explicitly.
    """

    def __init__(
        self,
        k: int = 19,
        reject_factor: float = 1.5,
        metric: str = "seuclidean",
        vote_weighting: str = "distance",
    ):
        if metric not in _METRICS:
            raise ValueError(f"unknown metric {metric!r} (use one of {_METRICS})")
        if vote_weighting not in _VOTE_WEIGHTINGS:
            raise ValueError(f"unknown vote_weighting {vote_weighting!r} (use one of {_VOTE_WEIGHTINGS})")
        self.k = k
        self.reject_factor = reject_factor
        self.metric = metric
        self.vote_weighting = vote_weighting
        self._db: np.ndarray | None = None  # (N, F), already in the metric's space
        self._scale: np.ndarray | None = None  # per-feature std, seuclidean only
        self._labels: np.ndarray | None = None  # (N,) of str
        self.classes: tuple[str, ...] = ()
        self.reject_distance: float = np.inf

    @staticmethod
    def _unit(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=-1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)

    def _to_metric_space(self, rows: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return self._unit(rows)
        if self.metric == "seuclidean":
            return rows / self._scale
        return rows

    def _distances(self, query: np.ndarray) -> np.ndarray:
        query = self._to_metric_space(query)
        if self.metric == "cosine":
            return 1.0 - self._db @ query
        if self.metric == "manhattan":
            return np.abs(self._db - query).sum(axis=1)
        return np.linalg.norm(self._db - query, axis=1)

    def _pairwise_nn_distance(self, calib: np.ndarray) -> np.ndarray:
        """Leave-one-out nearest-neighbor distance for the calibration rows."""
        if self.metric == "cosine":
            dists = 1.0 - calib @ calib.T
        elif self.metric == "manhattan":
            # blockwise: the full (N, N, F) broadcast would need gigabytes
            nn = np.empty(len(calib), dtype=np.float64)
            for start in range(0, len(calib), 128):
                block = calib[start : start + 128]
                block_dists = np.abs(block[:, None, :] - calib[None, :, :]).sum(axis=2)
                for row in range(len(block)):
                    block_dists[row, start + row] = np.inf
                nn[start : start + len(block)] = block_dists.min(axis=1)
            return nn
        else:
            # gram trick: same reason, and float64 keeps the subtraction stable
            sq = np.sum(calib.astype(np.float64) ** 2, axis=1)
            gram = calib.astype(np.float64) @ calib.astype(np.float64).T
            dists = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0))
        np.fill_diagonal(dists, np.inf)
        return dists.min(axis=1)

    def fit(self, embeddings: np.ndarray, labels: list[str], calibration_mask: np.ndarray | None = None) -> None:
        """Store the reference database and calibrate the rejection distance.

        The rejection distance is reject_factor times the 95th percentile of the
        leave-one-out nearest-neighbor distance inside the database: a query whose
        median top-k distance exceeds it does not look like anything we know.

        calibration_mask selects the rows the threshold is calibrated on: pass
        the real-example mask when the database contains augmented copies, whose
        artificial density would otherwise shrink the percentile.
        """
        db = np.asarray(embeddings, dtype=np.float32)
        mask = None if calibration_mask is None else np.asarray(calibration_mask, dtype=bool)
        if self.metric == "seuclidean":
            calib_raw = db if mask is None else db[mask]
            self._scale = np.maximum(calib_raw.std(axis=0), 1e-6).astype(np.float32)
        self._db = self._to_metric_space(db)
        self._labels = np.asarray(labels)
        self.classes = tuple(sorted(set(labels)))

        calib = self._db if mask is None else self._db[mask]
        nn_dists = self._pairwise_nn_distance(calib)
        self.reject_distance = self.reject_factor * float(np.percentile(nn_dists, 95))

    def classify(self, embedding: np.ndarray, label_weights: dict[str, float] | None = None) -> dict[str, float]:
        """Return per-class probabilities for one embedding.

        Probability of a class = its share of the k nearest neighbors' votes:
        the neighbor's label weight (label_weights, e.g. {"other": 0.6}) times
        the distance factor when vote_weighting is "distance". Neighbors are
        picked by distance alone. All zeros when the query is rejected.
        """
        if self._db is None:
            raise RuntimeError("fit() must be called first")

        d = self._distances(np.asarray(embedding, dtype=np.float32))
        top_idx = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]

        probs = dict.fromkeys(self.classes, 0.0)
        if float(np.median(d[top_idx])) > self.reject_distance:
            return probs

        top_labels = self._labels[top_idx]
        votes = np.ones(len(top_idx), dtype=np.float64)
        if label_weights:
            votes *= np.asarray([label_weights.get(str(lbl), 1.0) for lbl in top_labels], dtype=np.float64)
        if self.vote_weighting == "distance":
            votes /= np.maximum(d[top_idx], 1e-6)
        total = float(votes.sum())
        if total <= 0.0:
            return probs
        for cls in self.classes:
            probs[cls] = float(votes[top_labels == cls].sum()) / total
        return probs


@dataclass
class EmaHysteresis:
    """Turn noisy per-frame probabilities into stable enter/exit events.

    Per-class exponential moving average with thermostat-style thresholds:
    active above enter_threshold, inactive again only below exit_threshold.
    All time constants are in seconds: the caller passes the frame interval
    dt, so behavior does not change with the pipeline frame rate.

    Invalid frames are passed as probs=None; person_present tells them apart:
    - person detected but joints unreadable: the smoothed values freeze;
    - person not detected: freeze for grace_seconds, then decay as zeros.
    """

    classes: tuple[str, ...]
    smoothing_tau: float = 0.31  # seconds
    enter_threshold: float = 0.65
    exit_threshold: float = 0.45
    grace_seconds: float = 0.7
    smoothed: dict[str, float] = field(init=False)
    active: dict[str, bool] = field(init=False)
    _invalid_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.smoothed = dict.fromkeys(self.classes, 0.0)
        self.active = dict.fromkeys(self.classes, False)

    def update(self, probs: dict[str, float] | None, dt: float, person_present: bool = True) -> list[tuple[str, str]]:
        """Feed one frame of probabilities observed dt seconds after the previous one.

        probs=None marks an invalid frame (see class docstring for the two
        person_present cases). Returns [("enter"|"exit", class), ...].
        """
        if probs is None:
            if person_present:
                return []
            self._invalid_time += dt
            if self._invalid_time <= self.grace_seconds:
                return []
            probs = dict.fromkeys(self.classes, 0.0)
        else:
            self._invalid_time = 0.0

        alpha = 1.0 - math.exp(-max(dt, 1e-3) / self.smoothing_tau)
        events = []
        for cls in self.classes:
            p = probs.get(cls, 0.0)
            self.smoothed[cls] = alpha * p + (1.0 - alpha) * self.smoothed[cls]
            if not self.active[cls] and self.smoothed[cls] >= self.enter_threshold:
                self.active[cls] = True
                events.append(("enter", cls))
            elif self.active[cls] and self.smoothed[cls] < self.exit_threshold:
                self.active[cls] = False
                events.append(("exit", cls))
        return events


@functools.lru_cache(maxsize=2)
def load_pose_classifier(path: Path) -> tuple[PoseKNN, dict[str, float] | None, tuple[str, ...]]:
    """Load the shipped reference database and build the classifier it was tuned as.

    The npz carries the examples, the real-example calibration mask and the
    dials (dials_json), so nothing is hand-copied. Returns (fitted classifier,
    label weights for classify(), the pose names on_pose accepts — every class
    except the "other" guards). Cached per path and shared across brick
    instances; treat it as read-only.
    """
    data = np.load(path)
    for name in ("embeddings", "labels", "real", "dials_json"):
        if name not in data:
            raise ValueError(f"{path} is not a pose classifier database: missing {name!r}")
    dials = json.loads(str(data["dials_json"]))
    knn = PoseKNN(
        k=int(dials["k"]),
        reject_factor=float(dials["reject_factor"]),
        metric=str(dials["metric"]),
        vote_weighting=str(dials["vote_weighting"]),
    )
    knn.fit(data["embeddings"], list(data["labels"]), calibration_mask=data["real"])
    other_weight = float(dials.get("other_weight", 1.0))
    label_weights = {"other": other_weight} if other_weight != 1.0 else None
    pose_names = tuple(sorted(str(cls) for cls in set(knn.classes) - {"other"}))
    return knn, label_weights, pose_names
