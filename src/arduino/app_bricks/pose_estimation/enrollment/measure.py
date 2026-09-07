# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The numbers behind an enrollment: compose the reference database with the photos' embeddings,
measure whether each pose forms, derive its operating point, table the confusion among poses."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..classifier import PoseKNN
from ..vocabulary import PoseSpec

OTHER = "other"

MIN_PHOTOS_TO_MEASURE = 20
MIN_PHOTOS_TO_ACCEPT = 40
MIN_GROUPS = 5
REFERENCE_THRESHOLD = 0.55
PASS_RECALL = 0.70
CLEANING_SHARE = 0.30
GROUP_DISTANCE = 1.0
STATE_EXIT_GAP = 0.20
ACTION_EXIT_GAP = 0.40
MIN_ACTION_EXIT = 0.10
PERCENTILE_WITHOUT_OTHER = 20
OPERATING_POINT_GRID = np.round(np.arange(0.20, 0.951, 0.025), 3)
F1_TIE = 0.01
COLLISION_WARNING = 0.15
IMBALANCE_RATIO = 3
LEARNING_CURVE_FRACTIONS = (0.25, 0.375, 0.5, 0.75, 1.0)
LEARNING_CURVE_DRAWS = 2
LEARNING_CURVE_SEED = 42
# Own neighbours among the 9 nearest at which a pose fires on 30%, 70% and 90% of its photos (groups held out),
# measured on the built-in poses re-taught from 20 to 240 of their own rows and on the tennis strokes (122 points).
OWN_NEIGHBOURS_FOR_RECALL_30 = 3.0
OWN_NEIGHBOURS_FOR_RECALL_70 = 6.1
OWN_NEIGHBOURS_FOR_RECALL_90 = 7.7
GROUP_DEFINITION = "a group = photos whose skeletons are closer than 1.0 (near-identical frames, or the same pose held still)"


@dataclass(frozen=True)
class Bucket:
    """The photos of one pose (or of `other`), already turned into embeddings.

    Attributes:
        name (str): The pose name, or "other" for the user's own negatives.
        embeddings (np.ndarray): One row per usable photo, shape (n, EMBEDDING_SIZE).
        photos (tuple[str, ...]): The usable photos' paths, one per row, relative to the poses folder.
        found (int): Photos found in the folder, usable or not.
        discarded (tuple[tuple[str, str], ...]): (path, reason) of every photo the structural
            guards refused.
    """

    name: str
    embeddings: np.ndarray
    photos: tuple[str, ...]
    found: int
    discarded: tuple[tuple[str, str], ...] = ()

    @property
    def usable(self) -> int:
        return len(self.photos)


@dataclass(frozen=True)
class Outcome:
    """What the enrollment concluded about one custom pose.

    Attributes:
        name (str): The pose name.
        accepted (bool): Whether the pose passed and is part of the classifier.
        report (str): The full report text.
        enter (float | None): Operating enter threshold, when accepted.
        exit (float | None): Operating exit threshold, when accepted.
    """

    name: str
    accepted: bool
    report: str
    enter: float | None = None
    exit: float | None = None


@dataclass(frozen=True)
class Enrollment:
    """The composed classifier and one outcome per custom pose.

    Attributes:
        knn (PoseKNN): The classifier over the composed database, with the shipped scale and reject distance.
        outcomes (dict[str, Outcome]): Per custom pose name.
        set_aside (dict[str, int]): Shipped rows removed by the cleaning, by their shipped label.
    """

    knn: PoseKNN
    outcomes: dict[str, Outcome]
    set_aside: dict[str, int]


@dataclass(frozen=True)
class _Measure:
    own_shares: np.ndarray
    own_neighbours: float
    n0: float

    @property
    def recall(self) -> float:
        return float((self.own_shares >= REFERENCE_THRESHOLD).mean())


@dataclass(frozen=True)
class _LearningCurve:
    rows: tuple[int, ...]
    recalls: tuple[float, ...]
    n0s: tuple[float, ...]
    verdict: Literal["good", "mixed", "unclear"]


def _nearest(queries: np.ndarray, db: np.ndarray, k: int, skip: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k nearest database rows for each query, both already in the metric space.

    skip lists, per query, the database rows to leave out (the row itself, its group).
    """
    k = min(k, len(db) - (1 if skip else 0))
    idx = np.empty((len(queries), k), dtype=np.int64)
    dist = np.empty((len(queries), k), dtype=np.float64)
    db64 = db.astype(np.float64)
    db_sq = np.sum(db64**2, axis=1)
    for start in range(0, len(queries), 512):
        block = queries[start : start + 512].astype(np.float64)
        d = np.sqrt(np.maximum(np.sum(block**2, axis=1)[:, None] + db_sq[None, :] - 2.0 * block @ db64.T, 0.0))
        if skip:
            for row in range(len(block)):
                d[row, skip[start + row]] = np.inf
        top = np.argpartition(d, k - 1, axis=1)[:, :k]
        idx[start : start + len(block)] = top
        dist[start : start + len(block)] = np.take_along_axis(d, top, axis=1)
    return idx, dist


def _shares(idx: np.ndarray, dist: np.ndarray, labels: np.ndarray, reject_distance: float, classes: tuple[str, ...]) -> np.ndarray:
    """Distance-weighted vote share of each class, per query; all zeros for rejected queries."""
    weights = 1.0 / np.maximum(dist, 1e-6)
    weights[np.median(dist, axis=1) > reject_distance] = 0.0
    total = weights.sum(axis=1)
    out = np.zeros((len(idx), len(classes)))
    top_labels = labels[idx]
    for column, cls in enumerate(classes):
        out[:, column] = np.where(total > 0, (weights * (top_labels == cls)).sum(axis=1) / np.maximum(total, 1e-12), 0.0)
    return out


def group_photos(rows: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Group index of each row: rows closer than GROUP_DISTANCE in the metric space chain into one group."""
    scaled = rows.astype(np.float64) / scale
    sq = np.sum(scaled**2, axis=1)
    close = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * scaled @ scaled.T, 0.0)) <= GROUP_DISTANCE
    group = np.full(len(rows), -1, dtype=np.int64)
    next_group = 0
    for seed in range(len(rows)):
        if group[seed] >= 0:
            continue
        stack = [seed]
        group[seed] = next_group
        while stack:
            for neighbour in np.where(close[stack.pop()] & (group < 0))[0]:
                group[neighbour] = next_group
                stack.append(neighbour)
        next_group += 1
    return group


def _measure(db: np.ndarray, labels: np.ndarray, own_idx: np.ndarray, groups: np.ndarray, k: int, reject: float, cls: str) -> _Measure:
    """Own vote share of each row of a pose, with the row's whole group held out of the database."""
    skip = [own_idx[groups == groups[i]] for i in range(len(own_idx))]
    idx, dist = _nearest(db[own_idx], db, k, skip)
    shares = _shares(idx, dist, labels, reject, (cls,))[:, 0]
    own = float(((labels[idx] == cls) & np.isfinite(dist)).sum(axis=1).mean())
    n0 = len(own_idx) * (k - own) / max(own, 1e-9)
    return _Measure(own_shares=shares, own_neighbours=own, n0=n0)


def _learning_curve(db: np.ndarray, labels: np.ndarray, own_idx: np.ndarray, groups: np.ndarray, k: int, reject: float, cls: str) -> _LearningCurve:
    """n0 and recall at five sizes of the bucket (whole groups drawn), and the verdict on how n0 moves."""
    own = db[own_idx].astype(np.float64)
    foreign = db[labels != cls].astype(np.float64)
    _, fdist = _nearest(own, foreign, k)
    own_sq = np.sum(own**2, axis=1)
    d_own = np.sqrt(np.maximum(own_sq[:, None] + own_sq[None, :] - 2.0 * own @ own.T, 0.0))
    units = np.unique(groups)
    rng = np.random.default_rng(LEARNING_CURVE_SEED)

    def at(selected: np.ndarray) -> tuple[int, float, float]:
        counts, hits = [], []
        for i in np.where(selected)[0]:
            candidates = np.concatenate([d_own[i][selected & (groups != groups[i])], fdist[i]])
            is_own = np.concatenate([np.ones(len(candidates) - len(fdist[i]), bool), np.zeros(len(fdist[i]), bool)])
            kk = min(k, len(candidates))
            top = np.argpartition(candidates, kk - 1)[:kk]
            counts.append(is_own[top].sum())
            weights = 1.0 / np.maximum(candidates[top], 1e-6)
            hits.append(np.median(candidates[top]) <= reject and weights[is_own[top]].sum() / weights.sum() >= REFERENCE_THRESHOLD)
        n, v = int(selected.sum()), max(float(np.mean(counts)), 0.5)
        return n, float(np.mean(hits)), max(n * (k - v) / v, 0.5)

    rows, recalls, n0s, xs, ys = [], [], [], [], []
    for fraction in LEARNING_CURVE_FRACTIONS:
        draws = []
        for _ in range(LEARNING_CURVE_DRAWS if fraction < 1.0 else 1):
            chosen = units if fraction == 1.0 else rng.permutation(units)[: max(2, int(len(units) * fraction + 0.5))]
            draws.append(at(np.isin(groups, chosen)))
        rows.append(round(np.mean([n for n, _, _ in draws])))
        recalls.append(float(np.mean([r for _, r, _ in draws])))
        n0s.append(float(np.mean([n0 for _, _, n0 in draws])))
        xs += [np.log(n) for n, _, _ in draws]
        ys += [np.log(n0) for _, _, n0 in draws]
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = np.asarray(ys) - (slope * np.asarray(xs) + intercept)
    spread = float(np.sum((np.asarray(xs) - np.mean(xs)) ** 2))
    se = float(np.sqrt(np.sum(residuals**2) / max(len(xs) - 2, 1) / max(spread, 1e-12)))
    verdict = "good" if slope <= 0 else "mixed" if slope > 2 * se else "unclear"
    return _LearningCurve(rows=tuple(rows), recalls=tuple(recalls), n0s=tuple(n0s), verdict=verdict)


def _operating_point(own_shares: np.ndarray, other_shares: np.ndarray | None, spec: PoseSpec) -> tuple[float, float, str]:
    """Enter and exit thresholds of an accepted pose, and the note printed next to them."""
    if spec.enter is not None and spec.exit is not None:
        return spec.enter, spec.exit, "set by you"
    if other_shares is not None and len(other_shares):
        recalls = {float(e): float((own_shares >= e).mean()) for e in OPERATING_POINT_GRID}
        silences = {float(e): float((other_shares < e).mean()) for e in OPERATING_POINT_GRID}
        f1s = {e: 0.0 if recalls[e] + silences[e] == 0 else 2 * recalls[e] * silences[e] / (recalls[e] + silences[e]) for e in recalls}
        enter = max(e for e, f1 in f1s.items() if f1 >= max(f1s.values()) - F1_TIE)
        note = f"{100 * recalls[enter]:.0f}% of your photos fire, {100 * (1.0 - silences[enter]):.0f}% of other fires"
    else:
        enter = max(REFERENCE_THRESHOLD, float(np.percentile(own_shares, PERCENTILE_WITHOUT_OTHER)))
        note = f"{100 * float((own_shares >= enter).mean()):.0f}% of your photos fire"
    exit_ = enter - STATE_EXIT_GAP if spec.type == "state" else max(MIN_ACTION_EXIT, enter - ACTION_EXIT_GAP)
    return enter, exit_, note


def _rows_to_keep(
    shipped_rows: np.ndarray,
    shipped_labels: np.ndarray,
    custom_rows: np.ndarray,
    custom_labels: np.ndarray,
    custom_names: tuple[str, ...],
    k: int,
    scale: np.ndarray,
    reject: float,
) -> np.ndarray:
    """Mask of the shipped rows that stay: a row whose neighbourhood votes the new poses >= CLEANING_SHARE leaves."""
    if not custom_names:
        return np.ones(len(shipped_rows), bool)
    probe = np.vstack([shipped_rows, custom_rows]) / scale
    probe_labels = np.concatenate([shipped_labels, custom_labels])
    idx, dist = _nearest(probe[: len(shipped_rows)], probe, k, [np.array([i]) for i in range(len(shipped_rows))])
    return _shares(idx, dist, probe_labels, reject, custom_names).sum(axis=1) < CLEANING_SHARE


def _confusion_table(
    db: np.ndarray, labels: np.ndarray, k: int, reject: float, poses: tuple[str, ...], groups: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """For each pose (rows), the share of its rows on which each pose (columns) fires at the reference threshold.

    Custom poses hold their groups out, built-in poses hold the single row out; `none` completes each row to 1.
    """
    table: dict[str, dict[str, float]] = {}
    for name in poses:
        row_idx = np.where(labels == name)[0]
        if name in groups:
            skip = [row_idx[groups[name] == groups[name][i]] for i in range(len(row_idx))]
        else:
            skip = [np.array([i]) for i in row_idx]
        shares = _shares(*_nearest(db[row_idx], db, k, skip), labels, reject, poses)
        fires = {column: float((shares[:, j] >= REFERENCE_THRESHOLD).mean()) for j, column in enumerate(poses)}
        fires["none"] = max(0.0, 1.0 - sum(fires.values()))
        table[name] = fires
    return table
