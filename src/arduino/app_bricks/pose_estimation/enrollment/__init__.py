# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Enrollment of custom poses: from the photos' embeddings to a composed classifier, a verdict and a
report per pose. `photos` reads the folders through the model runner; everything else is numbers."""

from pathlib import Path

import numpy as np

from ..classifier import PoseKNN, load_pose_classifier
from ..vocabulary import PoseSpec
from .measure import (
    MIN_GROUPS,
    MIN_PHOTOS_TO_ACCEPT,
    MIN_PHOTOS_TO_MEASURE,
    OTHER,
    PASS_RECALL,
    Bucket,
    Enrollment,
    Outcome,
    _confusion_table,
    _learning_curve,
    _measure,
    _nearest,
    _operating_point,
    _rows_to_keep,
    _shares,
    group_photos,
    look_alikes,
)
from .report import render_report

__all__ = ["OTHER", "Bucket", "Enrollment", "Outcome", "enroll", "group_photos", "look_alikes"]


def enroll(asset_path: Path, specs: tuple[PoseSpec, ...], buckets: dict[str, Bucket], other: Bucket | None, now: str) -> Enrollment:
    """Compose the database for the declared vocabulary and judge every custom pose.

    Args:
        asset_path: The shipped classifier database.
        specs: The declared vocabulary, built-in and custom poses alike.
        buckets: The embedded photos of each custom pose, by name.
        other: The user's own negatives, or None.
        now: Timestamp printed in the reports.
    """
    shipped, label_weights, builtin_names, _ = load_pose_classifier(asset_path)
    if shipped.metric != "seuclidean" or shipped.vote_weighting != "distance" or label_weights is not None:
        raise ValueError("custom poses need a seuclidean, distance-weighted classifier that weighs every vote alike")
    asset = np.load(asset_path)
    k, reject = shipped.k, shipped.reject_distance
    scale = shipped.scale if shipped.scale is not None else np.ones(asset["embeddings"].shape[1], np.float32)
    shipped_rows, shipped_labels, shipped_real = asset["embeddings"].astype(np.float32), asset["labels"].astype(str), asset["real"].astype(bool)
    active = tuple(spec.name for spec in specs)
    custom = tuple(spec.name for spec in specs if not spec.builtin)
    measured = tuple(name for name in custom if buckets[name].usable >= MIN_PHOTOS_TO_MEASURE)

    custom_rows = np.vstack([buckets[name].embeddings for name in measured]) if measured else np.empty((0, shipped_rows.shape[1]), np.float32)
    custom_labels = np.concatenate([np.full(buckets[name].usable, name) for name in measured]) if measured else np.empty(0, str)
    keep = _rows_to_keep(shipped_rows, shipped_labels, custom_rows, custom_labels, measured, k, scale, reject)
    set_aside = {label: int(((~keep) & (shipped_labels == label)).sum()) for label in np.unique(shipped_labels[~keep])}

    kept_labels = np.where(np.isin(shipped_labels[keep], active), shipped_labels[keep], OTHER)
    other_rows = other.embeddings if other is not None else np.empty((0, shipped_rows.shape[1]), np.float32)
    rows = np.vstack([shipped_rows[keep], custom_rows, other_rows]).astype(np.float32)
    labels = np.concatenate([kept_labels, custom_labels, np.full(len(other_rows), OTHER)])
    real = np.concatenate([shipped_real[keep], np.zeros(len(custom_rows) + len(other_rows), bool)])
    knn = PoseKNN(k=k, reject_factor=shipped.reject_factor, metric=shipped.metric, vote_weighting=shipped.vote_weighting)
    knn.fit(rows, list(labels), calibration_mask=real, scale=scale, reject_distance=reject)
    db = rows / scale

    alike = {name: look_alikes(buckets[name].embeddings, scale) for name in custom}
    groups = {name: group_photos(alike[name]) for name in custom}
    own_idx = {name: np.where(labels == name)[0] for name in measured}
    measures = {name: _measure(db, labels, own_idx[name], alike[name], k, reject, name) for name in measured}
    other_idx = np.where(labels == OTHER)[0][-len(other_rows) :] if len(other_rows) else np.empty(0, np.int64)
    other_shares = None
    if len(other_idx):
        idx, dist = _nearest(db[other_idx], db, k, [np.array([i]) for i in other_idx])
        other_shares = _shares(idx, dist, labels, reject, measured) if measured else None

    pending: dict[str, tuple] = {}
    accepted: dict[str, tuple[float, float, str]] = {}
    for spec in specs:
        if spec.builtin:
            continue
        bucket = buckets[spec.name]
        measure = measures.get(spec.name)
        n, n_groups = bucket.usable, int(len(np.unique(groups[spec.name]))) if bucket.usable else 0
        curve = None
        if measure is not None and n >= MIN_PHOTOS_TO_ACCEPT and n_groups >= MIN_GROUPS:
            curve = _learning_curve(db, labels, own_idx[spec.name], groups[spec.name], alike[spec.name], k, reject, spec.name)
        if measure is not None and curve is not None and measure.recall > PASS_RECALL:
            accepted[spec.name] = _operating_point(
                measure.own_shares, other_shares[:, measured.index(spec.name)] if other_shares is not None else None, spec
            )
        pending[spec.name] = (spec, bucket, measure, n_groups, curve)

    confusion_rows = tuple(name for name in active if name in measured or (name in builtin_names and (labels == name).any()))
    table = _confusion_table(db, labels, k, reject, confusion_rows, {name: alike[name] for name in measured})
    row_counts = {name: int((labels == name).sum()) for name in confusion_rows}

    results: dict[str, Outcome] = {}
    for name, (spec, bucket, measure, n_groups, curve) in pending.items():
        point = accepted.get(name)
        report = render_report(
            now,
            spec,
            bucket,
            other,
            len(shipped_rows) - int((~keep).sum()),
            set_aside,
            shipped_labels,
            measure,
            n_groups,
            curve,
            point,
            table,
            row_counts,
            k,
        )
        results[name] = Outcome(
            name=name, accepted=point is not None, report=report, enter=point[0] if point else None, exit=point[1] if point else None
        )
    return Enrollment(knn=knn, outcomes=results, set_aside=set_aside)
