# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The enrollment report of one custom pose, as the app's log and the pose folder receive it."""

import numpy as np

from ..pose_classifier import ACTION_SMOOTHING_SECONDS, DEFAULT_ACTION_DURATION, DEFAULT_SMOOTHING_SECONDS
from ..pose_vocabulary import PoseSpec
from .measure import (
    CLEANING_SHARE,
    COLLISION_WARNING,
    GROUP_DEFINITION,
    IMBALANCE_RATIO,
    MIN_GROUPS,
    MIN_PHOTOS_TO_ACCEPT,
    MIN_PHOTOS_TO_MEASURE,
    OTHER,
    OWN_NEIGHBOURS_FOR_RECALL_30,
    OWN_NEIGHBOURS_FOR_RECALL_70,
    OWN_NEIGHBOURS_FOR_RECALL_90,
    PASS_RECALL,
    REFERENCE_THRESHOLD,
    Bucket,
    _LearningCurve,
    _Measure,
)


def _fmt(value: float) -> str:
    text = f"{value:.3f}"
    return text[:-1] if text.endswith("0") else text


def _photos_needed(n0: float, own_neighbours_target: float, k: int, have: int) -> int:
    needed = max(n0 * own_neighbours_target / (k - own_neighbours_target), have + 10)
    return int(np.ceil(needed / 10.0) * 10)


def render_report(
    now: str,
    spec: PoseSpec,
    bucket: Bucket,
    other: Bucket | None,
    kept: int,
    set_aside: dict[str, int],
    shipped_labels: np.ndarray,
    measure: _Measure | None,
    n_groups: int,
    curve: _LearningCurve | None,
    point: tuple[float, float, str] | None,
    table: dict[str, dict[str, float]],
    row_counts: dict[str, int],
    k: int,
) -> str:
    lines = [f"ENROLLMENT REPORT - {now}", f"pose: {spec.name}"]
    if spec.type == "action":
        lines.append(f"type: action, duration {DEFAULT_ACTION_DURATION if spec.duration is None else spec.duration:g} s")
    else:
        lines.append("type: state")
    lines.append(f"photos: {bucket.found} found, {bucket.usable} usable, {len(bucket.discarded)} discarded")
    lines += [f"  {path}  {reason}" for path, reason in bucket.discarded]
    lines += [f"groups: {n_groups}", f"  {GROUP_DEFINITION}"]
    if other is not None:
        lines.append(f"other (your own negatives): {other.found} found, {other.usable} usable, {len(other.discarded)} discarded")
        lines += [f"  {path}  {reason}" for path, reason in other.discarded]
    if measure is not None:
        aside = ", ".join(f"{count} {'guards' if label == OTHER else label}" for label, count in sorted(set_aside.items(), key=lambda item: -item[1]))
        lines.append(f"database: {kept} rows kept, {sum(set_aside.values())} set aside (voted a new pose >= {CLEANING_SHARE:.2f}): {aside or 'none'}")
        lines.append(
            f"measure: {100 * measure.recall:.0f}% of your photos fire at threshold {REFERENCE_THRESHOLD:.2f} "
            f"({100 * PASS_RECALL:.0f}% needed to pass; each group held out in turn)"
        )
        lines.append(f"  own neighbours among the {k} nearest: {measure.own_neighbours:.1f} / {k}")
    if curve is not None:
        lines.append("  learning curve: " + ", ".join(f"{n} photos {100 * r:.0f}%" for n, r in zip(curve.rows, curve.recalls)))
        lines.append(
            "  consistency: "
            + {
                "good": "good (more photos like these keep helping)",
                "mixed": "mixed (the photos spread over different variants: keep one, or split into two poses)",
                "unclear": "unclear (not enough photos to tell)",
            }[curve.verdict]
        )

    n = bucket.usable
    if point is not None:
        enter, exit_, note = point
        smoothing = (
            spec.smoothing if spec.smoothing is not None else (ACTION_SMOOTHING_SECONDS if spec.type == "action" else DEFAULT_SMOOTHING_SECONDS)
        )
        smoothing_text = f"smoothing {_fmt(smoothing)} s" + (" (set by you)" if spec.smoothing is not None else "")
        lines.append("verdict: ACCEPTED")
        if note == "set by you":
            lines.append(f"operating point: enter {_fmt(enter)}, exit {_fmt(exit_)} (set by you), {smoothing_text}")
        else:
            lines.append(f"operating point: enter {_fmt(enter)} ({note}), exit {_fmt(exit_)}, {smoothing_text}")
        lines += _confusion_lines(table)
        for row, fires in table.items():
            for column, share in fires.items():
                if column != row and column != "none" and share >= COLLISION_WARNING:
                    lines.append(f"warning: {column} fires on {100 * share:.0f}% of the {row} photos")
        for name, count in row_counts.items():
            if name != spec.name and count and n > IMBALANCE_RATIO * count:
                lines.append(f"warning: {spec.name} has {n // count}x more photos than {name}")
        return "\n".join(lines)

    if n < MIN_PHOTOS_TO_MEASURE:
        lines.append(f"verdict: NOT ACCEPTED (at least {MIN_PHOTOS_TO_MEASURE} usable photos are needed to measure; you have {n})")
        lines.append("next step: add photos")
    elif n < MIN_PHOTOS_TO_ACCEPT:
        forming = (
            "the pose is forming well"
            if measure.own_neighbours >= OWN_NEIGHBOURS_FOR_RECALL_70
            else "the pose is forming"
            if measure.own_neighbours >= OWN_NEIGHBOURS_FOR_RECALL_30
            else "the pose is far from forming"
        )
        lines.append(f"verdict: NOT ACCEPTED (at least {MIN_PHOTOS_TO_ACCEPT} usable photos are needed to accept; you have {n}; {forming})")
        lines.append("next step: add photos like these" if measure.own_neighbours >= OWN_NEIGHBOURS_FOR_RECALL_30 else "next step: add photos")
    elif n_groups < MIN_GROUPS:
        lines.append(f"verdict: NOT ACCEPTED (at least {MIN_GROUPS} groups are needed; you have {n_groups})")
        lines.append("next step: add photos taken at different times, distances or angles")
    else:
        lines.append(f"verdict: NOT ACCEPTED ({100 * measure.recall:.0f}% of your photos fire, {100 * PASS_RECALL:.0f}% needed)")
        if curve.verdict == "mixed":
            lines.append("next step: keep one variant of the pose, or split the folder into two poses")
            lines.append("  (the number of photos needed cannot be estimated while the photos mix variants)")
        else:
            for_70 = _photos_needed(measure.n0, OWN_NEIGHBOURS_FOR_RECALL_70, k, n)
            for_90 = _photos_needed(measure.n0, OWN_NEIGHBOURS_FOR_RECALL_90, k, n)
            lines.append(f"next step: add photos like these, about {for_70} in total for 70% recall, about {for_90} for 90%")
    return "\n".join(lines)


def _confusion_lines(table: dict[str, dict[str, float]]) -> list[str]:
    columns = [*table.keys(), "none"]
    width = max(len(name) for name in columns)
    lines = [f"confusion (rows: photos of; columns: % of them on which each pose fires at {REFERENCE_THRESHOLD:.2f}; each group held out)"]
    lines.append("  " + " " * width + "".join(f"  {column:>{max(len(column), 4)}}" for column in columns))
    for row, fires in table.items():
        lines.append(f"  {row:<{width}}" + "".join(f"  {100 * fires[column]:>{max(len(column), 4) - 1}.0f}%" for column in columns))
    return lines
