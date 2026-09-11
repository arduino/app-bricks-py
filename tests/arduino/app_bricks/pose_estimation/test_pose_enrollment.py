# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from arduino.app_bricks.pose_estimation.classifier import load_pose_classifier
from arduino.app_bricks.pose_estimation.enrollment import Bucket, enroll, group_photos, look_alikes
from arduino.app_bricks.pose_estimation.enrollment.measure import _LearningCurve, _Measure, _operating_point
from arduino.app_bricks.pose_estimation.enrollment.report import render_report
from arduino.app_bricks.pose_estimation.vocabulary import PoseSpec

ASSET = Path(__file__).resolve().parents[4] / "src" / "arduino" / "app_bricks" / "pose_estimation" / "assets" / "pose_classifier.npz"
DIM = 30
RNG = np.random.default_rng(7)
NOW = "2026-09-06 12:00:00"


def _cloud(center: np.ndarray, n: int, spread: float = 1.0) -> np.ndarray:
    return (center + RNG.normal(0.0, spread, size=(n, DIM))).astype(np.float32)


def _center(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 3.0, size=DIM).astype(np.float32)


STANDING, SITTING, GUARDS, NEW = _center(1), _center(2), _center(3), _center(4)


@pytest.fixture(scope="module")
def asset(tmp_path_factory) -> Path:
    """A synthetic shipped database: two built-in poses and a guard cloud, seuclidean like the real one."""
    rows = np.vstack([_cloud(STANDING, 60), _cloud(SITTING, 60), _cloud(GUARDS, 100, spread=2.0)])
    labels = ["standing"] * 60 + ["sitting"] * 60 + ["other"] * 100
    path = tmp_path_factory.mktemp("asset") / "db.npz"
    np.savez(
        path,
        embeddings=rows,
        labels=np.asarray(labels),
        real=np.ones(len(labels), bool),
        dials_json=np.asarray(json.dumps({"k": 9, "metric": "seuclidean", "vote_weighting": "distance", "reject_factor": 1.5, "other_weight": 1.0})),
        thresholds_json=np.asarray(json.dumps({"enter": {"standing": 0.8, "sitting": 0.55}, "exit": {"standing": 0.6, "sitting": 0.35}})),
    )
    return path


def _bucket(name: str, rows: np.ndarray, found: int | None = None, discarded=()) -> Bucket:
    return Bucket(
        name=name,
        embeddings=rows,
        photos=tuple(f"{name}/img_{i:03d}.jpg" for i in range(len(rows))),
        found=found or len(rows) + len(discarded),
        discarded=tuple(discarded),
    )


def _specs(*names: str, **options) -> tuple[PoseSpec, ...]:
    return tuple(PoseSpec(name=name, builtin=name in ("standing", "sitting"), **options.get(name, {})) for name in names)


def _enroll(asset: Path, bucket: Bucket, other: Bucket | None = None, actives=("standing", "sitting"), **options):
    specs = _specs(*actives, bucket.name, **options)
    return enroll(asset, specs, {bucket.name: bucket}, other, NOW)


class TestGroups:
    def test_a_burst_of_identical_photos_is_one_group(self, asset):
        scale = load_pose_classifier(asset)[0].scale
        assert len(np.unique(group_photos(look_alikes(np.repeat(NEW[None], 30, axis=0), scale)))) == 1

    def test_distinct_photos_are_distinct_groups(self, asset):
        scale = load_pose_classifier(asset)[0].scale
        rows = np.stack([_center(seed) for seed in range(10, 22)])
        assert len(np.unique(group_photos(look_alikes(rows, scale)))) == len(rows)

    def test_photos_drifting_in_small_steps_do_not_chain_into_one_group(self, asset):
        scale = load_pose_classifier(asset)[0].scale
        step = 0.9 * scale / np.sqrt(DIM)  # each step is 0.9 in the metric space
        rows = np.stack([NEW + i * step for i in range(6)])
        alike = look_alikes(rows, scale)
        assert alike[0, 1] and not alike[0, 2]
        assert group_photos(alike).tolist() == [0, 0, 1, 1, 2, 2]  # a group holds what is alike to its first photo


class TestComposition:
    def test_shipped_rows_inside_the_new_pose_are_set_aside_and_counted(self, asset):
        shipped = np.load(asset)
        planted = shipped["embeddings"].copy()
        planted[-3:] = _cloud(NEW, 3)  # three guards sitting right on the new pose
        path = asset.with_name("planted.npz")
        np.savez(
            path,
            embeddings=planted,
            labels=shipped["labels"],
            real=shipped["real"],
            dials_json=shipped["dials_json"],
            thresholds_json=shipped["thresholds_json"],
        )
        result = _enroll(path, _bucket("new_pose", _cloud(NEW, 60)))
        assert result.set_aside == {"other": 3}
        assert "3 guards" in result.outcomes["new_pose"].report

    def test_the_users_negatives_are_kept_whole_and_the_dials_stay_shipped(self, asset):
        shipped = load_pose_classifier(asset)[0]
        other = _bucket("other", _cloud(GUARDS, 30, spread=0.2))
        result = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60)), other)
        assert (result.knn._labels == "other").sum() == 100 + 30
        assert np.array_equal(result.knn.scale, shipped.scale)
        assert result.knn.reject_distance == shipped.reject_distance
        assert result.knn.k == shipped.k

    def test_left_out_built_in_poses_become_negatives(self, asset):
        result = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60)), actives=("sitting",))
        assert set(result.knn.classes) == {"sitting", "new_pose", "other"}


class TestVerdicts:
    def test_too_few_photos_to_measure(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 12))).outcomes["new_pose"].report
        assert "verdict: NOT ACCEPTED (at least 20 usable photos are needed to measure; you have 12)" in report
        assert report.endswith("next step: add photos")
        assert "database:" not in report and "measure:" not in report

    def test_too_few_photos_to_accept_but_forming_well(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 31))).outcomes["new_pose"].report
        assert "verdict: NOT ACCEPTED (at least 40 usable photos are needed to accept; you have 31; the pose is forming well)" in report
        assert report.endswith("next step: add photos like these")
        assert "measure:" in report and "learning curve" not in report

    def test_too_few_photos_to_accept_and_far_from_forming(self, asset):
        scattered = np.stack([_center(seed) for seed in range(100, 131)])
        report = _enroll(asset, _bucket("new_pose", scattered)).outcomes["new_pose"].report
        assert "the pose is far from forming)" in report
        assert report.endswith("next step: add photos")

    def test_one_take_has_too_few_groups(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 45, spread=0.01))).outcomes["new_pose"].report
        assert "groups: 1\n" in report
        assert "measure: 0% of your photos fire" in report  # every photo is alike to the judged one, nothing is left
        assert "verdict: NOT ACCEPTED (at least 5 groups are needed; you have 1)" in report
        assert report.endswith("next step: add photos taken at different times, distances or angles")

    def test_a_broad_pose_fails_the_recall_bar_with_a_photo_estimate(self, asset):
        broad = np.vstack([_cloud(_center(seed), 6, spread=0.2) for seed in range(200, 210)])
        outcome = _enroll(asset, _bucket("new_pose", broad)).outcomes["new_pose"]
        assert not outcome.accepted
        assert "% of your photos fire, 70% needed)" in outcome.report
        assert "next step: add photos like these, about " in outcome.report or "next step: keep one variant" in outcome.report

    def test_a_compact_pose_is_accepted_with_a_derived_point(self, asset):
        outcome = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60))).outcomes["new_pose"]
        assert outcome.accepted and outcome.enter >= 0.55 and outcome.exit == pytest.approx(outcome.enter - 0.20)
        report = outcome.report
        assert "verdict: ACCEPTED" in report
        assert "operating point: enter " in report and "% of your photos fire), exit " in report and "smoothing 0.31 s" in report
        assert (
            "confusion (rows: photos of; columns: % of them on which each pose fires at 0.55; near-identical photos of the judged one left out)"
            in report
        )
        assert "—" not in report

    def test_an_action_gets_the_action_exit_gap_and_smoothing(self, asset):
        outcome = _enroll(asset, _bucket("swing", _cloud(NEW, 60)), swing={"type": "action", "duration": 0.9}).outcomes["swing"]
        assert outcome.accepted and outcome.exit == pytest.approx(max(0.10, outcome.enter - 0.40))
        assert "type: action, duration 0.9 s" in outcome.report and "smoothing 0.15 s" in outcome.report

    def test_thresholds_set_by_the_app_are_used_and_said(self, asset):
        outcome = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60)), new_pose={"enter": 0.7, "exit": 0.4, "smoothing": 0.2}).outcomes["new_pose"]
        assert (outcome.enter, outcome.exit) == (0.7, 0.4)
        assert "operating point: enter 0.70, exit 0.40 (set by you), smoothing 0.20 s (set by you)" in outcome.report

    def test_discarded_photos_are_listed_with_path_and_reason(self, asset):
        bucket = _bucket("new_pose", _cloud(NEW, 60), discarded=[("new_pose/img_bad.jpg", "no person detected")])
        report = _enroll(asset, bucket).outcomes["new_pose"].report
        assert "photos: 61 found, 60 usable, 1 discarded\n  new_pose/img_bad.jpg  no person detected\n" in report


class TestOperatingPoint:
    def test_without_negatives_the_point_is_the_20th_percentile_floored_at_the_reference(self):
        spec = PoseSpec(name="p", builtin=False)
        high = np.linspace(0.6, 1.0, 50)
        enter, exit_, note = _operating_point(high, None, spec)
        assert enter == pytest.approx(np.percentile(high, 20)) and exit_ == pytest.approx(enter - 0.20)
        assert note == "80% of your photos fire"
        low = np.linspace(0.3, 0.6, 50)
        assert _operating_point(low, None, spec)[0] == 0.55

    def test_with_negatives_ties_go_to_the_highest_threshold(self):
        spec = PoseSpec(name="p", builtin=False)
        own = np.full(40, 0.9)  # every threshold up to 0.9 gives recall 1
        other = np.full(40, 0.1)  # and silence 1: a flat tie
        enter, _, note = _operating_point(own, other, spec)
        assert enter == 0.9
        assert note == "100% of your photos fire, 0% of other fires"

    def test_the_action_exit_never_drops_below_the_floor(self):
        spec = PoseSpec(name="p", builtin=False, type="action")
        assert _operating_point(np.full(40, 0.3), np.full(40, 0.0), spec)[1] == 0.10


class TestConfusionAndWarnings:
    def test_a_pose_planted_on_a_built_in_one_eats_its_rows(self, asset):
        result = _enroll(asset, _bucket("variant", _cloud(STANDING, 60)))
        assert result.set_aside.get("standing", 0) > 30  # the collision shows up in the rows set aside

    def test_a_collision_in_the_table_is_a_warning(self):
        spec = PoseSpec(name="variant", builtin=False)
        table = {"standing": {"standing": 0.55, "variant": 0.40, "none": 0.05}, "variant": {"standing": 0.09, "variant": 0.88, "none": 0.03}}
        measure = _Measure(own_shares=np.full(60, 0.9), own_neighbours=7.3, n0=15.0)
        report = render_report(
            NOW,
            spec,
            _bucket("variant", _cloud(NEW, 60)),
            None,
            3000,
            {},
            np.asarray([]),
            measure,
            9,
            None,
            (0.72, 0.52, "80% of your photos fire"),
            table,
            {"standing": 262, "variant": 60},
            9,
        )
        assert "warning: variant fires on 40% of the standing photos" in report
        assert "warning: standing fires on 9%" not in report

    def test_rows_of_the_confusion_table_add_up(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60))).outcomes["new_pose"].report
        lines = report.splitlines()
        table = lines[next(i for i, line in enumerate(lines) if line.startswith("confusion")) + 2 :]
        table = [line for line in table if not line.startswith("warning")]
        for line in table:
            assert sum(int(cell.rstrip("%")) for cell in line.split()[1:]) == pytest.approx(100, abs=2)

    def test_many_more_photos_than_a_built_in_pose_is_a_warning(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 200))).outcomes["new_pose"].report
        assert "warning: new_pose has 3x more photos than standing" in report

    def test_fewer_photos_than_a_built_in_pose_is_not(self, asset):
        report = _enroll(asset, _bucket("new_pose", _cloud(NEW, 60))).outcomes["new_pose"].report
        assert "more photos than" not in report


def test_a_mixed_bucket_gets_no_photo_estimate():
    spec = PoseSpec(name="p", builtin=False)
    bucket = _bucket("p", _cloud(NEW, 60))
    measure = _Measure(own_shares=np.full(60, 0.4), own_neighbours=3.4, n0=69.0)
    curve = _LearningCurve(rows=(15, 22, 30, 45, 60), recalls=(0.1, 0.2, 0.2, 0.3, 0.3), n0s=(40.0, 50.0, 60.0, 65.0, 69.0), verdict="mixed")
    report = render_report(NOW, spec, bucket, None, 3000, {}, np.asarray([]), measure, 20, curve, None, {}, {}, 9)
    assert report.endswith(
        "next step: keep one variant of the pose, or split the folder into two poses\n"
        "  (the number of photos needed cannot be estimated while the photos mix variants)"
    )
    good = _LearningCurve(rows=curve.rows, recalls=curve.recalls, n0s=curve.n0s, verdict="good")
    report = render_report(NOW, spec, bucket, None, 3000, {}, np.asarray([]), measure, 20, good, None, {}, {}, 9)
    assert report.endswith("next step: add photos like these, about 140 in total for 70% recall, about 410 for 90%")
    close = _Measure(own_shares=np.full(42, 0.5), own_neighbours=5.5, n0=27.0)
    report = render_report(NOW, spec, _bucket("p", _cloud(NEW, 42)), None, 3000, {}, np.asarray([]), close, 20, good, None, {}, {}, 9)
    assert "about 60 in total for 70% recall" in report  # never fewer than the photos already there


def test_a_shipped_pose_re_taught_from_its_own_rows_is_accepted(tmp_path):
    shipped = np.load(ASSET)
    labels = shipped["labels"].astype(str)
    standing = np.where(labels == "standing")[0]
    taken = standing[:80]
    keep = labels != "standing"  # the calibration re-taught a pose with all of its rows out of the database
    path = tmp_path / "without_standing.npz"
    np.savez(
        path,
        embeddings=shipped["embeddings"][keep],
        labels=shipped["labels"][keep],
        real=shipped["real"][keep],
        dials_json=shipped["dials_json"],
        thresholds_json=shipped["thresholds_json"],
    )
    specs = (PoseSpec(name="sitting", builtin=True), PoseSpec(name="fake_standing", builtin=False))
    result = enroll(path, specs, {"fake_standing": _bucket("fake_standing", shipped["embeddings"][taken])}, None, NOW)
    outcome = result.outcomes["fake_standing"]
    assert outcome.accepted
    assert "measure: " in outcome.report and int(outcome.report.split("measure: ")[1].split("%")[0]) >= 80
