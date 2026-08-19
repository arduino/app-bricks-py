# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import json

import numpy as np
import pytest

from arduino.app_bricks.pose_estimation.pose_classifier import (
    EMBEDDING_SIZE,
    IDX,
    KEYPOINT_NAMES,
    EmaHysteresis,
    PoseKNN,
    embed,
    load_pose_classifier,
    normalize_pose,
)

RNG = np.random.default_rng(7)

ASSET = Path(__file__).resolve().parents[4] / "src" / "arduino" / "app_bricks" / "pose_estimation" / "assets" / "pose_classifier.npz"


def _cluster(center: float, n: int = 40, dim: int = 30) -> np.ndarray:
    return RNG.normal(center, 0.05, size=(n, dim)).astype(np.float32)


def _write_db(path: Path, classes: list[str], thresholds: dict | None = None) -> Path:
    """A minimal classifier database: a tight three-row cluster per class."""
    payload = {
        "embeddings": np.vstack([_cluster(float(i), n=3) for i, _ in enumerate(classes)]),
        "labels": np.asarray([cls for cls in classes for _ in range(3)]),
        "real": np.ones(3 * len(classes), dtype=bool),
        "dials_json": np.asarray(
            json.dumps({"k": 3, "metric": "euclidean", "vote_weighting": "uniform", "reject_factor": 100.0, "other_weight": 1.0})
        ),
    }
    if thresholds is not None:
        payload["thresholds_json"] = np.asarray(json.dumps(thresholds))
    np.savez(path, **payload)
    return path


def _skeleton() -> np.ndarray:
    """A plausible upright person in pixel coordinates."""
    height = {"nose": 20, "eye": 18, "ear": 22, "shoulder": 60, "elbow": 100, "wrist": 140, "hip": 140, "knee": 210, "ankle": 280}
    xy = np.zeros((17, 2), dtype=np.float32)
    for name, i in IDX.items():
        side = -20 if name.startswith("left") else 20 if name.startswith("right") else 0
        xy[i] = (100 + side, height[name.split("_")[-1]])
    return xy


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embedding_is_position_and_scale_invariant(self):
        base = embed(normalize_pose(_skeleton()))
        moved = embed(normalize_pose(_skeleton() * 2.5 + np.asarray([310.0, -40.0], dtype=np.float32)))
        assert base.shape == (EMBEDDING_SIZE,)
        np.testing.assert_allclose(base, moved, atol=1e-4)

    def test_degenerate_torso_returns_none(self):
        flat = np.zeros((17, 2), dtype=np.float32)  # every joint collapsed: torso length 0
        assert normalize_pose(flat) is None

    def test_shoulder_center_vector_is_the_tail_of_the_features(self):
        norm = normalize_pose(_skeleton())
        feats = embed(norm)
        shoulder_center = (norm[IDX["left_shoulder"]] + norm[IDX["right_shoulder"]]) / 2.0
        np.testing.assert_allclose(feats[-2:], shoulder_center, atol=1e-6)


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------


class TestPoseKNN:
    def test_classifies_by_nearest_cluster(self):
        knn = PoseKNN(k=10)
        knn.fit(np.vstack([_cluster(0.0), _cluster(2.0)]), ["a"] * 40 + ["b"] * 40)
        probs = knn.classify(np.full(30, 2.01, dtype=np.float32))
        assert probs["b"] == 1.0
        assert probs["a"] == 0.0

    def test_rejects_far_queries_as_unknown(self):
        knn = PoseKNN(k=10)
        knn.fit(np.vstack([_cluster(0.0), _cluster(2.0)]), ["a"] * 40 + ["b"] * 40)
        probs = knn.classify(np.full(30, 50.0, dtype=np.float32))
        assert all(p == 0.0 for p in probs.values())

    def test_rejection_distance_is_calibrated_from_db(self):
        knn = PoseKNN(k=5, reject_factor=1.5, metric="euclidean")
        knn.fit(_cluster(0.0), ["a"] * 40)
        assert 0.0 < knn.reject_distance < 1.0  # tight cluster -> tight threshold

    def test_calibration_mask_ignores_augmented_density(self):
        real = RNG.normal(0.0, 0.5, size=(30, 30)).astype(np.float32)
        augmented = real + RNG.normal(0.0, 0.001, size=real.shape).astype(np.float32)
        both = np.vstack([real, augmented])
        labels = ["a"] * len(both)
        mask = np.array([True] * len(real) + [False] * len(augmented))

        naive = PoseKNN(k=5, metric="euclidean")
        naive.fit(both, labels)
        masked = PoseKNN(k=5, metric="euclidean")
        masked.fit(both, labels, calibration_mask=mask)
        assert masked.reject_distance > naive.reject_distance * 5

    def test_knn_cosine_metric_ignores_magnitude(self):
        emb = np.asarray([[10.0, 0.0], [0.0, 1.0], [0.0, 2.0]], np.float32)
        knn = PoseKNN(k=1, metric="cosine", reject_factor=100.0)
        knn.fit(emb, ["standing", "sitting", "sitting"])
        probs = knn.classify(np.asarray([0.001, 0.0], np.float32))
        assert probs["standing"] == pytest.approx(1.0)

    def test_knn_manhattan_tolerates_one_big_deviation(self):
        emb = np.asarray([[1.0, 1.0, 1.0, 1.0], [2.5, 0.0, 0.0, 0.0]], np.float32)
        labels = ["standing", "sitting"]
        query = np.zeros(4, np.float32)

        knn_l2 = PoseKNN(k=1, reject_factor=100.0, metric="euclidean")
        knn_l2.fit(emb, labels)
        assert knn_l2.classify(query)["standing"] == pytest.approx(1.0)

        knn_l1 = PoseKNN(k=1, reject_factor=100.0, metric="manhattan")
        knn_l1.fit(emb, labels)
        assert knn_l1.classify(query)["sitting"] == pytest.approx(1.0)

    def test_knn_seuclidean_gives_the_quiet_feature_a_voice(self):
        emb = np.asarray([[0.6, -0.5], [3.0, 0.5], [-3.0, 0.5], [-0.6, -0.5]], np.float32)
        labels = ["standing", "sitting", "sitting", "standing"]
        query = np.asarray([0.0, 0.5], np.float32)

        knn_l2 = PoseKNN(k=1, reject_factor=100.0, metric="euclidean")
        knn_l2.fit(emb, labels)
        assert knn_l2.classify(query)["standing"] == pytest.approx(1.0)

        knn_std = PoseKNN(k=1, reject_factor=100.0, metric="seuclidean")
        knn_std.fit(emb, labels)
        assert knn_std.classify(query)["sitting"] == pytest.approx(1.0)

    def test_knn_label_weights_scale_votes(self):
        emb = np.asarray([[0.0], [0.1], [0.2], [0.3]], np.float32)
        knn = PoseKNN(k=4, reject_factor=100.0, vote_weighting="uniform")
        knn.fit(emb, ["sitting", "sitting", "other", "other"])
        plain = knn.classify(np.asarray([0.15], np.float32))
        assert plain["sitting"] == pytest.approx(0.5)
        weighted = knn.classify(np.asarray([0.15], np.float32), label_weights={"other": 0.5})
        assert weighted["sitting"] == pytest.approx(2 / 3)
        assert weighted["other"] == pytest.approx(1 / 3)

    def test_knn_distance_weighting_lets_closer_neighbors_speak_louder(self):
        emb = np.asarray([[0.0], [0.1], [0.2], [0.3]], np.float32)
        labels = ["sitting", "sitting", "other", "other"]
        query = np.asarray([0.05], np.float32)

        uniform = PoseKNN(k=4, reject_factor=100.0, vote_weighting="uniform")
        uniform.fit(emb, labels)
        assert uniform.classify(query)["sitting"] == pytest.approx(0.5)

        weighted = PoseKNN(k=4, reject_factor=100.0, vote_weighting="distance")
        weighted.fit(emb, labels)
        assert weighted.classify(query)["sitting"] == pytest.approx(15 / 19)

    def test_rejection_threshold_lives_in_the_chosen_metric_space(self):
        rng = np.random.default_rng(42)
        emb = np.column_stack([rng.normal(0.0, 1.0, 40), rng.normal(0.0, 0.01, 40)]).astype(np.float32)
        labels = ["a"] * 40
        query = np.asarray([0.0, 0.1], np.float32)  # 10 sigmas away on the quiet feature

        loose = PoseKNN(k=5, metric="euclidean")
        loose.fit(emb, labels)
        assert loose.classify(query)["a"] == pytest.approx(1.0)

        strict = PoseKNN(k=5, metric="seuclidean")
        strict.fit(emb, labels)
        assert all(p == 0.0 for p in strict.classify(query).values())


# ---------------------------------------------------------------------------
# Temporal layer
# ---------------------------------------------------------------------------


def _run(ema: EmaHysteresis, probs, steps: int, dt: float = 0.1, person_present: bool = True):
    events = []
    for _ in range(steps):
        events += ema.update(probs, dt, person_present=person_present)
    return events


class TestEmaHysteresis:
    def test_enter_fires_once_and_exit_on_decay(self):
        ema = EmaHysteresis(classes=("sitting",))
        assert _run(ema, {"sitting": 1.0}, steps=20) == [("enter", "sitting")]
        assert _run(ema, {"sitting": 0.0}, steps=20) == [("exit", "sitting")]

    def test_hysteresis_gap_prevents_flicker(self):
        ema = EmaHysteresis(classes=("sitting",))
        _run(ema, {"sitting": 1.0}, steps=20)
        assert _run(ema, {"sitting": 0.55}, steps=20) == []
        assert ema.active["sitting"]

    def test_invalid_frames_freeze_while_person_present(self):
        ema = EmaHysteresis(classes=("sitting",))
        _run(ema, {"sitting": 1.0}, steps=20)
        assert _run(ema, None, steps=20, person_present=True) == []

    def test_a_frozen_pose_expires_when_the_person_stays_unreadable(self):
        ema = EmaHysteresis(classes=("sitting",))
        _run(ema, {"sitting": 1.0}, steps=20)
        assert _run(ema, None, steps=100, person_present=True) == [("exit", "sitting")]

    def test_per_class_thresholds(self):
        ema = EmaHysteresis(classes=("easy", "strict"), enter_threshold={"easy": 0.55, "strict": 0.70}, exit_threshold={"easy": 0.35, "strict": 0.50})
        assert _run(ema, {"easy": 0.6, "strict": 0.6}, steps=20) == [("enter", "easy")]
        assert _run(ema, {"easy": 0.4, "strict": 0.4}, steps=20) == []  # 0.4 sits in easy's hold zone
        assert _run(ema, {"easy": 0.2, "strict": 0.2}, steps=20) == [("exit", "easy")]

    def test_invalid_frames_decay_after_grace_when_person_absent(self):
        ema = EmaHysteresis(classes=("sitting",))
        _run(ema, {"sitting": 1.0}, steps=20)
        assert _run(ema, None, steps=30, person_present=False) == [("exit", "sitting")]


# ---------------------------------------------------------------------------
# The shipped database asset
# ---------------------------------------------------------------------------


class TestShippedDatabase:
    def test_asset_loads_with_its_own_dials(self):
        knn, label_weights, pose_names, thresholds = load_pose_classifier(ASSET)
        assert pose_names == ("left_arm_raised", "right_arm_raised", "sitting", "standing")
        assert set(knn.classes) == {*pose_names, "other"}
        assert 0.0 < knn.reject_distance < float("inf")
        assert label_weights is None or set(label_weights) == {"other"}
        assert all(pose in thresholds["enter"] and pose in thresholds["exit"] for pose in pose_names)

    def test_constructor_defaults_mirror_the_shipped_dials(self):
        dials = json.loads(str(np.load(ASSET)["dials_json"]))
        knn = PoseKNN()
        assert knn.k == dials["k"]
        assert knn.metric == dials["metric"]
        assert knn.vote_weighting == dials["vote_weighting"]
        assert knn.reject_factor == dials["reject_factor"]

    def test_a_database_row_classifies_to_a_full_distribution(self):
        knn, label_weights, _, _ = load_pose_classifier(ASSET)
        data = np.load(ASSET)
        probs = knn.classify(data["embeddings"][0], label_weights=label_weights)
        assert set(probs) == set(knn.classes)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_keypoint_order_matches_the_brick_contract(self):
        assert KEYPOINT_NAMES[0] == "nose"
        assert len(KEYPOINT_NAMES) == 17
        assert KEYPOINT_NAMES[IDX["right_ankle"]] == "right_ankle"


# ---------------------------------------------------------------------------
# The operating point travels in the database file
# ---------------------------------------------------------------------------


class TestThresholdsFromTheFile:
    def test_thresholds_come_from_the_file(self, tmp_path):
        stamped = {"enter": {"standing": 0.90}, "exit": {"standing": 0.70}}
        *_, thresholds = load_pose_classifier(_write_db(tmp_path / "stamped.npz", ["standing", "other"], stamped))
        assert thresholds == stamped

    def test_a_database_without_thresholds_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="thresholds_json"):
            load_pose_classifier(_write_db(tmp_path / "bare.npz", ["sitting", "other"]))

    def test_a_pose_missing_from_the_thresholds_fails_at_load_time(self, tmp_path):
        partial = {"enter": {"standing": 0.90}, "exit": {"standing": 0.70}}
        with pytest.raises(ValueError, match="jumping"):
            load_pose_classifier(_write_db(tmp_path / "partial-stamp.npz", ["jumping", "standing", "other"], partial))

    def test_malformed_thresholds_fail_at_load_time(self, tmp_path):
        with pytest.raises(ValueError, match="mappings"):
            load_pose_classifier(_write_db(tmp_path / "flat.npz", ["standing", "other"], {"enter": 0.9}))
