# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

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
        knn = PoseKNN(k=5, reject_factor=1.5)
        knn.fit(_cluster(0.0), ["a"] * 40)
        assert 0.0 < knn.reject_distance < 1.0  # tight cluster -> tight threshold

    def test_calibration_mask_ignores_augmented_density(self):
        real = RNG.normal(0.0, 0.5, size=(30, 30)).astype(np.float32)
        augmented = real + RNG.normal(0.0, 0.001, size=real.shape).astype(np.float32)
        both = np.vstack([real, augmented])
        labels = ["a"] * len(both)
        mask = np.array([True] * len(real) + [False] * len(augmented))

        naive = PoseKNN(k=5)
        naive.fit(both, labels)
        masked = PoseKNN(k=5)
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

        knn_l2 = PoseKNN(k=1, reject_factor=100.0)
        knn_l2.fit(emb, labels)
        assert knn_l2.classify(query)["standing"] == pytest.approx(1.0)

        knn_l1 = PoseKNN(k=1, reject_factor=100.0, metric="manhattan")
        knn_l1.fit(emb, labels)
        assert knn_l1.classify(query)["sitting"] == pytest.approx(1.0)

    def test_knn_seuclidean_gives_the_quiet_feature_a_voice(self):
        emb = np.asarray([[0.6, -0.5], [3.0, 0.5], [-3.0, 0.5], [-0.6, -0.5]], np.float32)
        labels = ["standing", "sitting", "sitting", "standing"]
        query = np.asarray([0.0, 0.5], np.float32)

        knn_l2 = PoseKNN(k=1, reject_factor=100.0)
        knn_l2.fit(emb, labels)
        assert knn_l2.classify(query)["standing"] == pytest.approx(1.0)

        knn_std = PoseKNN(k=1, reject_factor=100.0, metric="seuclidean")
        knn_std.fit(emb, labels)
        assert knn_std.classify(query)["sitting"] == pytest.approx(1.0)

    def test_knn_label_weights_scale_votes(self):
        emb = np.asarray([[0.0], [0.1], [0.2], [0.3]], np.float32)
        knn = PoseKNN(k=4, reject_factor=100.0)
        knn.fit(emb, ["sitting", "sitting", "other", "other"])
        plain = knn.classify(np.asarray([0.15], np.float32))
        assert plain["sitting"] == pytest.approx(0.5)
        weighted = knn.classify(np.asarray([0.15], np.float32), label_weights={"other": 0.5})
        assert weighted["sitting"] == pytest.approx(2 / 3)
        assert weighted["other"] == pytest.approx(1 / 3)


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
        ema = EmaHysteresis(classes=("arms_up",))
        assert _run(ema, {"arms_up": 1.0}, steps=20) == [("enter", "arms_up")]
        assert _run(ema, {"arms_up": 0.0}, steps=20) == [("exit", "arms_up")]

    def test_hysteresis_gap_prevents_flicker(self):
        ema = EmaHysteresis(classes=("arms_up",))
        _run(ema, {"arms_up": 1.0}, steps=20)
        assert _run(ema, {"arms_up": 0.55}, steps=20) == []
        assert ema.active["arms_up"]

    def test_invalid_frames_freeze_while_person_present(self):
        ema = EmaHysteresis(classes=("arms_up",))
        _run(ema, {"arms_up": 1.0}, steps=20)
        assert _run(ema, None, steps=100, person_present=True) == []

    def test_invalid_frames_decay_after_grace_when_person_absent(self):
        ema = EmaHysteresis(classes=("arms_up",))
        _run(ema, {"arms_up": 1.0}, steps=20)
        assert _run(ema, None, steps=30, person_present=False) == [("exit", "arms_up")]


# ---------------------------------------------------------------------------
# The shipped database asset
# ---------------------------------------------------------------------------


class TestShippedDatabase:
    def test_asset_loads_with_its_own_dials(self):
        knn, label_weights, pose_names = load_pose_classifier(ASSET)
        assert pose_names == ("left_arm_raised", "right_arm_raised", "sitting", "standing")
        assert set(knn.classes) == {*pose_names, "other"}
        assert 0.0 < knn.reject_distance < float("inf")
        assert label_weights is None or set(label_weights) == {"other"}

    def test_a_database_row_classifies_to_a_full_distribution(self):
        knn, label_weights, _ = load_pose_classifier(ASSET)
        data = np.load(ASSET)
        probs = knn.classify(data["embeddings"][0], label_weights=label_weights)
        assert set(probs) == set(knn.classes)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_keypoint_order_matches_the_brick_contract(self):
        assert KEYPOINT_NAMES[0] == "nose"
        assert len(KEYPOINT_NAMES) == 17
        assert KEYPOINT_NAMES[IDX["right_ankle"]] == "right_ankle"
