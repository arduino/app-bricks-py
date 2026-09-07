# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_bricks.pose_estimation import BUILTIN_POSE_NAMES
from arduino.app_bricks.pose_estimation.pose_vocabulary import PoseSpec, parse_poses


def test_none_selects_every_built_in_pose():
    specs = parse_poses(None, BUILTIN_POSE_NAMES)
    assert tuple(spec.name for spec in specs) == BUILTIN_POSE_NAMES
    assert all(spec.builtin and spec.type == "state" and spec.enter is None for spec in specs)


def test_custom_names_are_the_pose_folders():
    specs = parse_poses(
        ["sitting", {"name": "forehand", "type": "action", "duration": 0.7}], BUILTIN_POSE_NAMES, custom_names=("forehand", "backhand")
    )
    assert specs[1] == PoseSpec(name="forehand", builtin=False, type="action", duration=0.7)
    with pytest.raises(ValueError, match="unknown pose 'serve' .*pose folders: forehand, backhand"):
        parse_poses(["serve"], BUILTIN_POSE_NAMES, custom_names=("forehand", "backhand"))


def test_names_and_dicts_mix_in_one_list():
    specs = parse_poses(["standing", {"name": "sitting", "smoothing": 0.2}], BUILTIN_POSE_NAMES)
    assert [spec.name for spec in specs] == ["standing", "sitting"]
    assert specs[1].smoothing == 0.2


@pytest.mark.parametrize(
    ("poses", "message"),
    [
        ("sitting", "must be a list"),
        ([], "at least one pose"),
        ([42], "name or a dict"),
        ([{"type": "state"}], "'name' must be a non-empty string"),
        ([{"name": "sitting", "tau": 0.3}], "unknown key 'tau'"),
        (["jumping"], "unknown pose 'jumping' \(built-in: left_arm_raised, right_arm_raised, sitting, standing; no pose folders found\)"),
        ([{"name": "sitting", "type": "gesture"}], "unknown type 'gesture'"),
        ([{"name": "sitting", "type": "action"}], "built-in poses are held poses"),
        ([{"name": "sitting", "duration": 0.7}], "duration applies to actions only"),
        ([{"name": "sitting", "thresholds": 0.5}], "thresholds must be a dict"),
        ([{"name": "sitting", "thresholds": {"enter": 0.5}}], "thresholds must be a dict"),
        ([{"name": "sitting", "thresholds": {"enter": 0.5, "exit": None}}], "must be a number"),
        ([{"name": "sitting", "thresholds": {"enter": True, "exit": 0.1}}], "must be a number"),
        ([{"name": "sitting", "thresholds": {"enter": 1.5, "exit": 0.1}}], "0 <= exit < enter <= 1"),
        ([{"name": "sitting", "thresholds": {"enter": 0.4, "exit": 0.4}}], "0 <= exit < enter <= 1"),
        ([{"name": "sitting", "smoothing": 0}], "smoothing must be positive"),
        ([{"name": "sitting", "smoothing": "fast"}], "must be a number"),
        (["sitting", {"name": "sitting"}], "declared more than once: sitting"),
    ],
)
def test_malformed_declarations_are_refused(poses, message):
    with pytest.raises(ValueError, match=message):
        parse_poses(poses, BUILTIN_POSE_NAMES)


def test_a_spec_records_what_was_declared():
    spec = PoseSpec.from_item({"name": "sitting", "thresholds": {"enter": 1, "exit": 0}, "smoothing": 2}, BUILTIN_POSE_NAMES)
    assert spec == PoseSpec(name="sitting", builtin=True, enter=1.0, exit=0.0, smoothing=2.0)
    assert PoseSpec.from_item("standing", BUILTIN_POSE_NAMES) == PoseSpec(name="standing", builtin=True)
