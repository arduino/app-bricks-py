# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the multi-level container build planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_levels import (  # noqa: E402
    BuildLevelsError,
    Graph,
    MAX_LEVELS,
    build_plan,
    resolve_dev_build_set,
    resolve_release_build_set,
)


def make_graph(tmp_path: Path, spec: dict[str, dict]) -> Graph:
    """Create a temporary ``containers/`` tree from ``{name: {downstream, tag_prefix}}``."""
    containers_dir = tmp_path / "containers"
    for name, attrs in spec.items():
        ci_dir = containers_dir / name
        ci_dir.mkdir(parents=True)
        payload = {
            "tag_prefix": attrs.get("tag_prefix", "release"),
            "downstream": attrs.get("downstream", []),
        }
        (ci_dir / "ci.json").write_text(json.dumps(payload), encoding="utf-8")
    return Graph(containers_dir)


# The real chain under test: qairt -> aihub -> {gesture, llamacpp-npu}.
CHAIN = {
    "qairt-common-base": {"tag_prefix": "ai", "downstream": ["aihub-models-runner"]},
    "aihub-models-runner": {"tag_prefix": "ai", "downstream": ["gesture-recognition-runner", "llamacpp-npu-runner"]},
    "gesture-recognition-runner": {"tag_prefix": "gesture", "downstream": []},
    "llamacpp-npu-runner": {"tag_prefix": "llamacpp", "downstream": []},
    "standalone": {"tag_prefix": "release", "downstream": []},
}


def test_three_level_chain_from_root(tmp_path):
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "qairt-common-base")
    waves = build_plan(graph, build_set)
    assert len(waves) == MAX_LEVELS
    assert waves[0] == ["qairt-common-base"]
    assert waves[1] == ["aihub-models-runner"]
    assert waves[2] == ["gesture-recognition-runner", "llamacpp-npu-runner"]


def test_reverse_closure_from_leaf(tmp_path):
    """Selecting a leaf must pull in its ancestors so bases build first."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "gesture-recognition-runner")
    assert build_set == {"qairt-common-base", "aihub-models-runner", "gesture-recognition-runner"}
    waves = build_plan(graph, build_set)
    assert waves[0] == ["qairt-common-base"]
    assert waves[1] == ["aihub-models-runner"]
    assert waves[2] == ["gesture-recognition-runner"]
    # The sibling llamacpp-npu-runner is NOT an ancestor and must be excluded.
    assert "llamacpp-npu-runner" not in build_set


def test_dev_all_selects_everything(tmp_path):
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "all")
    assert build_set == set(CHAIN)


def test_node_without_parents_is_level_zero(tmp_path):
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "standalone")
    waves = build_plan(graph, build_set)
    assert waves[0] == ["standalone"]


def test_release_forward_closure_by_tag_prefix(tmp_path):
    """Release seeds match tag_prefix, then pull descendants regardless of prefix."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_release_build_set(graph, "ai")
    # qairt+aihub match 'ai'; their descendants (gesture, llamacpp-npu) come along.
    assert build_set == {
        "qairt-common-base",
        "aihub-models-runner",
        "gesture-recognition-runner",
        "llamacpp-npu-runner",
    }
    # standalone has tag_prefix 'release' and is unrelated -> excluded.
    assert "standalone" not in build_set


def test_release_forward_only_no_ancestors(tmp_path):
    """Release must not pull ancestors of a matched middle node."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_release_build_set(graph, "gesture")
    assert build_set == {"gesture-recognition-runner"}


def test_unknown_selected_container_raises(tmp_path):
    graph = make_graph(tmp_path, CHAIN)
    with pytest.raises(BuildLevelsError, match="Unknown container"):
        resolve_dev_build_set(graph, "does-not-exist")


def test_unknown_downstream_edge_raises(tmp_path):
    with pytest.raises(BuildLevelsError, match="unknown downstream"):
        make_graph(tmp_path, {"a": {"downstream": ["ghost"]}})


def test_cycle_detection(tmp_path):
    graph = make_graph(
        tmp_path,
        {
            "a": {"downstream": ["b"]},
            "b": {"downstream": ["a"]},
        },
    )
    with pytest.raises(BuildLevelsError, match="cycle"):
        build_plan(graph, {"a", "b"})


def test_depth_cap_exceeded_raises(tmp_path):
    # Build a chain longer than MAX_LEVELS.
    spec = {}
    names = [f"c{i}" for i in range(MAX_LEVELS + 1)]
    for i, name in enumerate(names):
        spec[name] = {"downstream": [names[i + 1]] if i + 1 < len(names) else []}
    graph = make_graph(tmp_path, spec)
    with pytest.raises(BuildLevelsError, match="deeper than the supported"):
        build_plan(graph, set(names))
