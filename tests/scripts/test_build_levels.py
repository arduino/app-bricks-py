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


def make_containers_dir(tmp_path: Path, spec: dict[str, dict]) -> Path:
    """Create a temporary ``containers/<group>/<name>/`` tree from a ``{name: attrs}`` spec.

    ``attrs`` may set ``downstream``, ``base_image`` and ``group`` (the sub-folder
    the container is filed under — and therefore the tag that releases it,
    ``bricks`` by default).
    """
    containers_dir = tmp_path / "containers"
    for name, attrs in spec.items():
        ci_dir = containers_dir / attrs.get("group", "bricks") / name
        ci_dir.mkdir(parents=True)
        payload = {"downstream": attrs.get("downstream", [])}
        if attrs.get("base_image"):
            payload["base_image"] = True
        (ci_dir / "ci.json").write_text(json.dumps(payload), encoding="utf-8")
    return containers_dir


def make_graph(tmp_path: Path, spec: dict[str, dict]) -> Graph:
    """Build a ``Graph`` over a temporary ``containers/`` tree."""
    return Graph(make_containers_dir(tmp_path, spec))


# The real chain under test, spread over the container groups as in the repo:
#   qairt -> aihub -> gesture
#   qairt -> llamacpp-npu
CHAIN = {
    "qairt-common-base": {"group": "base", "base_image": True, "downstream": ["aihub-models-runner", "llamacpp-npu-runner"]},
    "aihub-models-runner": {"group": "ai", "downstream": ["gesture-recognition-runner"]},
    "gesture-recognition-runner": {"group": "ai", "downstream": []},
    "llamacpp-npu-runner": {"group": "ai", "downstream": []},
    "standalone": {"group": "bricks", "downstream": []},
}


def test_containers_are_discovered_across_groups(tmp_path):
    """A container is identified by its leaf name, whatever group it is filed under."""
    containers_dir = make_containers_dir(tmp_path, CHAIN)
    graph = Graph(containers_dir)
    assert graph.containers == set(CHAIN)
    assert graph.directory["qairt-common-base"] == containers_dir / "base" / "qairt-common-base"
    assert graph.directory["gesture-recognition-runner"] == containers_dir / "ai" / "gesture-recognition-runner"


def test_duplicate_container_name_across_groups_raises(tmp_path):
    """The same leaf name in two groups is ambiguous: it maps to one image name."""
    containers_dir = tmp_path / "containers"
    for group in ("ai", "bricks"):
        ci_dir = containers_dir / group / "twin"
        ci_dir.mkdir(parents=True)
        (ci_dir / "ci.json").write_text(json.dumps({"downstream": []}), encoding="utf-8")
    with pytest.raises(BuildLevelsError, match="Duplicate container name"):
        Graph(containers_dir)


def test_empty_containers_dir_raises(tmp_path):
    """A flat containers/ tree (pre-reorg layout) yields no containers."""
    (tmp_path / "containers" / "python-base").mkdir(parents=True)
    (tmp_path / "containers" / "python-base" / "ci.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BuildLevelsError, match="No containers found"):
        Graph(tmp_path / "containers")


def test_three_level_chain_from_root(tmp_path):
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "qairt-common-base")
    waves = build_plan(graph, build_set)
    assert len(waves) == MAX_LEVELS
    assert waves[0] == ["qairt-common-base"]
    assert waves[1] == ["aihub-models-runner", "llamacpp-npu-runner"]
    assert waves[2] == ["gesture-recognition-runner"]


def test_reverse_closure_from_leaf(tmp_path):
    """Selecting a leaf must pull in its ancestors so bases build first."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_dev_build_set(graph, "gesture-recognition-runner")
    assert build_set == {"qairt-common-base", "aihub-models-runner", "gesture-recognition-runner"}
    waves = build_plan(graph, build_set)
    assert waves[0] == ["qairt-common-base"]
    assert waves[1] == ["aihub-models-runner"]
    assert waves[2] == ["gesture-recognition-runner"]
    # The sibling llamacpp-npu-runner is NOT an ancestor of gesture and must be excluded.
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


def test_release_seeds_the_whole_tagged_group(tmp_path):
    """An ``ai/X.Y.Z`` tag releases every container under ``containers/ai/``."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_release_build_set(graph, "ai")
    # The three ai containers, plus qairt-common-base pulled in as their base.
    assert build_set == {
        "qairt-common-base",
        "aihub-models-runner",
        "gesture-recognition-runner",
        "llamacpp-npu-runner",
    }
    # standalone lives in the bricks group -> excluded.
    assert "standalone" not in build_set

    waves = build_plan(graph, build_set)
    assert waves[0] == ["qairt-common-base"]
    assert waves[1] == ["aihub-models-runner", "llamacpp-npu-runner"]
    assert waves[2] == ["gesture-recognition-runner"]


def test_release_of_one_group_pulls_bases_from_another(tmp_path):
    """A dependency is built even when it lives in a group that was not tagged."""
    graph = make_graph(tmp_path, CHAIN)
    build_set = resolve_release_build_set(graph, "bricks")
    # Only 'standalone' is in bricks and it has no dependencies.
    assert build_set == {"standalone"}

    spec = dict(CHAIN)
    spec["standalone"] = {"group": "bricks", "downstream": []}
    spec["shared-base"] = {"group": "base", "base_image": True, "downstream": ["standalone"]}
    graph = make_graph(tmp_path / "second", spec)
    build_set = resolve_release_build_set(graph, "bricks")
    assert build_set == {"shared-base", "standalone"}
    waves = build_plan(graph, build_set)
    assert waves[0] == ["shared-base"]
    assert waves[1] == ["standalone"]


def test_release_of_base_group_alone_builds_nothing(tmp_path):
    """Base images are dependency-only: tagging the base group is a no-op."""
    graph = make_graph(tmp_path, CHAIN)
    assert resolve_release_build_set(graph, "base") == set()


def test_release_excludes_base_image_from_seeds_but_builds_it_as_ancestor(tmp_path):
    """A base_image is never a direct seed, but is rebuilt when a released child needs it."""
    spec = {
        "common-base": {"group": "base", "base_image": True, "downstream": ["app-a", "app-b"]},
        "app-a": {"group": "bricks", "downstream": []},
        "app-b": {"group": "bricks", "downstream": []},
        "orphan-base": {"group": "base", "base_image": True, "downstream": []},
    }
    graph = make_graph(tmp_path, spec)
    build_set = resolve_release_build_set(graph, "bricks")
    # common-base is pulled in as the ancestor of the two released apps...
    assert build_set == {"common-base", "app-a", "app-b"}
    # ...but orphan-base (base_image, no released dependents) is never built.
    assert "orphan-base" not in build_set

    waves = build_plan(graph, build_set)
    assert waves[0] == ["common-base"]
    assert waves[1] == ["app-a", "app-b"]


def test_release_diamond_pulls_all_parents(tmp_path):
    """Seeding one branch of a diamond must still rebuild the other parent of a shared child."""
    spec = {
        "base": {"group": "base", "base_image": True, "downstream": ["a", "b"]},
        "a": {"group": "bricks", "downstream": ["c"]},
        "b": {"group": "ai", "downstream": ["c"]},
        "c": {"group": "ai", "downstream": []},
    }
    graph = make_graph(tmp_path, spec)
    build_set = resolve_release_build_set(graph, "bricks")
    # Seeding only 'a' pulls c (descendant), then b (c's other parent) and base.
    assert build_set == {"base", "a", "b", "c"}
    waves = build_plan(graph, build_set)
    assert waves[0] == ["base"]
    assert waves[1] == ["a", "b"]
    assert waves[2] == ["c"]


def test_unknown_release_group_raises(tmp_path):
    """A typo'd tag prefix must fail loudly instead of silently building nothing."""
    graph = make_graph(tmp_path, CHAIN)
    with pytest.raises(BuildLevelsError, match="Unknown container group 'release'"):
        resolve_release_build_set(graph, "release")


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
