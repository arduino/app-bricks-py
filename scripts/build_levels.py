#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Compute the multi-level build plan for container images.

The dependency graph is declared in ``containers/*/ci.json`` via the
``downstream`` attribute (parent -> children edges). This module walks that
graph, resolves the full set of containers to (re)build for a given selection
and assigns each of them a topological level (build wave), so that a base image
is always built before the images that derive ``FROM`` it.

The result is emitted as ``level_<n>=<json-array>`` lines (plus
``all_containers=<json-array>``) that a GitHub Actions job can consume via
``$GITHUB_OUTPUT``. Each level is an independent matrix; levels run in sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Maximum number of build waves supported by the caller workflows
# (jobs build-l0 .. build-l{MAX_LEVELS - 1}). Bump this together with the
# number of chained jobs in docker-build.yml / docker-publish.yml.
MAX_LEVELS = 3


class BuildLevelsError(RuntimeError):
    """Raised when the build graph or the selection is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON file as a dict."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildLevelsError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BuildLevelsError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BuildLevelsError(f"Expected JSON object in {path}.")
    return payload


class Graph:
    """The container dependency graph built from all ``ci.json`` files."""

    def __init__(self, containers_dir: Path):
        """Load every ``containers/*/ci.json`` and build the edge maps."""
        self.children: dict[str, list[str]] = {}
        self.parents: dict[str, list[str]] = {}
        self.tag_prefix: dict[str, str] = {}
        self.base_image: dict[str, bool] = {}

        ci_files = sorted(containers_dir.glob("*/ci.json"))
        if not ci_files:
            raise BuildLevelsError(f"No containers found (looked for ci.json under {containers_dir}).")

        for ci_json in ci_files:
            name = ci_json.parent.name
            config = load_json(ci_json)
            self.children.setdefault(name, [])
            self.parents.setdefault(name, [])
            self.tag_prefix[name] = str(config.get("tag_prefix", ""))
            self.base_image[name] = bool(config.get("base_image", False))

        # Second pass: wire edges now that every node is known.
        for ci_json in ci_files:
            name = ci_json.parent.name
            downstream = load_json(ci_json).get("downstream") or []
            if not isinstance(downstream, list):
                raise BuildLevelsError(f"Expected 'downstream' to be a JSON array in {ci_json}.")
            for child in downstream:
                if child not in self.children:
                    raise BuildLevelsError(f"'{name}' lists unknown downstream container '{child}'.")
                self.children[name].append(child)
                self.parents[child].append(name)

    @property
    def containers(self) -> set[str]:
        """All known container names."""
        return set(self.children)


def _closure(seeds: set[str], edges: dict[str, list[str]]) -> set[str]:
    """Return the transitive closure of ``seeds`` following ``edges``."""
    result: set[str] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(edges.get(node, []))
    return result


def resolve_build_set(graph: Graph, seeds: set[str]) -> set[str]:
    """Expand ``seeds`` into the full set of containers to (re)build.

    The set is the forward closure of the seeds (all descendants, so downstream
    images are rebuilt) plus the ancestors of that whole forward-closed set (so
    every image being rebuilt has all of its base images rebuilt first). Taking
    ancestors over the forward-closed set — rather than over the seeds alone —
    keeps the result correct for diamond dependencies as well.
    """
    forward = _closure(seeds, graph.children)
    return forward | _closure(forward, graph.parents)


def resolve_dev_build_set(graph: Graph, select: str) -> set[str]:
    """Resolve the containers to build for the DEV workflow.

    ``select`` is either ``all`` or a comma-separated list of container names.
    """
    if select.strip() == "all":
        return set(graph.containers)

    seeds = {token.strip() for token in select.split(",") if token.strip()}
    unknown = sorted(seeds - graph.containers)
    if unknown:
        raise BuildLevelsError(f"Unknown container(s) selected: {', '.join(unknown)}")

    return resolve_build_set(graph, seeds)


def resolve_release_build_set(graph: Graph, tag_prefix: str) -> set[str]:
    """Resolve the containers to build for the RELEASE workflow.

    Seeds are the containers whose ``tag_prefix`` matches the pushed tag prefix,
    excluding those flagged ``base_image`` (common bases are never a direct
    release target). The build set then adds their descendants *and* ancestors,
    so a shared base image is rebuilt as a dependency even though it is not
    itself a release target.
    """
    seeds = {name for name, prefix in graph.tag_prefix.items() if prefix == tag_prefix and not graph.base_image[name]}
    return resolve_build_set(graph, seeds)


def assign_levels(graph: Graph, build_set: set[str]) -> dict[str, int]:
    """Assign a topological level to each container in ``build_set``.

    ``level(c) = 0`` when ``c`` has no parent inside ``build_set``; otherwise it
    is ``max(level(p)) + 1`` over the in-set parents (longest path from a root).
    Raises on dependency cycles.
    """
    levels: dict[str, int] = {}
    visiting: set[str] = set()

    def resolve(node: str) -> int:
        if node in levels:
            return levels[node]
        if node in visiting:
            raise BuildLevelsError(f"Dependency cycle detected involving '{node}'.")
        visiting.add(node)
        in_set_parents = [p for p in graph.parents.get(node, []) if p in build_set]
        level = 0 if not in_set_parents else max(resolve(p) for p in in_set_parents) + 1
        visiting.discard(node)
        levels[node] = level
        return level

    for node in build_set:
        resolve(node)
    return levels


def build_plan(graph: Graph, build_set: set[str]) -> list[list[str]]:
    """Group the build set into ordered, alphabetically-sorted waves."""
    levels = assign_levels(graph, build_set)
    max_level = max(levels.values(), default=-1)
    if max_level >= MAX_LEVELS:
        offenders = sorted(name for name, lvl in levels.items() if lvl >= MAX_LEVELS)
        raise BuildLevelsError(
            f"Dependency chain deeper than the supported {MAX_LEVELS} levels "
            f"(container(s) at level >= {MAX_LEVELS}: {', '.join(offenders)}). "
            f"Increase MAX_LEVELS and add matching build-l* jobs in the workflows."
        )

    waves: list[list[str]] = [[] for _ in range(MAX_LEVELS)]
    for name, level in levels.items():
        waves[level].append(name)
    for wave in waves:
        wave.sort()
    return waves


def emit_outputs(waves: list[list[str]], build_set: set[str]) -> None:
    """Write ``level_<n>`` and ``all_containers`` to $GITHUB_OUTPUT / stdout."""
    lines = [f"level_{index}={json.dumps(wave)}" for index, wave in enumerate(waves)]
    lines.append(f"all_containers={json.dumps(sorted(build_set))}")

    output = "\n".join(lines) + "\n"
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(output)

    # Always echo to stdout as well, so the plan is visible in the job log.
    sys.stdout.write(output)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["dev", "release"], help="Which workflow is requesting the plan.")
    parser.add_argument("--select", default="all", help="dev mode: 'all' or comma-separated container names.")
    parser.add_argument("--tag-prefix", default="", help="release mode: tag prefix that selects the seed containers.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = create_parser().parse_args(sys.argv[1:] if argv is None else argv)

    try:
        graph = Graph(REPO_ROOT / "containers")
        if args.mode == "dev":
            build_set = resolve_dev_build_set(graph, args.select)
        else:
            if not args.tag_prefix:
                raise BuildLevelsError("--tag-prefix is required in release mode.")
            build_set = resolve_release_build_set(graph, args.tag_prefix)

        emit_outputs(build_plan(graph, build_set), build_set)
    except BuildLevelsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
