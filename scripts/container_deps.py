#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Derive the container graph from the Dockerfiles.

Containers live in ``containers/<group>/<name>/`` and are identified by their
leaf directory name, which is also their image name. The base image of each
one is declared exactly once, in the ``FROM`` of its Dockerfile's final stage:
this module resolves it through multi-stage builds and tells whether it is
another container of this repository (``FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION}``)
or an external image. CI therefore needs no second, drift-prone copy of the
dependency graph; ``docker-bake.hcl`` links the same parents so bake builds
them in order.

    python3 -m scripts.container_deps                 # JSON map of base image and parent per container
    python3 -m scripts.container_deps list            # JSON array of every container
    python3 -m scripts.container_deps closure NAME... # the selection widened with its parents and children
    python3 -m scripts.container_deps tree            # the hierarchy, grouped by external base image
    docker buildx bake --print | python3 -m scripts.container_deps check-bake   # docker-bake.hcl agrees with the Dockerfiles
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_GLOB = "*/*/Dockerfile"

FROM_PATTERN = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", re.IGNORECASE)
PARENT_PATTERN = re.compile(r"^\$\{REGISTRY\}app-bricks/([a-z0-9._-]+):\$\{BASE_IMAGE_VERSION\}$")


class ContainerDepsError(RuntimeError):
    """Raised when the Dockerfiles do not describe a valid graph."""


def resolve_base_image(dockerfile: Path) -> str:
    """Return the image the Dockerfile's final stage builds on.

    Multi-stage builds produce the last stage, so resolution starts there and
    follows ``FROM <alias>`` references through earlier stages until it reaches
    an image that is not a stage of the same Dockerfile.
    """
    stages: list[tuple[str, str | None]] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = FROM_PATTERN.match(line)
        if match:
            stages.append((match.group(1), match.group(2)))
    if not stages:
        raise ContainerDepsError(f"No FROM instruction in {dockerfile}")

    aliases = {alias.lower(): base for base, alias in stages if alias}
    base = stages[-1][0]
    seen: set[str] = set()
    while base.lower() in aliases:
        if base.lower() in seen:
            raise ContainerDepsError(f"Circular stage references in {dockerfile}")
        seen.add(base.lower())
        base = aliases[base.lower()]
    return base


def parent_container(base_image: str) -> str | None:
    """Return the container name when the base image is built by this repository."""
    match = PARENT_PATTERN.match(base_image)
    return match.group(1) if match else None


class Containers:
    """The containers of the repository, with their base image and parent."""

    def __init__(self, containers_dir: Path) -> None:
        """Read every ``containers/<group>/<name>/Dockerfile``."""
        self.directory: dict[str, Path] = {}
        self.base: dict[str, str] = {}
        self.parent: dict[str, str | None] = {}

        dockerfiles = sorted(containers_dir.glob(DOCKERFILE_GLOB))
        if not dockerfiles:
            raise ContainerDepsError(f"No containers found (looked for {DOCKERFILE_GLOB} under {containers_dir}).")

        for dockerfile in dockerfiles:
            name = dockerfile.parent.name
            if name in self.directory:
                raise ContainerDepsError(
                    f"Duplicate container name '{name}': {self.directory[name]} and {dockerfile.parent}. "
                    f"Container names must be unique across groups (the name is also the image name)."
                )
            self.directory[name] = dockerfile.parent
            self.base[name] = resolve_base_image(dockerfile)
            self.parent[name] = parent_container(self.base[name])

        for name, parent in self.parent.items():
            if parent is not None and parent not in self.directory:
                raise ContainerDepsError(f"'{name}' builds FROM unknown container '{parent}'.")

    @property
    def names(self) -> list[str]:
        """Every container name, sorted."""
        return sorted(self.directory)

    def children(self, name: str) -> list[str]:
        """The containers building FROM ``name``."""
        return sorted(child for child, parent in self.parent.items() if parent == name)

    def closure(self, selection: list[str]) -> list[str]:
        """Widen a selection so related images stay consistent.

        The containers deriving from the selection are added, so a parent is
        never rebuilt without its children, then the parents of the whole set,
        so every rebuilt image sits on a freshly built base.
        """
        unknown = sorted(set(selection) - set(self.directory))
        if unknown:
            raise ContainerDepsError(f"Unknown container(s): {', '.join(unknown)}")

        selected = set(selection)
        frontier = set(selection)
        while frontier:
            frontier = {child for name in frontier for child in self.children(name)} - selected
            selected |= frontier
        frontier = set(selected)
        while frontier:
            frontier = {parent for name in frontier if (parent := self.parent[name])} - selected
            selected |= frontier
        return sorted(selected)

    def check_bake(self, definition: dict) -> list[str]:
        """Return what disagrees between a ``docker buildx bake --print`` definition and the Dockerfiles.

        Every container must be a target of the default group, every target must
        be a container, and a target must link exactly the parent its Dockerfile
        builds FROM: bake rewrites that FROM to the freshly built parent only
        through the link, without it the image is pulled from the registry instead.
        """
        targets = definition.get("target") or {}
        problems = [f"'{name}' has a Dockerfile but no bake target in the default group" for name in sorted(set(self.names) - set(targets))]
        problems += [f"bake target '{name}' has no Dockerfile under containers/" for name in sorted(set(targets) - set(self.names))]
        for name in sorted(set(self.names) & set(targets)):
            contexts = targets[name].get("contexts") or {}
            linked = sorted(value.removeprefix("target:") for value in contexts.values() if value.startswith("target:"))
            expected = [self.parent[name]] if self.parent[name] else []
            if linked != expected:
                problems.append(
                    f"bake target '{name}' links {', '.join(linked) or 'no parent'} but its Dockerfile builds FROM "
                    f"{expected[0] if expected else 'an external image'}"
                )
        return problems

    def to_dict(self) -> dict[str, dict[str, str | None]]:
        """Map every container to its base image and parent container."""
        return {name: {"base": self.base[name], "parent": self.parent[name]} for name in self.names}

    def tree(self) -> str:
        """Render the hierarchy, grouped by external base image."""
        lines: list[str] = []

        def render(name: str, prefix: str) -> None:
            children = self.children(name)
            for index, child in enumerate(children):
                last = index == len(children) - 1
                lines.append(f"{prefix}{'└─' if last else '├─'} {child}")
                render(child, prefix + ("   " if last else "│  "))

        roots = [name for name in self.names if self.parent[name] is None]
        for base in sorted({self.base[root] for root in roots}):
            lines.append(base)
            base_roots = [root for root in roots if self.base[root] == base]
            for index, root in enumerate(base_roots):
                last = index == len(base_roots) - 1
                lines.append(f"{'└─' if last else '├─'} {root}")
                render(root, "   " if last else "│  ")
            lines.append("")
        return "\n".join(lines).rstrip()


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="Print every container name as a JSON array.")
    closure_parser = subparsers.add_parser("closure", help="Widen a selection with its parents and children, as a JSON array.")
    closure_parser.add_argument("containers", nargs="+", help="Selected container names.")
    subparsers.add_parser("tree", help="Print the container hierarchy.")
    subparsers.add_parser("check-bake", help="Check a `docker buildx bake --print` definition, read from stdin, against the Dockerfiles.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = create_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        containers = Containers(REPO_ROOT / "containers")
        if args.command == "list":
            print(json.dumps(containers.names))
        elif args.command == "closure":
            print(json.dumps(containers.closure(args.containers)))
        elif args.command == "tree":
            print(containers.tree())
        elif args.command == "check-bake":
            problems = containers.check_bake(json.load(sys.stdin))
            for problem in problems:
                print(f"Error: {problem}", file=sys.stderr)
            if problems:
                return 1
            print("docker-bake.hcl agrees with the Dockerfiles")
        else:
            print(json.dumps(containers.to_dict(), indent=2))
    except ContainerDepsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
