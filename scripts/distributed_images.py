#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""List every container image a release distributes, as ``name:version`` pairs.

Containers live in ``containers/<group>/<name>/``, where the group is the
sub-folder (``ai``, ``bricks`` or ``base``) and also the prefix of the tag that
releases it: ``bricks/1.2.3`` builds only the containers under
``containers/bricks/``, plus the base images they need (see
``scripts.build_levels``). The library shipped by that release also uses the
containers of the other groups, released earlier by their own tags: the compose
files under ``src/`` pin the exact image version of each one. The SBOM archive
attached to the release must cover all of these images, so this module lists:

- the containers built by this release, at the version being released;
- the containers of every other group, at the version pinned in the compose
  files. A group is released as a whole, so all its pinned references must
  share the same version, otherwise the run fails.

In both cases the base images the containers derive from are included. A base
shared by two groups appears once per version, e.g. ``python-slim`` at the ai
version and at the bricks version.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.build_levels import BuildLevelsError, Graph, resolve_release_build_set


REPO_ROOT = Path(__file__).resolve().parents[1]

# Compose files updated by the release bot; generated copies under static/ are skipped.
COMPOSE_GLOBS = ("**/brick_compose*.yaml", "**/service_compose*.yaml")
IMAGE_REFERENCE = re.compile(r"app-bricks/(?P<name>[a-z0-9._-]+):(?P<version>[A-Za-z0-9._-]+)")


class DistributedImagesError(RuntimeError):
    """Raised when the distributed image set cannot be resolved."""


def compose_files(src_dir: Path) -> list[Path]:
    """Return the compose files that pin container image versions."""
    files = {path for pattern in COMPOSE_GLOBS for path in src_dir.glob(pattern) if "static" not in path.parts}
    return sorted(files)


def compose_image_versions(src_dir: Path) -> dict[str, set[str]]:
    """Map every container referenced by the compose files to the versions it is pinned to."""
    versions: dict[str, set[str]] = {}
    for path in compose_files(src_dir):
        for match in IMAGE_REFERENCE.finditer(path.read_text(encoding="utf-8")):
            versions.setdefault(match.group("name"), set()).add(match.group("version"))
    return versions


def group_version(graph: Graph, group: str, referenced: dict[str, set[str]]) -> str:
    """Return the single version the compose files pin a group's containers to."""
    versions = {version for name, name_versions in referenced.items() if graph.group.get(name) == group for version in name_versions}
    if not versions:
        raise DistributedImagesError(f"No compose file references a container of group '{group}': cannot resolve its distributed version.")
    if len(versions) > 1:
        raise DistributedImagesError(f"Compose files pin group '{group}' to several versions: {', '.join(sorted(versions))}.")
    return versions.pop()


def distributed_images(graph: Graph, released_group: str, version: str, src_dir: Path, pinned_only: bool = False) -> list[tuple[str, str]]:
    """Return the sorted ``(name, version)`` pairs of every image a release distributes.

    With ``pinned_only`` the released group is left out, keeping only the images
    the release ships but does not build: those pinned by the compose files.
    """
    referenced = compose_image_versions(src_dir)
    images: set[tuple[str, str]] = set()
    for group in sorted(graph.groups):
        if pinned_only and group == released_group:
            continue
        build_set = resolve_release_build_set(graph, group)
        if not build_set:
            continue
        group_ver = version if group == released_group else group_version(graph, group, referenced)
        images.update((name, group_ver) for name in build_set)
    return sorted(images)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="Container group being released (the pushed tag's prefix).")
    parser.add_argument("--version", required=True, help="Version being released (the tag without prefix).")
    parser.add_argument("--pinned-only", action="store_true", help="Leave out the released group, list only the images pinned by compose files.")
    parser.add_argument("--format", choices=["lines", "json"], default="lines", help="One name:version per line, or a JSON array of them.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print the ``name:version`` specs accepted by ``scripts/sbom_delta.py``."""
    args = create_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        graph = Graph(REPO_ROOT / "containers")
        if args.group not in graph.groups:
            raise DistributedImagesError(f"Unknown container group '{args.group}'. Known groups: {', '.join(sorted(graph.groups))}.")
        images = distributed_images(graph, args.group, args.version, REPO_ROOT / "src", pinned_only=args.pinned_only)
        specs = [f"{name}:{version}" for name, version in images]
        print(json.dumps(specs) if args.format == "json" else "\n".join(specs))
    except (BuildLevelsError, DistributedImagesError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
