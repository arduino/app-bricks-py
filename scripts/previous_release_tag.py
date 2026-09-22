#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Print the library release tag preceding a version, the starting point of its generated release notes.

git ls-remote --tags origin 'refs/tags/release/*' 'refs/tags/bricks/*' | python3 -m scripts.previous_release_tag 0.13.0
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable


RELEASE_PREFIXES = ("release/", "bricks/")
VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:(?P<stage>rc|a|b)(?P<number>\d+))?$")
STAGE_ORDER: dict[str | None, int] = {"a": 0, "b": 1, "rc": 2, None: 3}


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Return the sort key of an ``X.Y.Z`` version with an optional ``rcN``, ``aN`` or ``bN`` suffix.

    Raises:
        ValueError: when the version has another shape.
    """
    match = VERSION_PATTERN.match(version)
    if not match:
        raise ValueError(f"'{version}' is not X.Y.Z with an optional rcN, aN or bN suffix")
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        STAGE_ORDER[match["stage"]],
        int(match["number"] or 0),
    )


def tag_names(lines: Iterable[str]) -> list[str]:
    """Return the tag names in the output of ``git ls-remote --tags`` or ``git tag``, the peeled ``^{}`` entries included."""
    return [line.split()[-1].removeprefix("refs/tags/") for line in lines if line.strip()]


def previous_release_tag(version: str, tags: Iterable[str]) -> str | None:
    """Return the release tag with the highest version below ``version``, or None when no release precedes it.

    Tags outside the release prefixes, or whose version has another shape, are ignored.

    Raises:
        ValueError: when ``version`` has another shape.
    """
    current = version_key(version)
    candidates: list[tuple[tuple[int, int, int, int, int], str]] = []
    for tag in tags:
        prefix, _, tag_version = tag.partition("/")
        if f"{prefix}/" not in RELEASE_PREFIXES:
            continue
        try:
            key = version_key(tag_version)
        except ValueError:
            continue
        if key < current:
            candidates.append((key, tag))
    return max(candidates)[1] if candidates else None


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="The version being released, X.Y.Z with an optional rcN, aN or bN suffix.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print the tag preceding the version among the tags read from stdin, nothing when there is none."""
    args = create_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        previous = previous_release_tag(args.version, tag_names(sys.stdin))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if previous:
        print(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
