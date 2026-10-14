# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Scripts copied into the Linux images must have LF line endings in the working tree.

The images are built from the working tree, so a checkout that converts to CRLF (Git for
Windows with core.autocrlf=true) bakes ``#!/usr/bin/env python3\\r`` into the image and the
script fails with ``env: 'python3\\r': No such file or directory``. .gitattributes forces LF
on checkout; this test catches a checkout made before it, or a rule that stops covering them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = ("containers", "scripts")


def shebang_scripts() -> list[Path]:
    scripts: list[Path] = []
    for directory in SCRIPT_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*")):
            if not path.is_file() or any(part in {".venv", "__pycache__", "models"} for part in path.parts):
                continue
            with path.open("rb") as file:
                if file.read(2) == b"#!":
                    scripts.append(path)
    return scripts


SCRIPTS = shebang_scripts()


def test_the_scripts_are_found():
    assert REPO_ROOT / "scripts" / "licensed" / "run.py" in SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=[str(path.relative_to(REPO_ROOT)) for path in SCRIPTS])
def test_script_has_lf_line_endings(script: Path):
    assert b"\r\n" not in script.read_bytes(), (
        f"{script.relative_to(REPO_ROOT)} has CRLF line endings: re-checkout with .gitattributes in place (git rm --cached -r . && git reset --hard)"
    )
