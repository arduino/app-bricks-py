# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The API docs extractor recognises every spelling of the dataclass decorator.

A dataclass has no ``__init__`` in the source, so its signature is built from its
annotated attributes — but only when the extractor sees the decorator. It used to see
only the bare ``@dataclass``: ``@dataclass(frozen=True)`` rendered as ``Answer()``.
"""

import sys
import textwrap
from pathlib import Path

import pytest

# Make the repo-root ``docs_generator`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs_generator.extractor import extract_docstrings_with_types  # noqa: E402

MODULE = '''
from dataclasses import dataclass
import dataclasses

__all__ = ["Bare", "Frozen", "Qualified", "QualifiedFrozen", "Plain"]


@dataclass
class Bare:
    """Data class with the bare decorator."""

    a: int
    b: str = "x"


@dataclass(frozen=True)
class Frozen:
    """Data class with options."""

    score: float
    legend: list[str]


@dataclasses.dataclass
class Qualified:
    """Data class with the qualified decorator."""

    a: int


@dataclasses.dataclass(frozen=True, slots=True)
class QualifiedFrozen:
    """Data class with the qualified decorator and options."""

    a: int


class Plain:
    """Not a dataclass: its signature comes from __init__."""

    a: int

    def __init__(self, b: str) -> None:
        """Build it."""
        self.b = b
'''


@pytest.fixture
def signatures(tmp_path):
    module = tmp_path / "mod.py"
    module.write_text(textwrap.dedent(MODULE), encoding="utf-8")
    return {info.name: info.signature for info in extract_docstrings_with_types(str(module), "mod") if info.kind == "class"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Bare", "Bare(a: int, b: str)"),
        ("Frozen", "Frozen(score: float, legend: list[str])"),
        ("Qualified", "Qualified(a: int)"),
        ("QualifiedFrozen", "QualifiedFrozen(a: int)"),
    ],
)
def test_every_dataclass_spelling_signs_with_its_attributes(signatures, name, expected):
    assert signatures[name] == expected


def test_a_plain_class_still_signs_with_its_init(signatures):
    assert signatures["Plain"] == "Plain(b: str)"
