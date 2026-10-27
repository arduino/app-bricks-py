# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs_generator.extractor import extract_docstrings_with_types  # noqa: E402

BASE = '''
class Base:
    """The base."""

    def __init__(self, model: str) -> None:
        """Keep the model.

        Args:
            model (str): The model.
        """

    @property
    def model(self) -> str:
        """The model."""
        return ""

    def start(self) -> None:
        """Start the base."""

    def stop(self) -> None:
        """Stop the base."""

    def _hidden(self) -> None:
        """Never documented."""
'''

PACKAGE_INIT = "from .base import Base as Base\n"

BRICK = '''
from pkg.support import Base


class Brick(Base):
    """The brick."""

    def __init__(self, model: str, extra: int) -> None:
        """Keep more.

        Args:
            model (str): The model.
            extra (int): More.
        """

    def stop(self) -> None:
        """Stop the brick, its own way."""

    def run(self) -> None:
        """Run the brick."""
'''


def test_the_members_a_class_inherits_are_documented_with_it(tmp_path: Path):
    (tmp_path / "pkg" / "support").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "support" / "__init__.py").write_text(PACKAGE_INIT)
    (tmp_path / "pkg" / "support" / "base.py").write_text(BASE)
    (tmp_path / "pkg" / "brick").mkdir()
    (tmp_path / "pkg" / "brick" / "__init__.py").write_text(BRICK)

    (brick,) = extract_docstrings_with_types(str(tmp_path / "pkg" / "brick" / "__init__.py"), "pkg.brick")

    assert brick.signature == "Brick(model: str, extra: int)", "the constructor is the brick's own"
    assert [m.name for m in brick.methods] == ["__init__", "stop", "run", "start"], "own members first, then the inherited ones not redefined"
    assert next(m for m in brick.methods if m.name == "stop").doc.short_description == "Stop the brick, its own way."
    assert [p.name for p in brick.properties] == ["model"]
    assert all(m.module_name == "pkg.brick" for m in brick.methods), "documented as members of the brick"


def test_a_base_class_outside_the_source_tree_is_skipped(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "brick.py").write_text('''
from threading import Thread


class Brick(Thread):
    """A brick on a foreign base."""

    def run(self) -> None:
        """Run the brick."""
''')

    (brick,) = extract_docstrings_with_types(str(tmp_path / "pkg" / "brick.py"), "pkg.brick")

    assert [m.name for m in brick.methods] == ["run"]
