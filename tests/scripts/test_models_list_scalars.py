# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards on the shipped models list.

``models/models-*.yaml`` is copied verbatim into the wheel by the build backend, so a
YAML quirk here reaches every consumer of the library. The one that bit us: the
Norwegian language code ``no`` is a YAML 1.1 boolean, so an unquoted ``- no`` became
``False`` in the Whisper language list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_LIST_FILES = sorted((REPO_ROOT / "models").glob("models-*.yaml"))


def _scalars_in_lists(node: object, trail: str = "") -> list[tuple[str, object]]:
    """Every scalar that sits directly inside a list, with a readable path."""
    found: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += _scalars_in_lists(value, f"{trail}.{key}" if trail else str(key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, (dict, list)):
                found += _scalars_in_lists(item, f"{trail}[{index}]")
            else:
                found.append((f"{trail}[{index}]", item))
    return found


def test_models_list_files_exist():
    assert MODELS_LIST_FILES, "no models/models-*.yaml found — the build would fail too"


@pytest.mark.parametrize("path", MODELS_LIST_FILES, ids=lambda p: p.name)
def test_list_entries_are_never_coerced_scalars(path: Path):
    """No list item may parse as a bool, int, float or None.

    These lists hold language codes, board names and brick ids, which are all strings.
    Anything else means YAML coerced a bare word such as ``no``, ``yes``, ``on`` or
    ``off``. Quote the value to fix it.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    coerced = [(where, value) for where, value in _scalars_in_lists(data) if not isinstance(value, str)]

    listed = ", ".join(f"{where} = {value!r}" for where, value in coerced)
    assert not coerced, f"quote these values in {path.name}: {listed}"


@pytest.mark.parametrize("path", MODELS_LIST_FILES, ids=lambda p: p.name)
def test_language_lists_keep_norwegian_as_a_string(path: Path):
    """The regression that motivated the guard above, pinned explicitly."""
    for where, value in _scalars_in_lists(yaml.safe_load(path.read_text(encoding="utf-8"))):
        if "supported_languages" in where:
            assert value is not False, f"{where} is the Norwegian code 'no' read as a boolean"
