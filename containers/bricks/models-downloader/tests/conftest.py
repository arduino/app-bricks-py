# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pytest configuration for the models-downloader container tests.

Adds the container ``src`` directory to ``sys.path`` so ``list_models`` and its
``common`` package imports resolve the same way they do at runtime.
"""

import os
import sys

import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(autouse=True)
def _clear_scandir_cache():
    """Reset the module-level scandir cache between tests."""
    import list_models

    list_models._SEARCH_DIR_CACHE.clear()
    yield
    list_models._SEARCH_DIR_CACHE.clear()


@pytest.fixture(autouse=True)
def exit_refreshes(monkeypatch, tmp_path):
    """Keep the materialized listing of the tests off the real /models.

    The refreshes a downloader registers for its exit are collected here instead of
    running when pytest exits, so a test runs them itself; the ones it runs right
    away find no models root, and write nothing, unless the test points them at one.
    """
    import types

    import list_models

    registered = []
    monkeypatch.setattr(list_models, "atexit", types.SimpleNamespace(register=registered.append))
    monkeypatch.setattr(list_models, "_REFRESH_AT_EXIT", False)
    monkeypatch.setattr(list_models, "MODELS_ROOT_DIR", str(tmp_path / "no-models-root"))
    return registered
