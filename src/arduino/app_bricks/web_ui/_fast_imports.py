# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Import-time shortcut for Socket.IO: load the server without the dependencies of the sync client, part of an app's start time on the board."""

import importlib
import importlib.util
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager


SOCKETIO_CLIENT_DEPENDENCIES = ("requests", "websocket")
"""Optional dependencies that ``engineio.client`` imports at module level, used only by the Socket.IO sync client."""


class _ImportOnFirstUse:
    """Stands in for a module and imports it on the first attribute access."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> object:
        return getattr(importlib.import_module(self._name), attr)

    def __repr__(self) -> str:
        return f"<module {self._name!r}, imported on first use>"


@contextmanager
def defer_socketio_client_dependencies(names: Sequence[str] = SOCKETIO_CLIENT_DEPENDENCIES) -> Generator[None]:
    """Import Socket.IO inside the block without loading the dependencies of its sync client.

    ``import socketio`` always loads its sync client, which imports requests (with urllib3 and charset_normalizer)
    and websocket-client: about 0.27 s on the board, although WebUI only runs the server. Inside the block those
    modules are hidden from the import system, so an optional ``try: import requests`` falls back to None. On exit,
    every module imported inside the block that was left with None gets a stand-in that imports the real module on
    first use: the sync client still works and pays that cost only if the app uses it. Dependencies that are not
    installed are left alone, and engineio reports them missing as usual.

    The modules are hidden for the whole process: use it at module import time, not while other threads import.
    """
    hidden = [name for name in names if name not in sys.modules and importlib.util.find_spec(name) is not None]
    already_imported = set(sys.modules)
    for name in hidden:
        sys.modules[name] = None  # pyright: ignore[reportArgumentType]  # None makes `import name` raise ImportError
    try:
        yield
    finally:
        for name in hidden:
            if name in sys.modules and sys.modules[name] is None:  # pyright: ignore[reportUnnecessaryComparison]
                del sys.modules[name]
        for module_name in sys.modules.keys() - already_imported:
            module_globals = getattr(sys.modules[module_name], "__dict__", {})
            for name in hidden:
                if name in module_globals and module_globals[name] is None:
                    module_globals[name] = _ImportOnFirstUse(name)
