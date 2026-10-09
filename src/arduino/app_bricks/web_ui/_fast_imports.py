# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Import-time shortcuts for the web stack (FastAPI, Uvicorn, Socket.IO), which is most of an app's start time on the board."""

import gc
import importlib
import importlib.util
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager


@contextmanager
def gc_paused() -> Generator[None]:
    """Keep the cyclic garbage collector off inside the block.

    Importing the web stack allocates enough objects to trigger ~50 collections that find almost nothing
    to free, about 0.2 s on the board. The previous state is restored on exit.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@contextmanager
def defer_pydantic_model_builds() -> Generator[None]:
    """Make pydantic models defined inside the block build their validators on first use instead of at class creation.

    Importing FastAPI builds ~25 pydantic models for its OpenAPI schema, about 0.35 s of app start time on the
    board. WebUI disables OpenAPI (``openapi_url=None``), so those models are never used and, deferred, never built.
    Models defined after the block, such as the app's own request bodies, are built as usual.

    It relies on a pydantic internal: if that is gone, models are built at import time as before.
    """
    try:
        from pydantic._internal._config import config_defaults  # pyright: ignore[reportPrivateImportUsage]
    except ImportError:
        yield
        return

    previous = config_defaults.get("defer_build", False)
    config_defaults["defer_build"] = True
    try:
        yield
    finally:
        config_defaults["defer_build"] = previous


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
