# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Lazy package exports (PEP 562): names resolved from their submodule on first access, to keep app start fast."""

import importlib
from collections.abc import Callable, Mapping, MutableMapping


def lazy_exports(
    package: str, namespace: MutableMapping[str, object], exports: Mapping[str, str]
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Build the module ``__getattr__`` and ``__dir__`` of a package whose exports load on first access.

    Assign the result under ``if not TYPE_CHECKING:`` and import the same names under ``if TYPE_CHECKING:``:
    type checkers then resolve each name, and still reject unknown ones, which a visible module ``__getattr__``
    would make them accept.

    Args:
        package (str): The package ``__name__``.
        namespace (MutableMapping[str, object]): The package ``globals()``, where a resolved name is cached so
            that later accesses skip ``__getattr__``.
        exports (Mapping[str, str]): Exported name to the submodule, relative to the package, that defines it.

    Returns:
        tuple[Callable[[str], object], Callable[[], list[str]]]: The ``__getattr__`` and ``__dir__`` functions.
    """

    def __getattr__(name: str) -> object:
        submodule = exports.get(name)
        if submodule is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f".{submodule}", package), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *exports})

    return __getattr__, __dir__
