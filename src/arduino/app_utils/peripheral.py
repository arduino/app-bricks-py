# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from functools import wraps
from typing import Any, overload

from . import peripheral_registry
from .peripheral_registry import Peripheral

__all__ = ["peripheral"]


@overload
def peripheral[C: Peripheral](user_class: None = None) -> Callable[[type[C]], type[C]]: ...


@overload
def peripheral[C: Peripheral](user_class: type[C]) -> type[C]: ...


def peripheral[C: Peripheral](user_class: type[C] | None = None) -> type[C] | Callable[[type[C]], type[C]]:
    """Class decorator marking a class as a peripheral, released automatically when the app shuts down.

    Every instance is registered so that its ``stop()`` is called during the application shutdown,
    after all the bricks have been stopped and within the shutdown time budget. This is what keeps
    an exclusive device, such as a CSI camera assigned by the platform's camera service, from being
    left behind when the process exits.

    Unlike ``@brick`` this does not manage the lifecycle: no ``start()`` is called for you and no
    ``loop``/``execute`` thread is run. The class only needs a ``stop()`` method, which must be safe
    to call on an instance that was never started or was already stopped. That requirement is part
    of the signature: a class without ``stop()`` does not satisfy ``Peripheral`` and a type checker
    rejects the decoration.

    Only a weak reference to each instance is kept, so decorating a class never keeps its instances
    alive. Can be used as ``@peripheral`` or ``@peripheral()``.

    Example:
        ```python
        @peripheral
        class MySensor:
            def start(self) -> None: ...
            def stop(self) -> None: ...  # released on shutdown even if never called explicitly
        ```
    """
    if user_class is None:  # Used as @peripheral()
        return _decorate_class
    return _decorate_class(user_class)  # Used as @peripheral


def _decorate_class[C: Peripheral](user_class: type[C]) -> type[C]:
    """Patches user_class.__init__ to register every new instance for release on shutdown."""
    original_init = user_class.__init__

    @wraps(original_init)
    def new_init(self: C, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Registered only once __init__ has completed, so an instance whose construction failed
        # is never asked to stop. The registry is looked up through its module at call time, like
        # @brick does with App, so tests can swap in a fresh one.
        peripheral_registry.Peripherals.register(self)

    user_class.__init__ = new_init
    return user_class
