# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from functools import wraps
from typing import Any, overload


class BrickDecorator:
    """A class that acts as a namespace for the brick decorators to avoid name clashes with user code.
    - @brick is the main class decorator used to transform a class into an Arduino brick.
    - @brick.loop and @brick.execute are the method decorators used to hook them to the AppController.
    """

    @overload
    def __call__(self, user_class: None = None) -> Callable[[type], type]: ...

    @overload
    def __call__[C](self, user_class: type[C]) -> type[C]: ...

    def __call__(self, user_class: type | None = None) -> type | Callable[[type], type]:
        """Handles decorating the class.
        Can be used as @brick or @brick().
        """
        if user_class is None:  # Used as @brick()
            return self._decorate_class
        else:  # Used as @brick
            return self._decorate_class(user_class)

    def _decorate_class[C](self, user_class: type[C]) -> type[C]:
        """Patches user_class.__init__ method to automatically register every new instance with the central AppController."""
        original_init = user_class.__init__

        @wraps(original_init)
        def new_init(self: C, *args: Any, **kwargs: Any) -> None:
            # We need to import 'app' here to avoid circular dependencies
            import arduino.app_utils.app as app

            if original_init is not None:
                original_init(self, *args, **kwargs)

            # Register the brick instance with the framework for lifecycle management
            app.App.register(self)

        user_class.__init__ = new_init
        return user_class

    def execute[F: Callable[..., object]](self, _func: F | None = None) -> F | Callable[[F], F]:
        """Method decorator that marks a method as a one-shot, blocking tasks.
        The AppController will run this method only once, in a dedicated thread.
        Can be used as @brick.execute or @brick.execute().
        """

        def decorator(func: F) -> F:
            setattr(func, "_is_execute", True)
            return func

        if _func is None:  # Used as @brick.execute()
            return decorator
        else:  # Used as @brick.execute
            return decorator(_func)

    def loop[F: Callable[..., object]](self, _func: F | None = None) -> F | Callable[[F], F]:
        """Method decorator that marks a method as a non-blocking, iterative tasks.
        The AppController will run this method repeatedly, in a dedicated thread.
        Can be used as @brick.loop or @brick.loop().
        """

        def decorator(func: F) -> F:
            setattr(func, "_is_loop", True)
            return func

        if _func is None:  # Used as @brick.loop()
            return decorator
        else:  # Used as @brick.loop
            return decorator(_func)


brick = BrickDecorator()
