# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from functools import wraps


class BrickDecorator:
    """A class that acts as a namespace for the brick decorators to avoid name clashes with user code.
    - @brick is the main class decorator used to transform a class into an Arduino brick.
    - @brick.loop and @brick.execute are the method decorators used to hook them to the AppController.
    """

    def __call__(self, user_class=None):
        """Handles decorating the class.
        Can be used as @brick or @brick().
        """
        if user_class is None:  # Used as @brick()
            return self._decorate_class
        else:  # Used as @brick
            return self._decorate_class(user_class)

    def _decorate_class(self, user_class):
        """Patches user_class.__init__ method to automatically register every new instance with the central AppController."""
        original_init = user_class.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # We need to import 'app' here to avoid circular dependencies
            import arduino.app_utils.app as app

            if original_init is not None:
                original_init(self, *args, **kwargs)

            # Register the brick instance with the framework for lifecycle management
            app.App.register(self)

        user_class.__init__ = new_init
        return user_class

    def execute(self, _func=None):
        """Method decorator that marks a method as a one-shot, blocking tasks.
        The AppController will run this method only once, in a dedicated thread.
        Can be used as @brick.execute or @brick.execute().
        """

        def decorator(func):
            func._is_execute = True
            return func

        if _func is None:  # Used as @brick.execute()
            return decorator
        else:  # Used as @brick.execute
            return decorator(_func)

    def loop(self, _func=None):
        """Method decorator that marks a method as a non-blocking, iterative tasks.
        The AppController will run this method repeatedly, in a dedicated thread.
        Can be used as @brick.loop or @brick.loop().
        """

        def decorator(func):
            func._is_loop = True
            return func

        if _func is None:  # Used as @brick.loop()
            return decorator
        else:  # Used as @brick.loop
            return decorator(_func)


brick = BrickDecorator()
