# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Decorator for adding pipe operator support to transformation functions.

This module provides a decorator that wraps static functions to support
the | (pipe) operator for functional composition.

Note: Due to numpy's element-wise operator behavior, using the pipe operator
with numpy arrays (array | function) is not supported. Use function(array) instead.
"""

from typing import Callable


class PipeableFunction:
    """
    Wrapper class that adds pipe operator support to a function.

    This allows functions to be composed using the | operator in a left-to-right manner.
    """

    def __init__(self, func: Callable, *args, **kwargs):
        """
        Initialize a pipeable function.

        Args:
            func: The function to wrap
            *args: Positional arguments to partially apply
            **kwargs: Keyword arguments to partially apply
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """Call the wrapped function with combined arguments."""
        combined_args = self.args + args
        combined_kwargs = {**self.kwargs, **kwargs}
        return self.func(*combined_args, **combined_kwargs)

    def __ror__(self, other):
        """
        Right-hand side of pipe operator (|).

        This allows: value | pipeable_function

        Args:
            other: The value being piped into this function

        Returns:
            Result of applying this function to the value
        """
        return self(other)

    def __or__(self, other):
        """
        Left-hand side of pipe operator (|).

        This allows: pipeable_function | other_function

        Args:
            other: Another function to compose with

        Returns:
            A new pipeable function that combines both
        """
        if not callable(other):
            # Raise TypeError immediately instead of returning NotImplemented
            # This prevents Python from trying the reverse operation for nothing
            raise TypeError(f"unsupported operand type(s) for |: '{type(self).__name__}' and '{type(other).__name__}'")

        def composed(value):
            return other(self(value))

        return PipeableFunction(composed)

    def __repr__(self):
        """String representation of the pipeable function."""
        # Get function name safely
        func_name = getattr(self.func, "__name__", None)
        if func_name is None:
            func_name = getattr(type(self.func), "__name__", None)
        if func_name is None:
            from functools import partial

            if type(self.func) == partial:
                func_name = "partial"
        if func_name is None:
            func_name = "unknown"  # Fallback

        if self.args or self.kwargs:
            args_str = ", ".join(map(str, self.args))
            kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            return f"{func_name}({all_args})"
        return f"{func_name}()"
