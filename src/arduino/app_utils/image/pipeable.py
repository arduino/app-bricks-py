# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

"""
Decorator for adding pipe operator support to transformation functions.

This module provides a decorator that wraps static functions to support
the | (pipe) operator for functional composition.
"""

from typing import Callable
from functools import wraps


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
            return NotImplemented
        
        def composed(value):
            return other(self(value))
        
        return PipeableFunction(composed)
    
    def __repr__(self):
        """String representation of the pipeable function."""
        if self.args or self.kwargs:
            args_str = ', '.join(map(str, self.args))
            kwargs_str = ', '.join(f'{k}={v}' for k, v in self.kwargs.items())
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))
            return f"{self.__name__}({all_args})"
        return f"{self.__name__}()"


def pipeable(func: Callable) -> Callable:
    """
    Decorator that makes a function pipeable using the | operator.
    
    The decorated function can be used in two ways:
    1. Normal function call: func(args)
    2. Pipe operator: value | func or func | other_func
    
    Args:
        func: Function to make pipeable
        
    Returns:
        Wrapped function that supports pipe operations
        
    Examples:
        @pipeable
        def add_one(x):
            return x + 1
        
        result = 5 | add_one | add_one  -> 7
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and kwargs:
            # Both positional and keyword args - return partially applied
            return PipeableFunction(func, *args, **kwargs)
        elif args:
            # Only positional args - return partially applied
            return PipeableFunction(func, *args, **kwargs)
        elif kwargs:
            # Only keyword args - return partially applied
            return PipeableFunction(func, **kwargs)
        else:
            # No args - return pipeable version of original function
            return PipeableFunction(func)
    
    # Also add the pipeable functionality directly to the wrapper
    wrapper.__ror__ = lambda self, other: func(other)
    wrapper.__or__ = lambda self, other: PipeableFunction(lambda x: other(func(x)))
    
    return wrapper