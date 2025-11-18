# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import inspect


def _has_callable_method(obj_or_cls, method_name):
    """Checks if an object or class has a callable method with the correct signature.
    The method must only accept the `self` parameter.
    This function correctly handles both bound methods (on instances) and
    unbound functions (on classes).

    Args:
        obj_or_cls: The object instance or class to check.
        method_name: The name of the method to check.

    Returns:
        bool: True if the method exists and has the right signature, False otherwise.

    Raises:
        TypeError: If the method exists but has an incorrect signature.
    """
    if not hasattr(obj_or_cls, method_name):
        return False

    method = getattr(obj_or_cls, method_name)
    if not callable(method):
        return False

    # Handle both bound and unbound methods
    try:
        func = method.__func__
    except AttributeError:
        func = method

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) == 1:
        if params[0].name == "self":
            return True

        # Wrong parameter name
        raise TypeError(
            f"Method '{method_name}' has an invalid signature. "
            f"The '{method_name}' method is expected to only have the 'self' parameter, "
            f"but it is defined with the name {params[0].name}. Please rename the "
            f"method signature to avoid conflict."
        )
    elif len(params) > 1:
        # Wrong parameter count
        raise TypeError(
            f"Method '{method_name}' has an invalid signature. "
            f"The '{method_name}' method is expected to only have the 'self' parameter, "
            f"but it is defined with {len(params)} total parameters. Please correct the "
            f"method signature to avoid conflict."
        )
    else:
        # No parameters at all
        raise TypeError(
            f"Method '{method_name}' has an invalid signature. "
            f"The '{method_name}' method is expected to only have the 'self' parameter, "
            f"but it is defined with no parameters. Please correct the method signature to avoid conflict."
        )


def _brick_name(brick) -> str:
    return type(brick).__name__
