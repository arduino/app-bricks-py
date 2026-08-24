# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""App-wide singleton layer over the instance-based arduino-router-bridge library.

Apps talk to the single router socket configured by the app environment (APP_SOCKET),
so this module manages one shared connection and exposes the ergonomic API built on
it: the `Bridge` facade and the `@notify`, `@call` and `@provide` decorators.
"""

from functools import wraps
import inspect
import os
import threading

from arduino.router_bridge import DEFAULT_ADDRESS
from arduino.router_bridge import Bridge as RouterBridge

__all__ = ["Bridge", "notify", "call", "provide"]

_bridge: RouterBridge | None = None
_bridge_lock = threading.Lock()


def _get_bridge() -> RouterBridge:
    """Returns the app-wide shared bridge, creating it on first use and (re)connecting it when needed.
    The router address is taken from the APP_SOCKET environment variable, falling back to the library default.
    """
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = RouterBridge(os.environ.get("APP_SOCKET", DEFAULT_ADDRESS))
        _bridge.connect()  # No-op when already connected, restarts a disconnected bridge
        return _bridge


class Bridge:
    """Process-wide access to the microcontroller RPC bridge, connected lazily on first use."""

    @staticmethod
    def notify(method_name: str, *params: object) -> None:
        """Sends a notification to the microcontroller without waiting for a response.
        Best-effort: never blocks waiting for a connection, the notification is
        dropped if the router is not connected.

        Args:
            method_name (str): The name of the method to notify on the microcontroller.
            *params: The parameters to pass to the method.

        Examples:
            Bridge.notify("set_led", "green", True)
            Bridge.notify("log_message", "Hello, microcontroller!")
        """
        _get_bridge().notify(method_name, *params)

    @staticmethod
    def call(method_name: str, *params, timeout: float | None = 10):
        """Calls a method on the microcontroller and waits for a response.
        Raises an exception if the call fails or times out.

        Args:
            method_name (str): The name of the method to call on the microcontroller.
            *params: The parameters to pass to the method.
            timeout (float, optional): The maximum time to wait for a response in seconds. If None, waits indefinitely. Defaults to 10s.

        Raises:
            ValueError: If the method does not exist or the call fails.
            TimeoutError: If the call takes more time than the specified timeout.
            ConnectionError: If the connection drops while waiting for the response.
            RuntimeError: If invoked from a provided handler (nested calls are not supported), or if the call fails unexpectedly.

        Examples:
            temperature = Bridge.call("get_temperature", "sensor1")
            print(f"Temperature: {temperature}")
        """
        return _get_bridge().call(method_name, *params, timeout=timeout)

    @staticmethod
    def provide(method_name: str, handler):
        """Makes a method available to the microcontroller, so it can call it remotely.
        The handler should be a callable that can take arguments.

        The handler is registered with the router as soon as a connection is available
        and re-registered transparently on every reconnection. Handlers run sequentially
        on a dedicated thread: they may send notifications, but must not call back into
        the bridge with `call` (nested calls are rejected with a RuntimeError).

        Args:
            method_name (str): The name under which the function should be provided to the microcontroller.
            handler (callable): The function to call when the microcontroller requires it.

        Raises:
            ValueError: If handler is not callable.

        Examples:
            def get_country(lon: str, lat: str) -> str:
                ... lookup country by lon and lat ...
                return country_name

            Bridge.provide("get_country", get_country)
        """
        _get_bridge().provide(method_name, handler)

    @staticmethod
    def unprovide(method_name: str) -> None:
        """Makes a method no more available to the microcontroller.

        Args:
            method_name (str): The name under which the function is already provided to the microcontroller.

        Examples:
            Bridge.unprovide("get_country")
        """
        _get_bridge().unprovide(method_name)


def notify(method_name: str | None = None):
    """Decorator that transforms a function into a notification for the microcontroller.

    When the decorated function is called, an RPC 'notify' (fire-and-forget) is sent
    to the microcontroller. The notify's arguments are taken from the decorated function's arguments.
    The RPC method name defaults to the decorated function's name if not specified.
    The connection is established lazily, on the first invocation of the decorated function.

    Args:
        method_name (str, optional): The name of the RPC method to call. Defaults to the decorated function's name.

    Raises:
        TypeError: If the decorated function is called with unexpected keyword arguments.

    Examples:
        @notify()
        def set_led(color: str, status: bool): ... # Body is not needed

        @notify("leds.green.set_status")
        def set_green_led(status: bool): ...

        set_led("green", True) # Sends "set_led" RPC notification
        set_green_led(True) # Sends "leds.green.set_status" RPC notification
    """

    def decorator[**P](func: Callable[P, object]) -> Callable[P, None]:
        actual_method_name = method_name if method_name is not None else func.__name__

        if _is_unbound_or_class_method(func):
            raise TypeError(f"'{func.__name__}' is expected to be a function but is a method or a classmethod.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Any kwargs passed to the decorated function are unexpected
            if kwargs:
                raise TypeError(f"Unexpected {list(kwargs.keys())} keyword args: only positional args are supported.")

            _get_bridge().notify(actual_method_name, *args)

        return wrapper

    return decorator


def call(method_name: str | None = None, timeout: float | None = 10):
    """Decorator that transforms a function into an RPC call to the microcontroller.

    When the decorated function is called, an RPC 'call' (request and response) is sent
    to the microcontroller. The call's arguments are taken from the decorated function's arguments.
    The RPC method name defaults to the decorated function's name if not specified.
    A default timeout for the RPC call can be set via the decorator but it can be overridden
    by passing a 'timeout' keyword argument when calling the decorated function.
    The connection is established lazily, on the first invocation of the decorated function.

    Args:
        method_name (str, optional): The name of the RPC method to call. Defaults to the decorated function's name.
        timeout (float, optional): The maximum time to wait for a response in seconds. If None, waits indefinitely. Defaults to 10s.

    Raises:
        TypeError: If the decorated function is called with unexpected keyword arguments.
        ValueError: If the method does not exist or the call fails.
        TimeoutError: If the call takes more time than the specified timeout.
        ConnectionError: If the connection drops while waiting for the response.
        RuntimeError: If the call fails unexpectedly.

    Examples:
        @call()
        def get_led(color: str) -> bool: ... # Body is not needed

        @call("leds.green.status", timeout=3)
        def get_green_led() -> bool: ...

        state = get_led("green")
        state = get_green_led()
    """

    def decorator[R](func: Callable[..., R]) -> Callable[..., R]:
        actual_method_name = method_name if method_name is not None else func.__name__

        if _is_unbound_or_class_method(func):
            raise TypeError(f"'{func.__name__}' is expected to be a function but is a method or a classmethod.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # An optional 'timeout' keyword overrides the decorator's default
            actual_timeout = kwargs.pop("timeout", timeout)

            # Any remaining kwargs passed to the decorated function are unexpected
            if kwargs:
                raise TypeError(f"Unexpected {list(kwargs.keys())} keyword args: only positional args are supported.")

            return _get_bridge().call(actual_method_name, *args, timeout=actual_timeout)

        return wrapper

    return decorator


def provide(method_name: str | None = None):
    """Decorator that makes a method available to the microcontroller, so it can call it remotely.

    The decorated function is automatically registered using its own name as method name,
    unless `method_name` is provided. The registration with the router happens as soon as
    a connection is available and is renewed transparently on every reconnection.
    The decorated function runs on a dedicated thread: it may send notifications, but must
    not call back into the bridge with `call` (nested calls are rejected with a RuntimeError).

    Args:
        method_name (str, optional): The name under which the function should be registered.

    Examples:
        @provide()
        def get_country(lon: str, lat: str) -> str:
            ... lookup country by lon and lat ...
            return country_name

        @provide("custom.rpc.name")
        def another_handler(param):
            ... logic ...
    """

    def decorator[F: Callable[..., object]](func: F) -> F:
        actual_method_name = method_name if method_name is not None else func.__name__

        if _is_unbound_or_class_method(func):
            raise TypeError(f"'{func.__name__}' is expected to be a function but is a method or a classmethod.")

        _get_bridge().provide(actual_method_name, func)

        # Return the original function, registration is only a side-effect
        return func

    return decorator


# Helper that implements a heuristic to determine if a function is a method (unbound) or @classmethod
def _is_unbound_or_class_method(func: Callable[..., object]) -> bool:
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            return False
        first_param = params[0]
        return first_param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ) and first_param.name in ("self", "cls")
    except ValueError:
        return False
