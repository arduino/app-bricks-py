# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from functools import wraps
import inspect
import queue
import socket
import threading
import msgpack
import time
import os
from urllib.parse import urlparse
from .logger import Logger

logger = Logger("Bridge")


_reconnect_delay = 3.0  # seconds

# Error codes for RPC messages received from the RPC router. These are defined in the RPC router itself.
ROUTE_ALREADY_EXISTS_ERR = 0x05

# Error codes for RPC messages sent to Arduino_RPClite. These are defined in the lib itself.
MALFORMED_CALL_ERR = 0xFD
FUNCTION_NOT_FOUND_ERR = 0xFE
GENERIC_ERR = 0xFF


class Bridge:
    @staticmethod
    def notify(method_name: str, *params):
        """Sends a notification to the microcontroller without waiting for a response.

        Args:
            method_name (str): The name of the method to notify on the microcontroller.
            *params: The parameters to pass to the method.

        Examples:
            Bridge.notify("set_led", "green", True)
            Bridge.notify("log_message", "Hello, microcontroller!")
        """
        ClientServer().notify(method_name, *params)

    @staticmethod
    def call(method_name: str, *params, timeout: int = 10):
        """Calls a method on the microcontroller and waits for a response.
        Raises an exception if the call fails or times out.

        Args:
            method_name (str): The name of the method to call on the microcontroller.
            *params: The parameters to pass to the method.
            timeout (int, optional): The maximum time to wait for a response in seconds. Defaults to 10s.

        Raises:
            ValueError: If the method does not exist.
            TimeoutError: If the call takes more time than the specified timeout.
            RuntimeError: If the call fails unexpectedly.

        Examples:
            temperature = Bridge.call("get_temperature", "sensor1")
            print(f"Temperature: {temperature}")
        """
        return ClientServer().call(method_name, *params, timeout=timeout)

    @staticmethod
    def provide(method_name: str, handler: callable):
        """Makes a method available to the microcontroller, so it can call it remotely.
        The handler should be a callable that can take arguments.

        Args:
            method_name (str): The name under which the function should be provided to the microcontroller.
            handler (callable): The function to call when the microcontroller requires it.

        Raises:
            ValueError: If handler is not callable.
            RuntimeError: If the request fails unexpectedly.

        Examples:
            def get_country(lon: str, lat: str) -> str:
                ... lookup country by lon and lat ...
                return country_name

            Bridge.provide("get_country", get_country)
        """
        ClientServer().provide(method_name, handler)

    @staticmethod
    def unprovide(method_name: str):
        """Makes a method no more available to the microcontroller.

        Args:
            method_name (str): The name under which the function is already provided to the microcontroller.

        Raises:
            RuntimeError: If the request fails unexpectedly.

        Examples:
            Bridge.unprovide("get_country")
        """
        ClientServer().unprovide(method_name)


def notify(method_name: str = None, address: str = "unix:///var/run/arduino-router.sock"):
    """Decorator that transforms a function into a notification for the microcontroller.

    When the decorated function is called, an RPC 'notify' (fire-and-forget) is sent
    to the microcontroller. The notify's arguments are taken from the decorated function's arguments.
    The RPC method name defaults to the decorated function's name if not specified.

    Args:
        method_name (str, optional): The name of the RPC method to call. Defaults to the decorated function's name.
        address (str, optional): The address of the microcontroller router to connect to. Can be a TCP socket or a Unix socket. Defaults to "unix:///var/run/arduino-router.sock".

    Raises:
        TypeError: If the decorated function is called with unexpected keyword arguments.

    Examples:
        @notify()
        def set_led(color: str, status: bool): ... # Body is not needed

        @notify("leds.green.set_status", timeout=3)
        def set_green_led(status: bool): ...

        set_led("green", True) # Sends "set_led" RPC notification
        set_green_led(True) # Sends "leds.green.set_status" RPC notification
    """
    instance = ClientServer(address)

    def decorator(func):
        actual_method_name = method_name if method_name is not None else func.__name__

        func_is_unsupported = _is_unbound_or_class_method(func)
        if func_is_unsupported:
            raise TypeError(f"'{func.__name__}' is expected to be a function but is a method or a classmethod.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Any kwargs passed to the decorated function are unexpected.
            if kwargs:
                raise TypeError(f"Unexpected {list(kwargs.keys())} keyword args: only positional args are supported.")

            instance.notify(actual_method_name, *args)

        return wrapper

    return decorator


def call(method_name: str = None, timeout: int | None = 10, address: str = "unix:///var/run/arduino-router.sock"):
    """Decorator that transforms a function into an RPC notification.

    When the decorated function is called, an RPC 'call' (request and response) is sent
    to the microcontroller. The call's arguments are taken from the decorated function's arguments.
    The RPC method name defaults to the decorated function's name if not specified.
    A default timeout for the RPC call can be set via the decorator but it can be overridden.
    by passing a 'timeout' keyword argument when calling the decorated function.

    Args:
        method_name (str, optional): The name of the RPC method to call. Defaults to the decorated function's name.
        timeout (int, optional): The maximum time to wait for a response in seconds. If None, waits indefinitely. Defaults to 10s.
        address (str, optional): The address of the microcontroller router to connect to. Can be a TCP socket or a Unix socket. Defaults to "unix:///var/run/arduino-router.sock".

    Raises:
        TypeError: If the decorated function is called with unexpected keyword arguments.
        ValueError: If the method does not exist.
        TimeoutError: If the call takes more time than the specified timeout.
        RuntimeError: If the call fails unexpectedly.

    Examples:
        @call()
        def get_led(color: str) -> bool: ... # Body is not needed

        @call("leds.green.status", timeout=3)
        def get_green_led() -> bool: ...

        state = get_led("green")
        state = get_green_led()
    """
    instance = ClientServer(address)

    def decorator(func):
        actual_method_name = method_name if method_name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the first argument is a timeout and use it if so
            actual_timeout = kwargs.pop("timeout", timeout)

            func_is_unsupported = _is_unbound_or_class_method(func)
            if func_is_unsupported:
                raise TypeError(f"'{func.__name__}' is expected to be a function but is a method or a classmethod.")

            # Any remaining kwargs passed to the decorated function are unexpected.
            if kwargs:
                raise TypeError(f"Unexpected {list(kwargs.keys())} keyword args: only positional args are supported.")

            if actual_timeout is not None:
                return instance.call(actual_method_name, *args, timeout=actual_timeout)
            else:
                return instance.call(actual_method_name, *args)  # Wait indefinitely for a response

        return wrapper

    return decorator


def provide(method_name=None, address: str = "unix:///var/run/arduino-router.sock"):
    """Decorator that makes a method available to the microcontroller, so it can call it remotely.

    The decorated function is automatically registered using its own name as method name,
    unless `method_name` is provided.

    Args:
        method_name (str, optional): The name under which the function should be registered.
        address (str, optional): The address of the microcontroller router to connect to. Can be a TCP socket or a Unix socket. Defaults to "unix:///var/run/arduino-router.sock".

    Raises:
        ValueError: If handler is not callable.
        RuntimeError: If the request fails unexpectedly.

    Examples:
        @provide()
        def get_country(lon: str, lat: str) -> str:
            ... lookup country by lon and lat ...
            return country_name

        @provide("custom.rpc.name")
        def another_handler(param):
            ... logic ...
    """
    instance = ClientServer(address)

    def decorator(func):
        actual_method_name = method_name if method_name is not None else func.__name__

        try:
            instance.provide(actual_method_name, func)
        except Exception as e:
            raise RuntimeError(f"Failed to register method '{actual_method_name}': {e}") from e

        # Return the original function, registration is only a side-effect
        return func

    return decorator


# Helper that implements a heuristic to determine if a function is a method (unbound) or @classmethod
def _is_unbound_or_class_method(func):
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


class SingletonMeta(type):
    _instance = None
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ClientServer(metaclass=SingletonMeta):
    def __init__(self, address: str = "unix:///var/run/arduino-router.sock"):
        self.next_msgid = 0
        self.next_msgid_lock = threading.Lock()
        self.callbacks = {}  # msgid -> (on_result, on_error)
        self.callbacks_lock = threading.Lock()
        self.handlers = {}  # method name -> function
        self.handlers_lock = threading.Lock()

        address_config = os.environ.get("APP_SOCKET", address)
        urlparsed = urlparse(address_config)
        if urlparsed.scheme == "unix":
            self.socket_type = "unix"
            self._peer_addr = urlparsed.path
        elif urlparsed.scheme == "tcp":
            self.socket_type = "tcp"
            self._peer_addr = (urlparsed.hostname, urlparsed.port)

        self._conn = None
        self._conn_lock = threading.Lock()
        self._is_connected_flag = threading.Event()  # This avoids locking recv calls

        self._connect()

        self._read_thread = threading.Thread(target=self._conn_manager, name="Bridge.read_loop", daemon=True)
        self._read_thread.start()

    def notify(self, method_name: str, *params):
        """Sends a notification to the server without waiting for a response."""
        request = [2, method_name, params]
        try:
            self._send_bytes(msgpack.packb(request))
        except ConnectionError:
            # Fire-and-forget semantics
            pass
        except Exception as e:
            logger.error(f"Failed to send notification for method '{method_name}': {e}")

    def call(self, method_name: str, *params, timeout: int = 10):
        """Calls a method on the server and waits for a response."""
        msgid = self._increment_next_msgid()
        request = [0, msgid, method_name, params]

        resp_queue = queue.Queue(maxsize=1)

        def on_result(result):
            resp_queue.put((True, result))

        def on_error(error):
            resp_queue.put((False, error))

        with self.callbacks_lock:
            self.callbacks[msgid] = (on_result, on_error)

        try:
            self._send_bytes(msgpack.packb(request))
        except Exception as e:
            with self.callbacks_lock:
                self.callbacks.pop(msgid, None)
            raise RuntimeError(f"Failed to call method '{method_name}': {e}") from e

        try:
            (success, response) = resp_queue.get(timeout=timeout)
            if success:
                return response
            else:
                err_code, err_msg = response
                raise ValueError(f"Request '{method_name}' failed: {err_msg} ({err_code})")
        except queue.Empty:
            # Timed out waiting for response
            with self.callbacks_lock:
                if self.callbacks.pop(msgid, None):
                    try:
                        self.notify("$/cancelRequest", msgid)
                    except Exception:
                        pass
            raise TimeoutError(f"Request '{method_name}' timed out after {timeout}s")
        except Exception:
            with self.callbacks_lock:  # Ensure callback is cleaned up on any exception path
                self.callbacks.pop(msgid, None)
            raise

    def provide(self, method_name: str, handler):
        """Makes a method available to the microcontroller, so it can call it remotely.
        The handler should be a callable that can take arguments.
        """
        if not callable(handler):
            raise ValueError("Handler must be a callable.")

        try:
            self.call("$/register", method_name)
        except Exception as e:
            raise RuntimeError(f"Failed to register method '{method_name}': {e}")

        with self.handlers_lock:
            self.handlers[method_name] = handler

    def unprovide(self, method_name: str):
        """Makes a method no more available to the microcontroller."""
        with self.handlers_lock:
            if method_name not in self.handlers:
                return  # Nothing to unregister

        try:
            self.call("$/unregister", method_name)
        except Exception as e:
            raise RuntimeError(f"Failed to unregister method '{method_name}': {e}")

        try:
            with self.handlers_lock:
                self.handlers.pop(method_name, None)
        except KeyError:
            return  # Method was already unregistered

    def _increment_next_msgid(self):
        """Increments the next message ID, ensuring it is unique and within bounds."""
        with self.next_msgid_lock:
            self.next_msgid = (self.next_msgid + 1) % (2**32)
            while self.next_msgid in self.callbacks:
                self.next_msgid = (self.next_msgid + 1) % (2**32)
            return self.next_msgid

    def _conn_manager(self):
        """Manages connection and reconnection attempts. Once the connection is established, delegates to the read loop."""
        while True:
            # Ensure we're connected to the router
            self._connect()  # This retries internally until connected
            self._read_loop()  # This blocks until connection is lost or errors out
            time.sleep(_reconnect_delay)  # Wait before trying to reconnect

    def _connect(self):
        """Makes sure we're connected to the router by retrying periodically until we have a clean connection.
        This method **must be** the only one allowed to set _is_connected_flag, this allows us to use a
        lockless algorithm for connection management, in particular for recv calls.
        """
        if self._is_connected():
            return

        if self._conn:
            with self._conn_lock:
                # We're in a dirty state since we have a valid _conn object but looks like we're not connected.
                # Clean up the old, probably broken, connection object.
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

        self._is_connected_flag.clear()

        while not self._is_connected():
            try:
                with self._conn_lock:
                    if self.socket_type == "unix":
                        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        self._conn.connect(self._peer_addr)
                    elif self.socket_type == "tcp":
                        self._conn = socket.create_connection(self._peer_addr, timeout=5)
                self._conn.settimeout(None)  # Set blocking recv
                self._is_connected_flag.set()

                # Run this function in a separate thread for receiving the call response as it would block waiting for the response
                def register_methods_on_reconnect():
                    with self.handlers_lock:
                        for method in self.handlers.keys():
                            try:
                                self.call("$/register", method)
                            except Exception as e:
                                logger.error(f"Failed to re-register method '{method}' after reconnection: {e}")

                if self.handlers:
                    t = threading.Thread(target=register_methods_on_reconnect, name="Bridge.register_methods_on_reconnect", daemon=True)
                    t.start()

                return
            except Exception as e:
                logger.error(f"Failed to connect to router: {e}")
                time.sleep(_reconnect_delay)  # Try to connect again after a delay

    def _is_connected(self) -> bool:
        """Performs a lightweight check to verify if the connection is usable and active.
        Takes care to not block or remove bytes from the buffer.
        """
        if self._conn is None:
            return False

        try:
            # Make sure we don't block or remove bytes from the buffer (peek only)
            data = self._conn.recv(8, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            if len(data) == 0:
                return False
            return True
        except BlockingIOError:
            return True  # Socket is open and reading from it would block
        except ConnectionResetError as e:
            logger.warning(f"Connection reset in connection loop: {e}")
            return False  # Socket was closed for some other reason
        except Exception as e:
            logger.error(f"Unexpected error while checking socket status: {e}")
            return False  # Assume the socket is broken for any other exception

    def _read_loop(self):
        """The core loop that reads and processes messages from the active socket."""
        unpacker = msgpack.Unpacker()
        try:
            while True:
                try:
                    data = self._conn.recv(4096)
                    if not data:
                        logger.info("Connection closed by router")
                        break
                    unpacker.feed(data)
                    for msg in unpacker:
                        self._handle_msg(msg)
                except ConnectionResetError as e:
                    logger.warning(f"Connection reset in read loop: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in read loop: {e}")
                    continue
        finally:
            # Connection was lost unexpectedly but we were meant to be running, tell the user
            self._fail_pending_callbacks(ConnectionError("Connection to router lost."))

    # TODO: verify if this is still needed
    def _decode_method(self, method_name: any) -> str:
        """Decodes the method name from bytes to string if necessary."""
        if isinstance(method_name, bytes):
            return method_name.decode()
        if isinstance(method_name, str):
            return method_name
        else:
            raise ValueError(f"Invalid method name type: {type(method_name)}. Expected str or bytes.")

    def _handle_msg(self, msg: list):
        """Processes a single deserialized MessagePack-RPC message."""
        if not msg or not isinstance(msg, list):
            logger.warning("Invalid RPC message received (must be a non-empty list).")
            return

        msg_type = msg[0]
        try:
            if msg_type == 0:  # Request: [0, msgid, method, params]
                if len(msg) != 4:
                    raise ValueError(f"Invalid RPC request: expected length 4, got {len(msg)}")
                _, msgid, method, params = msg
                if not isinstance(params, (list, tuple)):
                    raise ValueError("Invalid RPC request params: expected array or tuple")

                method_name = self._decode_method(method)

                with self.handlers_lock:
                    handler = self.handlers.get(method_name)

                if handler:
                    try:
                        result = handler(*params)  # Unpack params
                        self._send_response(msgid, None, result)
                    except Exception as e:
                        logger.error(f"Failed to run user-provided call handler for method '{method_name}': {e}")
                        self._send_response(msgid, e, None)
                else:
                    self._send_response(msgid, NameError(f"Method not found: '{method_name}'", method_name), None)

            elif msg_type == 1:  # Response: [1, msgid, error, result]
                if len(msg) != 4:
                    raise ValueError(f"Invalid RPC response: expected length 4, got {len(msg)}")
                _, msgid, error, result = msg
                if error and (not isinstance(error, list) or len(error) < 2):
                    raise ValueError("Invalid error format in RPC response")

                with self.callbacks_lock:
                    cbs = self.callbacks.pop(msgid, None)
                if cbs:
                    on_result, on_error = cbs
                    if result is None and error is None:
                        on_result(None)
                    else:
                        # Treat ROUTE_ALREADY_EXISTS_ERR error as OK. It only means that the router already knows about the
                        # method and registering it is not necessary. It's an internal and recoverable situation.
                        if result is not None or (error is not None and error[0] == ROUTE_ALREADY_EXISTS_ERR):
                            on_result(result)
                        elif error is not None:
                            on_error(error)
                        else:
                            on_result([GENERIC_ERR, "Unknown error occurred."])
                else:
                    logger.warning(f"Response for unknown msgid {msgid} received.")

            elif msg_type == 2:  # Notification: [2, method, params]
                if len(msg) != 3:
                    raise ValueError(f"Invalid RPC notification: expected length 3, got {len(msg)}")
                _, method, params = msg
                if not isinstance(params, (list, tuple)):
                    raise ValueError("Invalid RPC notification params: expected array or tuple")

                method_name = self._decode_method(method)

                with self.handlers_lock:
                    handler = self.handlers.get(method_name)

                if handler:
                    try:
                        handler(*params)
                    except Exception as e:
                        logger.error(f"Failed to run user-provided notification handler for method '{method_name}': {e}")
            else:
                logger.warning(f"Invalid RPC message type received: {msg_type}")

        except ValueError as ve:
            logger.error(f"Message validation error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error while handling message: {e}")

    def _fail_pending_callbacks(self, reason: Exception):
        """Invokes error callbacks for all pending requests and clears their callbacks."""
        with self.callbacks_lock:
            for _, (_, on_error) in list(self.callbacks.items()):
                if on_error:
                    try:
                        on_error(reason)
                    except Exception as e:
                        logger.error(f"Failed to run 'on_error' callback: {e}")
            self.callbacks.clear()

    def _send_response(self, msgid: int, error, response):
        """Helper to pack and send a response message."""
        err = None
        if error is not None:
            err_code = GENERIC_ERR
            err_msg = str(error)
            if isinstance(error, NameError):
                err_code = FUNCTION_NOT_FOUND_ERR
            elif isinstance(error, TypeError) or isinstance(error, ValueError):
                err_code = MALFORMED_CALL_ERR
            err = [err_code, err_msg]

        msg = [1, msgid, err, response]
        try:
            self._send_bytes(msgpack.packb(msg))
        except ConnectionError:
            pass  # Response sending is best-effort if connection drops while handling request.
        except Exception as e:  # e.g., msgpack encoding error
            logger.error(f"Failed to pack/send response: {e}")

    def _send_bytes(self, packed_data: bytes):
        """Sends packed data, handling connection waits and errors."""
        if not self._is_connected_flag.is_set():
            # Wait hoping for an auto-reconnection by _conn_manager
            if not self._is_connected_flag.wait(timeout=_reconnect_delay):
                raise ConnectionError(f"Not connected to router, send failed.")

        with self._conn_lock:
            if self._conn is None:
                raise ConnectionError(f"No connection object for router, send failed.")
            try:
                self._conn.sendall(packed_data)
            except socket.error as e:
                raise ConnectionError(f"Send failed due to socket error: {e}")
