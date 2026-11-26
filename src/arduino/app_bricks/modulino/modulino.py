# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, override

from arduino.app_utils import AppError, Bridge, brick
from arduino.app_utils.ledmatrix import MODULINO_MATRIX, Frame, FrameDesigner

_CONNECT_HINT = (
    "Check that the Modulino is connected to the Qwiic connector of the board (or to the hub port given) "
    "and, if you changed its I2C address, that you pass the new one."
)
_SKETCH_HINT = "Check that the sketch of the app is running: the Modulino brick adds the ModulinoBridge library to it."
_DUPLICATE_HINT = "Another Modulino of the same type may already use this I2C address on the same bus or hub port."


class ModulinoError(AppError):
    """Raised when the board can't do an operation on a Modulino, for example because the Modulino doesn't answer.

    Invalid arguments raise `TypeError` or `ValueError` instead.
    """


def _check_int(name: str, value: object, minimum: int | None = None, maximum: int | None = None) -> None:
    """Raise TypeError if `value` is not an integer, ValueError if it is out of `minimum`..`maximum`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and maximum is not None and not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    if minimum is not None and maximum is None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _check_number(name: str, value: object) -> None:
    """Raise TypeError if `value` is not an int or a float (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")


def _check_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def _check_callable(callback: object) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")


class RGBColor(NamedTuple):
    """An RGB color, 0-255 per channel.

    It is a tuple `(r, g, b)`, so it can be unpacked (`r, g, b = color`) and passed to libraries
    that take an RGB tuple, such as Pillow. OpenCV takes colors in BGR order: pass `color[::-1]`.
    """

    r: int
    """Red, 0-255."""
    g: int
    """Green, 0-255."""
    b: int
    """Blue, 0-255."""


HUB_PORT_COUNT = 8


class ModulinoHubPort:
    """A port of a Modulino Hub. Get it with `ModulinoHub.port()` and pass it to a Modulino constructor."""

    def __init__(self, hub_address: int, port: int) -> None:
        """Initialize a ModulinoHubPort.

        Args:
            hub_address (int): The I2C address of the hub.
            port (int): The port number, 0-7.
        """
        self.hub_address = hub_address
        self.port = port


def _check_hub_port(hub_port: object) -> None:
    """Raise TypeError if `hub_port` is neither None nor a ModulinoHubPort."""
    if hub_port is not None and not isinstance(hub_port, ModulinoHubPort):
        raise TypeError("hub_port must be a ModulinoHubPort, get it with ModulinoHub.port()")


class ModulinoHub:
    """Modulino Hub (I2C multiplexer with 8 ports).

    It holds no resources: the hub is created on the MCU together with the first
    Modulino that uses one of its ports, and released with the last one.
    """

    def __init__(self, address: int = 0x70) -> None:
        """Initialize a ModulinoHub with the given I2C address.

        Args:
            address (int): The I2C address of the hub, set by the A0-A2 jumpers (0x70-0x77). Defaults to 0x70.

        Raises:
            TypeError: If `address` is not an integer.
            ValueError: If `address` is not between 0x70 and 0x77.
        """
        _check_int("address", address)
        if not 0x70 <= address <= 0x77:
            raise ValueError(f"invalid hub address 0x{address:x}, must be 0x70-0x77")
        self.address = address

    def port(self, n: int) -> ModulinoHubPort:
        """Get a port of the hub.

        Args:
            n (int): The port number, 0-7.

        Returns:
            ModulinoHubPort: The port, to pass as `hub_port` to a Modulino constructor.

        Raises:
            TypeError: If `n` is not an integer.
            ValueError: If `n` is not between 0 and 7.
        """
        _check_int("n", n, 0, HUB_PORT_COUNT - 1)
        return ModulinoHubPort(self.address, n)


@brick
class Modulino:
    """Base class of the Modulini.

    Don't instantiate it: use the class of your Modulino (`ModulinoThermo`, `ModulinoButtons`, ...).
    Each subclass sets `_TYPE`, the type name known by the MCU.
    """

    _TYPE = ""

    def __init__(self, i2c_address: int, hub_port: ModulinoHubPort | None = None) -> None:
        """Create the Modulino on the board.

        Args:
            i2c_address (int): The I2C address of the Modulino, 0xFF for autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If `Modulino` is instantiated directly, or an argument has the wrong type.
            ValueError: If `i2c_address` is not between 0x00 and 0xFF.
            ModulinoError: If the board can't create the Modulino.
        """
        if not type(self)._TYPE:
            raise TypeError("Modulino is a base class: use the class of your Modulino, e.g. ModulinoThermo")
        _check_int("i2c_address", i2c_address, 0x00, 0xFF)
        _check_hub_port(hub_port)
        self.i2c_address = i2c_address
        self.hub_address = hub_port.hub_address if hub_port is not None else 0
        self.port = hub_port.port if hub_port is not None else 0
        # The MCU builds the RPC name and returns "" if the modulino can't be created
        # (e.g. a modulino of the same type and address already exists on the same bus or hub port).
        # The name is also the key the MCU uses in delete_modulino and poll_subscribe/unsubscribe.
        error = f"cannot create modulino {self._TYPE} at 0x{i2c_address:x}{self._location()}"
        try:
            name = Bridge.call("create_modulino", self._TYPE, i2c_address, self.hub_address, self.port)
        except Exception as e:
            raise ModulinoError(error, hint=_SKETCH_HINT) from e
        if not isinstance(name, str) or not name:
            raise ModulinoError(error, hint=_DUPLICATE_HINT)
        self.name = name

    def _location(self) -> str:
        return f" on hub 0x{self.hub_address:x} port {self.port}" if self.hub_address else ""

    def start(self) -> None:
        """Start the Modulino: the board initializes it.

        Raises:
            ModulinoError: If the Modulino doesn't answer.
        """
        if not self._call("start", "begin"):
            raise ModulinoError(f"modulino {self.name}: cannot start", hint=_CONNECT_HINT)

    def stop(self) -> None:
        """Stop the Modulino: the board stops its notifications and turns its outputs off.

        The outputs are the motors, LEDs, sound and vibration; the Latch Relay keeps its position.

        Raises:
            ModulinoError: If the board can't stop the Modulino.
        """
        try:
            Bridge.call("delete_modulino", self.name)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot stop") from e

    # Helpers for the RPC methods of the modulino, named "<name>:<method>" on the MCU.
    # `what` describes the operation in the error message: "modulino <name>: cannot <what>".

    def _call(self, what: str, method: str, *args: Any) -> Any:  # noqa: ANN401 the result type is set by the MCU method
        """Call an RPC method and return its result as is (None for the void methods)."""
        try:
            return Bridge.call(f"{self.name}:{method}", *args)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot {what}") from e

    def _command(self, what: str, method: str, *args: Any) -> None:
        """Call an RPC method that returns True on success."""
        if not self._call(what, method, *args):
            raise ModulinoError(f"modulino {self.name}: cannot {what}")

    def _read(self, what: str, method: str, *args: Any) -> Any:  # noqa: ANN401 the result type is set by the MCU method
        """Call an RPC method that returns a negative value when the read fails."""
        value = self._call(what, method, *args)
        if value < 0:
            raise ModulinoError(f"modulino {self.name}: cannot {what}: read failed")
        return value

    def _subscribe(self, what: str, handler: Callable[..., object], period_ms: int) -> None:
        """Provide the handler of the notifications of the modulino and start them every `period_ms`."""
        try:
            Bridge.provide(f"{self.name}_callback", handler)
            Bridge.notify("poll_subscribe", self.name, period_ms)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot {what}") from e

    def _unsubscribe(self, what: str) -> None:
        """Stop the notifications of the modulino."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot {what}") from e


class ModulinoThermo(Modulino):
    """Class to handle a Modulino Thermo (temperature and humidity sensor, I2C address 0x44)."""

    _TYPE = "thermo"
    I2C_ADDRESS = 0x44

    def __init__(self, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoThermo.

        Args:
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If `hub_port` is not a ModulinoHubPort.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(self.I2C_ADDRESS, hub_port)

    def get_humidity(self) -> float:
        """Get the humidity.

        Returns:
            float: The relative humidity in percent.

        Raises:
            ModulinoError: If the board can't read the humidity.
        """
        return self._call("get humidity", "getHumidity")

    def get_temperature(self) -> float:
        """Get the temperature.

        Returns:
            float: The temperature in degrees Celsius.

        Raises:
            ModulinoError: If the board can't read the temperature.
        """
        return self._call("get temperature", "getTemperature")


class MovementSample(NamedTuple):
    """A movement sample of a Modulino Movement.

    It is a tuple `(x, y, z, roll, pitch, yaw)`, so it can be unpacked (`x, y, z, roll, pitch, yaw = sample`)
    or read by field (`sample.x`).
    """

    x: float
    """Acceleration on the X axis, in g."""
    y: float
    """Acceleration on the Y axis, in g."""
    z: float
    """Acceleration on the Z axis, in g."""
    roll: float
    """Angular rate around the X axis, in degrees per second."""
    pitch: float
    """Angular rate around the Y axis, in degrees per second."""
    yaw: float
    """Angular rate around the Z axis, in degrees per second."""


class ModulinoMovement(Modulino):
    """Class to handle a Modulino Movement (accelerometer and gyroscope)."""

    _TYPE = "movement"

    def __init__(self, i2c_address: int = 0x6A, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoMovement with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the movement modulino. Defaults to 0x6A.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def get_sample(self) -> MovementSample:
        """Get a movement sample.

        Returns:
            MovementSample: The movement sample containing x, y, z, roll, pitch, and yaw values.

        Raises:
            ModulinoError: If the board can't read the sample.
        """
        what = "get movement sample"
        # Data are cached in the MCU: the "update" call refreshes the cache read by the getters below.
        self._call(what, "update")
        return MovementSample(
            x=self._call(what, "getX"),
            y=self._call(what, "getY"),
            z=self._call(what, "getZ"),
            roll=self._call(what, "getRoll"),
            pitch=self._call(what, "getPitch"),
            yaw=self._call(what, "getYaw"),
        )

    def start_sample(self, ms: int, callback: Callable[[MovementSample], object]) -> None:
        """Start sampling movement data at the specified interval.

        The callback function will be called with each movement sample as an argument
        at the specified interval.
        To stop sampling, call the `stop_sample` method.
        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            ms (int): The sampling interval in milliseconds (minimum 100 ms).
            callback (Callable[[MovementSample], object]): Called with each sample:
                  def callback(sample: MovementSample) -> None:

        Raises:
            TypeError: If `ms` is not an integer or `callback` is not callable.
            ValueError: If `ms` is less than 100.
            ModulinoError: If the board can't start the sampling.
        """
        _check_int("ms", ms, 100)
        _check_callable(callback)

        # The MCU notifies the 6 values of the sample
        def handler(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
            callback(MovementSample(x, y, z, roll, pitch, yaw))

        self._subscribe("start sampling", handler, ms)

    def stop_sample(self) -> None:
        """Stop sampling movement data.

        Raises:
            ModulinoError: If the board can't stop the sampling.
        """
        self._unsubscribe("stop sampling")


class ModulinoButtons(Modulino):
    """Class to handle a Modulino Buttons."""

    _TYPE = "buttons"

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoButtons with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the buttons modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def set_leds(self, led_a: bool, led_b: bool, led_c: bool) -> None:
        """Set the state of the LEDs on the ModulinoButtons.

        Args:
            led_a (bool): State of LED A.
            led_b (bool): State of LED B.
            led_c (bool): State of LED C.

        Raises:
            ModulinoError: If the board can't set the LEDs.
        """
        self._call("set LEDs", "setLeds", led_a, led_b, led_c)

    def set_on_change_callback(self, callback: Callable[[int, int, int], object]) -> None:
        """Set the callback function to be called when a button state changes.

        The callback will report the current state of each button after the change event (1 pressed, 0 released).
        The callback must accept 3 arguments: button_a, button_b, and button_c.

        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[int, int, int], object]): The callback function to be called on button state change.
                The callback function should have the following signature:
                def callback(button_a: int, button_b: int, button_c: int) -> None:

        Raises:
            TypeError: If `callback` is not callable.
            ModulinoError: If the board can't start the notifications.
        """
        _check_callable(callback)
        self._subscribe("set callback", callback, 10)

    def remove_on_change_callback(self) -> None:
        """Remove the callback function for button state changes.

        Raises:
            ModulinoError: If the board can't stop the notifications.
        """
        self._unsubscribe("remove callback")

    def set_debounce(self, ms: int) -> None:
        """Set the debounce time for button state changes.

        Args:
            ms (int): Debounce time in milliseconds. Defaults to 30 after `start()`.

        Raises:
            TypeError: If `ms` is not an integer.
            ValueError: If `ms` is negative.
            ModulinoError: If the board can't set the debounce time.
        """
        _check_int("ms", ms, 0)
        self._call("set debounce", "setDebounce", ms)


class ModulinoJoystick(Modulino):
    """Class to handle a Modulino Joystick."""

    _TYPE = "joystick"

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoJoystick with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the joystick modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def set_on_change_callback(self, callback: Callable[[int, int, int], object]) -> None:
        """Set the callback function for joystick state changes.

        The callback function will be called whenever the joystick state changes
        on x, y axes or the button state.
        The callback function should have the following signature:
                def callback(x: int, y: int, button: int) -> None:

        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[int, int, int], object]): The callback function to be called when the joystick state changes.

        Raises:
            TypeError: If `callback` is not callable.
            ModulinoError: If the board can't start the notifications.
        """
        _check_callable(callback)
        self._subscribe("set callback", callback, 10)

    def remove_on_change_callback(self) -> None:
        """Remove the callback function for joystick state changes.

        Raises:
            ModulinoError: If the board can't stop the notifications.
        """
        self._unsubscribe("remove callback")

    def set_debounce(self, ms: int) -> None:
        """Set the debounce time for the joystick modulino.

        Args:
            ms (int): Debounce time in milliseconds. Defaults to 30 after `start()`.

        Raises:
            TypeError: If `ms` is not an integer.
            ValueError: If `ms` is negative.
            ModulinoError: If the board can't set the debounce time.
        """
        _check_int("ms", ms, 0)
        self._call("set debounce", "setDebounce", ms)


class ModulinoKnob(Modulino):
    """Class to handle a Modulino Knob."""

    _TYPE = "knob"

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoKnob with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the knob modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def set_on_change_callback(self, callback: Callable[[int, int, int], object]) -> None:
        """Set the callback function for knob state changes.

        The callback function will be called on every change of the knob state,
        on the increment or decrement of the knob value or the button press.

        The callback function should accept three integer arguments: the position of the knob, the button state
        (1 pressed, 0 released) and the direction, how much the position changed since the previous notification
        (its sign gives the direction of the rotation, 0 when only the button changed).

        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[int, int, int], object]): The callback function to be called when the knob state changes.

        Raises:
            TypeError: If `callback` is not callable.
            ModulinoError: If the board can't start the notifications.
        """
        _check_callable(callback)
        self._subscribe("set callback", callback, 10)

    def remove_on_change_callback(self) -> None:
        """Remove the callback function for knob state changes.

        Raises:
            ModulinoError: If the board can't stop the notifications.
        """
        self._unsubscribe("remove callback")

    def set_debounce(self, ms: int) -> None:
        """Set the debounce time for the knob modulino.

        Args:
            ms (int): Debounce time in milliseconds. Defaults to 30 after `start()`.

        Raises:
            TypeError: If `ms` is not an integer.
            ValueError: If `ms` is negative.
            ModulinoError: If the board can't set the debounce time.
        """
        _check_int("ms", ms, 0)
        self._call("set debounce", "setDebounce", ms)

    def set_start_position(self, position: int) -> None:
        """Set the start position for the knob modulino.

        Args:
            position (int): The start position value, -32768 to 32767.

        Raises:
            TypeError: If `position` is not an integer.
            ValueError: If `position` is out of range.
            ModulinoError: If the board can't set the position.
        """
        _check_int("position", position, -32768, 32767)
        self._call("set start position", "setStartPosition", position)


class ModulinoDistance(Modulino):
    """Class to handle a Modulino Distance (time-of-flight sensor, I2C address 0x29).

    `start()` waits up to 50 ms for the first measure, so the first `get_distance()` is already valid.
    """

    _TYPE = "distance"
    I2C_ADDRESS = 0x29

    FAR = 0
    """Proximity state: nothing closer than the threshold."""
    NEAR = 1
    """Proximity state: an object is closer than the threshold."""

    def __init__(self, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoDistance.

        Args:
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If `hub_port` is not a ModulinoHubPort.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(self.I2C_ADDRESS, hub_port)

    def get_distance(self) -> float:
        """Get the last measured distance.

        Returns:
            float: The distance in millimeters, NaN if the measure is not valid (e.g. nothing in range).

        Raises:
            ModulinoError: If the board can't read the distance.
        """
        return self._call("get distance", "getDistance")

    def set_threshold_callback(self, threshold_mm: int, callback: Callable[[int, int], object], hysteresis_mm: int = 20) -> None:
        """Call a callback when an object enters or leaves the zone closer than a threshold.

        The state becomes `NEAR` when the distance goes below `threshold_mm` and goes back
        to `FAR` only when it rises above `threshold_mm + hysteresis_mm`, so an object standing
        on the threshold does not flood the callback. The callback is called only on state changes.
        A new state must be confirmed by the following measures: `NEAR` only if they are still below
        `threshold_mm`, `FAR` only if they are still above `threshold_mm + hysteresis_mm`.
        An invalid measure (e.g. nothing in range) counts as `FAR`.
        Calling this again restarts from `FAR`, so an object already near is notified as `NEAR` again.

        This runs through the Linux side and is not a safety-rated emergency stop.

        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            threshold_mm (int): The threshold distance in millimeters.
            callback (Callable[[int, int], object]): Called with the new state (`NEAR` or `FAR`)
                and the distance in millimeters (-1 if not valid):
                def callback(state: int, distance_mm: int) -> None:
            hysteresis_mm (int, optional): The hysteresis in millimeters. Defaults to 20.

        Raises:
            TypeError: If `threshold_mm` or `hysteresis_mm` is not an integer, or `callback` is not callable.
            ValueError: If `threshold_mm` is not positive, `hysteresis_mm` is negative or their sum is above 32767.
            ModulinoError: If the board can't set the threshold or start the notifications.
        """
        _check_int("threshold_mm", threshold_mm, 1)
        _check_int("hysteresis_mm", hysteresis_mm, 0)
        if threshold_mm + hysteresis_mm > 32767:
            raise ValueError(f"threshold_mm + hysteresis_mm must be at most 32767, got {threshold_mm + hysteresis_mm}")
        _check_callable(callback)
        self._command("set threshold", "setThreshold", threshold_mm, hysteresis_mm)

        # The MCU notifies 3 integers, the third is unused
        def handler(state: int, distance_mm: int, _unused: int) -> None:
            callback(state, distance_mm)

        self._subscribe("set callback", handler, 20)  # the sensor measures every 20 ms

    def remove_threshold_callback(self) -> None:
        """Remove the threshold callback.

        Raises:
            ModulinoError: If the board can't stop the notifications.
        """
        self._unsubscribe("remove callback")

    def set_debounce(self, ms: int) -> None:
        """Set how long a new state must be stable before the callback is called.

        Defaults to 0: a state change is confirmed by the next measure (about 20 ms).

        Args:
            ms (int): Debounce time in milliseconds.

        Raises:
            TypeError: If `ms` is not an integer.
            ValueError: If `ms` is negative.
            ModulinoError: If the board can't set the debounce time.
        """
        _check_int("ms", ms, 0)
        self._call("set debounce", "setDebounce", ms)


class ModulinoBuzzer(Modulino):
    """Class to handle a Modulino Buzzer."""

    _TYPE = "buzzer"

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoBuzzer with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the buzzer modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def play_tone(self, frequency: int, duration_ms: int) -> None:
        """Play a tone on the buzzer.

        Args:
            frequency (int): The tone frequency in Hz.
            duration_ms (int): The tone duration in milliseconds.

        Raises:
            TypeError: If `frequency` or `duration_ms` is not an integer.
            ValueError: If `frequency` or `duration_ms` is negative.
            ModulinoError: If the board can't play the tone.
        """
        _check_int("frequency", frequency, 0)
        _check_int("duration_ms", duration_ms, 0)
        self._command("play tone", "tone", frequency, duration_ms)

    def stop_tone(self) -> None:
        """Stop the tone currently playing on the buzzer.

        Raises:
            ModulinoError: If the board can't stop the tone.
        """
        self._call("stop tone", "noTone")


class ModulinoPixels(Modulino):
    """Class to handle a Modulino Pixels (8 RGB LEDs)."""

    _TYPE = "pixels"
    LED_COUNT = 8
    """Number of LEDs, with index 0 to 7."""

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoPixels with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the pixels modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def set_pixel(self, idx: int, r: int, g: int, b: int, brightness: int = 25) -> None:
        """Set the color and brightness of one LED.

        Args:
            idx (int): The LED index, 0-7.
            r (int): Red, 0-255.
            g (int): Green, 0-255.
            b (int): Blue, 0-255.
            brightness (int, optional): Brightness in percent, 0-100. Defaults to 25.

        Raises:
            TypeError: If an argument is not an integer.
            ValueError: If an argument is out of range.
            ModulinoError: If the board can't set the LED.
        """
        _check_int("idx", idx, 0, self.LED_COUNT - 1)
        for name, value in (("r", r), ("g", g), ("b", b)):
            _check_int(name, value, 0, 255)
        _check_int("brightness", brightness, 0, 100)
        self._command(f"set LED {idx}", "set", idx, r, g, b, brightness)

    def set_color(self, idx: int, color: tuple[int, int, int], brightness: int = 25) -> None:
        """Set the color and brightness of one LED.

        Args:
            idx (int): The LED index, 0-7.
            color (tuple[int, int, int]): The color, a `RGBColor` or any `(r, g, b)` tuple with 0-255 per channel.
            brightness (int, optional): Brightness in percent, 0-100. Defaults to 25.

        Raises:
            TypeError: If `color` is not an `(r, g, b)` tuple, or another argument is not an integer.
            ValueError: If an argument is out of range.
            ModulinoError: If the board can't set the LED.
        """
        try:
            r, g, b = color
        except (TypeError, ValueError) as e:
            raise TypeError("color must be an (r, g, b) tuple") from e
        self.set_pixel(idx, r, g, b, brightness)

    def clear_pixel(self, idx: int) -> None:
        """Turn off one LED.

        Args:
            idx (int): The LED index, 0-7.

        Raises:
            TypeError: If `idx` is not an integer.
            ValueError: If `idx` is out of range.
            ModulinoError: If the board can't turn the LED off.
        """
        _check_int("idx", idx, 0, self.LED_COUNT - 1)
        self._command(f"clear LED {idx}", "clear", idx)

    def clear_all_pixels(self) -> None:
        """Turn off all the LEDs.

        Raises:
            ModulinoError: If the board can't turn the LEDs off.
        """
        self._call("clear LEDs", "clearAll")


class ModulinoLight(Modulino):
    """Class to handle a Modulino Light (color and ambient light sensor, I2C address 0x53).

    Every getter takes a new measure of all the sensors, so it can take up to 200 ms.
    """

    _TYPE = "light"
    I2C_ADDRESS = 0x53

    def __init__(self, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoLight.

        Args:
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If `hub_port` is not a ModulinoHubPort.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(self.I2C_ADDRESS, hub_port)

    def get_lux(self) -> int:
        """Get the illuminance.

        Returns:
            int: The illuminance in lux.

        Raises:
            ModulinoError: If the measure fails.
        """
        return self._read("get lux", "getLux")

    def get_ambient_light(self) -> int:
        """Get the raw ambient light.

        Returns:
            int: The raw ambient light measure of the sensor (not converted to lux).

        Raises:
            ModulinoError: If the measure fails.
        """
        return self._read("get ambient light", "getAmbientLight")

    def get_ir(self) -> int:
        """Get the raw infrared light.

        Returns:
            int: The raw infrared measure of the sensor.

        Raises:
            ModulinoError: If the measure fails.
        """
        return self._read("get infrared light", "getIR")

    def get_color(self) -> RGBColor:
        """Get the color seen by the sensor.

        Returns:
            RGBColor: The color, 0-255 per channel.

        Raises:
            ModulinoError: If the measure fails.
        """
        # The MCU packs the color as 0x00RRGGBB
        packed: int = self._read("get color", "getColor")
        return RGBColor((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)


class ModulinoLatchRelay(Modulino):
    """Class to handle a Modulino Latch Relay.

    The relay keeps its position (on or off) when the power is removed, and also when the modulino is stopped.
    """

    _TYPE = "latch_relay"

    UNKNOWN = -1
    """Relay status: unknown, until the first `set()` or `reset()` after the power on."""
    OFF = 0
    """Relay status: off (open circuit)."""
    ON = 1
    """Relay status: on (closed circuit)."""

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoLatchRelay with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the latch relay modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def set(self) -> None:
        """Turn the relay on (close the circuit).

        Raises:
            ModulinoError: If the board can't switch the relay.
        """
        self._call("set the relay", "set")

    def reset(self) -> None:
        """Turn the relay off (open the circuit).

        Raises:
            ModulinoError: If the board can't switch the relay.
        """
        self._call("reset the relay", "reset")

    def get_status(self) -> int:
        """Get the relay status.

        Returns:
            int: `ON`, `OFF` or `UNKNOWN` (until the first `set()` or `reset()` after the power on).

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return self._call("get status", "getStatus")


class ModulinoVibro(Modulino):
    """Class to handle a Modulino Vibro (vibration motor)."""

    _TYPE = "vibro"

    STOP = 0
    """Power level: no vibration."""
    GENTLE = 25
    """Power level: soft vibration."""
    MODERATE = 30
    """Power level: moderate vibration."""
    MEDIUM = 35
    """Power level: medium vibration."""
    INTENSE = 40
    """Power level: intense vibration."""
    POWERFUL = 45
    """Power level: powerful vibration."""
    MAXIMUM = 50
    """Power level: strongest vibration."""

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None) -> None:
        """Initialize the ModulinoVibro with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the vibro modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        super().__init__(i2c_address, hub_port)

    def on(self, duration_ms: int, power: int = MAXIMUM, block: bool = False) -> None:
        """Vibrate for the given time.

        The modulino stops by itself after `duration_ms`. By default the call returns right away.

        Args:
            duration_ms (int): The vibration duration in milliseconds.
            power (int, optional): The power level, `STOP` (0) to `MAXIMUM` (50). Any value in between
                is allowed, the named levels are `GENTLE`, `MODERATE`, `MEDIUM`, `INTENSE` and `POWERFUL`. Defaults to `MAXIMUM`.
            block (bool, optional): If True, wait `duration_ms` and then call `off()`. Defaults to False.

        Raises:
            TypeError: If `duration_ms` or `power` is not an integer.
            ValueError: If `duration_ms` is negative or `power` is out of range.
            ModulinoError: If the board can't start the vibration.
        """
        _check_int("duration_ms", duration_ms, 0)
        _check_int("power", power, self.STOP, self.MAXIMUM)
        self._command("turn on", "on", duration_ms, power)
        if block:
            time.sleep(duration_ms / 1000)
            self.off()

    def off(self) -> None:
        """Stop the vibration.

        Raises:
            ModulinoError: If the board can't stop the vibration.
        """
        self._call("turn off", "off")


class ModulinoMotors(Modulino):
    """Class to handle a Modulino Motors (two DC motors on channels A and B, or one stepper motor).

    The modulino starts in DC mode. The stepper methods (`move`, `hold`, `release`, `set_half_step`
    and the move complete callback) are available after `set_stepper_mode(True)`.
    `stop()` stops both DC motors and, in stepper mode, releases the coils.
    """

    _TYPE = "motors"

    DECAY_SLOW = 0
    """Decay mode: smoother and quieter, slower to follow speed changes."""
    DECAY_MIXED_SLOW = 1
    """Decay mode: mixed, 30% fast and 70% slow."""
    DECAY_MIXED_FAST = 2
    """Decay mode: mixed, 60% fast and 40% slow."""
    DECAY_FAST = 3
    """Decay mode: quick response to changes, more vibration and noise."""

    _CHANNEL_A = 0
    _CHANNEL_B = 1

    def __init__(self, i2c_address: int = 0xFF, hub_port: ModulinoHubPort | None = None, steps_per_revolution: int | None = None) -> None:
        """Initialize the ModulinoMotors with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the motors modulino. Defaults to 0xFF: the address is found by autoscan.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
            steps_per_revolution (int, optional): Full steps per shaft revolution of the stepper motor (1 to 32767).
                Required only by `move()`. Defaults to None.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `i2c_address` or `steps_per_revolution` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        if steps_per_revolution is not None:
            _check_int("steps_per_revolution", steps_per_revolution, 1, 32767)
        super().__init__(i2c_address, hub_port)
        self.steps_per_revolution = steps_per_revolution
        self._stepper = False

    def _require_stepper(self, what: str) -> None:
        if not self._stepper:
            raise RuntimeError(f"modulino {self.name}: cannot {what}: stepper mode not enabled")

    def _require_dc(self, what: str) -> None:
        if self._stepper:
            raise RuntimeError(f"modulino {self.name}: cannot {what}: stepper mode enabled")

    def _status(self) -> int:
        return self._read("get status", "getStatus")

    @override
    def start(self) -> None:
        """Start the motors modulino, in DC mode with both motors stopped.

        Raises:
            ModulinoError: If the Modulino doesn't answer.
        """
        super().start()
        self._stepper = False

    # Common methods

    def set_stepper_mode(self, enabled: bool) -> None:
        """Switch between DC mode (two DC motors) and stepper mode (one stepper motor).

        Enabling the stepper mode stops the DC motors first.

        Args:
            enabled (bool): True for stepper mode, False for DC mode.

        Raises:
            TypeError: If `enabled` is not a boolean.
            ModulinoError: If the board can't switch the mode.
        """
        _check_bool("enabled", enabled)
        self._command("set stepper mode", "setStepperMode", enabled)
        self._stepper = enabled

    def is_stepper_mode(self) -> bool:
        """Return True if the module is in stepper mode (read from the module).

        Returns:
            bool: True in stepper mode, False in DC mode.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return bool(self._status() & 0x02)

    def set_decay(self, mode: int) -> None:
        """Set the current decay mode of the motor driver.

        Args:
            mode (int): `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`.

        Raises:
            TypeError: If `mode` is not an integer.
            ValueError: If `mode` is not a decay mode.
            ModulinoError: If the board can't set the decay mode.
        """
        _check_int("mode", mode, self.DECAY_SLOW, self.DECAY_FAST)
        self._command("set decay", "setDecay", mode)

    def get_decay(self) -> int:
        """Return the decay mode (read from the module).

        Returns:
            int: `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return (self._status() >> 4) & 0x03

    def set_half_full_scale(self, enabled: bool) -> None:
        """Set the current range of the motor driver.

        Args:
            enabled (bool): True for up to 1.9 A with better accuracy at low currents,
                False for up to 3.8 A (default).

        Raises:
            TypeError: If `enabled` is not a boolean.
            ModulinoError: If the board can't set the current range.
        """
        _check_bool("enabled", enabled)
        self._command("set half full scale", "setHalfFullScale", enabled)

    def is_half_full_scale(self) -> bool:
        """Return True if the half full scale current range is enabled (read from the module).

        Returns:
            bool: True for the 1.9 A range, False for the 3.8 A range.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return bool(self._status() & 0x08)

    def get_current_a(self) -> float:
        """Return the current of channel A (in stepper mode, the current of coil A).

        Returns:
            float: The current in milliamps.

        Raises:
            ModulinoError: If the board can't read the current.
        """
        return self._read("get current", "getCurrent", self._CHANNEL_A)

    def get_current_b(self) -> float:
        """Return the current of channel B (in stepper mode, the current of coil B).

        Returns:
            float: The current in milliamps.

        Raises:
            ModulinoError: If the board can't read the current.
        """
        return self._read("get current", "getCurrent", self._CHANNEL_B)

    # DC mode

    def _set_speed(self, channel: int, percent: int) -> None:
        _check_int("percent", percent, -100, 100)
        self._require_dc("set speed")
        self._command("set speed", "setSpeed", channel, percent)

    def set_speed_a(self, percent: int) -> None:
        """Set the speed of the DC motor on channel A, leaving channel B unchanged.

        Args:
            percent (int): The speed from -100 to 100 percent, the sign gives the direction. 0 stops the motor.

        Raises:
            TypeError: If `percent` is not an integer.
            ValueError: If `percent` is out of range.
            RuntimeError: In stepper mode.
            ModulinoError: If the board can't set the speed.
        """
        self._set_speed(self._CHANNEL_A, percent)

    def set_speed_b(self, percent: int) -> None:
        """Set the speed of the DC motor on channel B, leaving channel A unchanged.

        Args:
            percent (int): The speed from -100 to 100 percent, the sign gives the direction. 0 stops the motor.

        Raises:
            TypeError: If `percent` is not an integer.
            ValueError: If `percent` is out of range.
            RuntimeError: In stepper mode.
            ModulinoError: If the board can't set the speed.
        """
        self._set_speed(self._CHANNEL_B, percent)

    def stop_motors(self) -> None:
        """Stop both DC motors.

        Raises:
            RuntimeError: In stepper mode.
            ModulinoError: If the board can't stop the motors.
        """
        self._require_dc("stop motors")
        self._command("stop motors", "stop")

    def set_frequency(self, hz: int) -> None:
        """Set the PWM frequency of the DC motors.

        Args:
            hz (int): The frequency from 200 to 60000 Hz. Defaults to 20000 Hz, above the audible range.

        Raises:
            TypeError: If `hz` is not an integer.
            ValueError: If `hz` is out of range.
            ModulinoError: If the board can't set the frequency.
        """
        _check_int("hz", hz, 200, 60000)
        self._command("set frequency", "setFrequency", hz)

    def get_frequency(self) -> int:
        """Return the PWM frequency of the DC motors (the last value set, the module does not report it).

        Returns:
            int: The frequency in Hz.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return (self._status() >> 8) & 0xFFFF

    # Stepper mode

    def move(self, steps: int, rpm: float, release_ms: int = 0, block: bool = False) -> None:
        """Move the stepper motor by a number of steps.

        Requires the stepper mode and `steps_per_revolution`. In half step mode a step is a half step.

        Args:
            steps (int): The number of steps, negative to move backwards.
            rpm (float): The shaft speed in revolutions per minute.
            release_ms (int, optional): 0 keeps the motor holding its position at the end of the move,
                1 to 255 releases the coils that many milliseconds after the end of the move. Defaults to 0.
            block (bool, optional): If True, wait until the move is complete. Defaults to False.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `rpm` or `release_ms` is out of range.
            RuntimeError: If the stepper mode is not enabled or `steps_per_revolution` is not set.
            ModulinoError: If the board can't move the motor.
        """
        _check_int("steps", steps)
        _check_number("rpm", rpm)
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        _check_int("release_ms", release_ms, 0, 255)
        if self.steps_per_revolution is None:
            raise RuntimeError(f"modulino {self.name}: cannot move: steps_per_revolution not set")
        self._require_stepper("move")
        steps_per_revolution = self.steps_per_revolution * (2 if self.is_half_step() else 1)
        period = int(600000 / (rpm * steps_per_revolution))  # 0.1 ms ticks between steps
        if not 1 <= period <= 65535:
            raise ValueError(f"rpm out of range for {steps_per_revolution} steps per revolution")
        self._command("move", "move", steps, period, release_ms)
        if block:
            while self.is_busy():
                time.sleep(0.01)

    def is_busy(self) -> bool:
        """Return True while the stepper motor is moving (read from the module).

        Returns:
            bool: True while a move is running.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return bool(self._status() & 0x01)

    def hold(self) -> None:
        """Energize the stepper coils, so the motor holds its position.

        Raises:
            RuntimeError: If the stepper mode is not enabled.
            ModulinoError: If the board can't energize the coils.
        """
        self._require_stepper("hold")
        self._command("hold", "hold")

    def release(self) -> None:
        """Release the stepper coils, so the motor turns freely.

        Raises:
            RuntimeError: If the stepper mode is not enabled.
            ModulinoError: If the board can't release the coils.
        """
        self._require_stepper("release")
        self._command("release", "release")

    def set_half_step(self, enabled: bool) -> None:
        """Set half step mode: twice the steps per revolution, smoother movement.

        Args:
            enabled (bool): True for half step, False for full step.

        Raises:
            TypeError: If `enabled` is not a boolean.
            RuntimeError: If the stepper mode is not enabled.
            ModulinoError: If the board can't set the half step mode.
        """
        _check_bool("enabled", enabled)
        self._require_stepper("set half step")
        self._command("set half step", "setHalfStep", enabled)

    def is_half_step(self) -> bool:
        """Return True if half step mode is enabled (read from the module).

        Returns:
            bool: True in half step mode, False in full step mode.

        Raises:
            ModulinoError: If the board can't read the status.
        """
        return bool(self._status() & 0x04)

    def set_on_move_complete_callback(self, callback: Callable[[], object]) -> None:
        """Call a callback each time a stepper move completes.

        Set it before calling `move()`. If a new move starts before the previous one completes,
        the callback is called once, at the end of the last move.

        The callback runs on the bridge thread: it must not call the methods of the Modulini,
        signal the app instead (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[], object]): Called with no arguments:
                def callback() -> None:

        Raises:
            TypeError: If `callback` is not callable.
            RuntimeError: If the stepper mode is not enabled.
            ModulinoError: If the board can't start the notifications.
        """
        self._require_stepper("set move complete callback")
        _check_callable(callback)

        # The MCU notifies 3 integers: busy (always 0) and two unused values
        def handler(_busy: int, _unused1: int, _unused2: int) -> None:
            callback()

        self._subscribe("set callback", handler, 20)

    def remove_on_move_complete_callback(self) -> None:
        """Remove the move complete callback.

        Raises:
            ModulinoError: If the board can't stop the notifications.
        """
        self._unsubscribe("remove callback")


class ModulinoLEDMatrix(Modulino):
    """Class to handle a Modulino LED Matrix (12x8 LEDs).

    Frames are `Frame` objects of 12x8 pixels (`MODULINO_MATRIX`), create them with `designer()`.
    The matrix is monochrome by default, `set_grayscale(True)` enables 16 brightness levels.
    The LED matrix is not supported on a Modulino Hub: connect it to the main bus.
    """

    _TYPE = "led_matrix"

    MAX_FRAMES = 20
    """Maximum number of frames of an animation (MODULINO_LED_MATRIX_MAX_FRAMES of the sketch library)."""

    _MODE_MONOCHROME = 0
    _MODE_GRAYSCALE = 1
    _GRAYSCALE_LEVELS = 16

    def __init__(self, i2c_address: int = 0x39) -> None:
        """Initialize the ModulinoLEDMatrix with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the LED matrix, 0x08 to 0x77. Defaults to 0x39.

        Raises:
            TypeError: If `i2c_address` is not an integer.
            ValueError: If `i2c_address` is out of range.
            ModulinoError: If the board can't create the Modulino.
        """
        _check_int("i2c_address", i2c_address)
        if not 0x08 <= i2c_address <= 0x77:
            raise ValueError(f"i2c_address must be between 0x08 and 0x77, got 0x{i2c_address:x}")
        super().__init__(i2c_address)
        self._grayscale = False
        self._on_animation_end: Callable[[], object] | None = None

    @override
    def start(self) -> None:
        """Start the LED matrix, in monochrome mode.

        Raises:
            ModulinoError: If the Modulino doesn't answer.
        """
        super().start()
        self._grayscale = False

        # Always provided: the MCU notifies (3 unused integers) the end of every animation without loop
        def handler(_unused0: int, _unused1: int, _unused2: int) -> None:
            self._animation_ended()

        try:
            Bridge.provide(f"{self.name}_callback", handler)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot start") from e

    def _animation_ended(self) -> None:
        callback = self._on_animation_end
        if callback is not None:
            callback()

    @property
    def brightness_levels(self) -> int:
        """Number of brightness levels shown by the matrix: 2 in monochrome mode, 16 in grayscale mode."""
        return self._GRAYSCALE_LEVELS if self._grayscale else 2

    def designer(self) -> FrameDesigner:
        """Return a FrameDesigner for this matrix: 12x8 frames with the brightness levels of the current mode.

        Returns:
            FrameDesigner: The designer, e.g. `matrix.designer().empty()` for a blank frame.
        """
        return FrameDesigner(MODULINO_MATRIX, self.brightness_levels)

    def set_grayscale(self, enabled: bool) -> None:
        """Set the display mode.

        Args:
            enabled (bool): True for grayscale (16 brightness levels), False for monochrome (on/off).

        Raises:
            TypeError: If `enabled` is not a boolean.
            ModulinoError: If the board can't set the mode.
        """
        _check_bool("enabled", enabled)
        mode = self._MODE_GRAYSCALE if enabled else self._MODE_MONOCHROME
        self._command("set the mode", "setMode", mode)
        self._grayscale = enabled

    def is_grayscale(self) -> bool:
        """Return True if the matrix is in grayscale mode.

        Returns:
            bool: True in grayscale mode, False in monochrome mode.
        """
        return self._grayscale

    def _check_frame(self, frame: object) -> None:
        if not isinstance(frame, Frame):
            raise TypeError(f"frame must be a Frame, got {type(frame).__name__}")
        if (frame.width, frame.height) != MODULINO_MATRIX:
            raise ValueError(f"frame must be {MODULINO_MATRIX.width}x{MODULINO_MATRIX.height}, got {frame.width}x{frame.height}")

    def draw(self, frame: Frame) -> None:
        """Show a frame on the matrix. A running animation is stopped.

        In monochrome mode the pixels with a value greater than 0 are on. In grayscale mode the values are
        rescaled from the frame brightness levels to the 16 levels of the matrix.

        Args:
            frame (Frame): A 12x8 frame, with any number of brightness levels.

        Raises:
            TypeError: If `frame` is not a Frame.
            ValueError: If `frame` is not 12x8.
            ModulinoError: If the board can't draw the frame.
        """
        self._check_frame(frame)
        pixels = frame.to_bytes(self._GRAYSCALE_LEVELS) if self._grayscale else frame.to_bytes()
        self._command("draw", "draw", pixels)

    def clear(self) -> None:
        """Turn off every LED. A running animation is stopped.

        Raises:
            ModulinoError: If the board can't clear the matrix.
        """
        self.draw(Frame.empty(MODULINO_MATRIX))

    def play_animation(self, frames: Sequence[tuple[Frame, int] | tuple[Frame, int, str]], loop: bool = False) -> None:
        """Play an animation on the matrix, the frames are timed by the board.

        The frames are shown on/off (a pixel is on when its value is greater than 0), also in grayscale mode.
        The call returns right away. A new animation replaces the running one, `draw()` and `clear()` stop it.
        Without loop, the callback set with `on_animation_end_callback()` is called when the animation ends.

        Args:
            frames (Sequence[tuple[Frame, int] | tuple[Frame, int, str]]): `(frame, duration_ms)` or `(frame, duration_ms, label)`
                items, as taken by `sequence_to_c()`: 12x8 frames shown for `duration_ms` milliseconds each, up to `MAX_FRAMES`.
            loop (bool, optional): If True, repeat the animation until it is stopped. Defaults to False.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If `frames` is empty, has more than `MAX_FRAMES` items, or has a frame that is not 12x8 or a negative duration.
            ModulinoError: If the board can't play the animation.
        """
        if not frames:
            raise ValueError("frames must not be empty")
        if len(frames) > self.MAX_FRAMES:
            raise ValueError(f"too many frames: {len(frames)}, the maximum is {self.MAX_FRAMES}")
        _check_bool("loop", loop)
        entries: list[list[int]] = []
        for item in frames:
            frame, duration_ms = item[0], item[1]
            self._check_frame(frame)
            _check_int("duration_ms", duration_ms, 0, 0x7FFFFFFF)
            entries.append([*frame.to_words(), duration_ms])  # 3 words + duration
        what = "play animation"
        self._call(what, "stop_animation")  # drops the frames of the previous animation
        try:
            for entry in entries:
                Bridge.notify(f"{self.name}:load_frame", entry)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot {what}") from e
        count = self._call(what, "play_animation", loop)
        if count != len(entries):
            raise ModulinoError(f"modulino {self.name}: cannot {what}: {count} of {len(entries)} frames loaded")
        try:
            # Starts the playback: the MCU shows the first frame right away, then sets the
            # polling period to the duration of each frame and unsubscribes at the end.
            Bridge.notify("poll_subscribe", self.name, 1)
        except Exception as e:
            raise ModulinoError(f"modulino {self.name}: cannot {what}") from e

    def stop_animation(self) -> None:
        """Stop the animation, the current frame stays on the matrix.

        Raises:
            ModulinoError: If the board can't stop the animation.
        """
        self._call("stop animation", "stop_animation")

    def on_animation_end_callback(self, callback: Callable[[], object]) -> None:
        """Call a callback each time an animation without loop ends, after the duration of its last frame.

        An animation stopped by `stop_animation()`, `draw()`, `clear()` or a new `play_animation()` does not call it.
        The callback runs on the bridge thread: it must not call the methods of the Modulini, signal the app instead
        (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[], object]): Called with no arguments:
                def callback() -> None:

        Raises:
            TypeError: If `callback` is not callable.
        """
        _check_callable(callback)
        self._on_animation_end = callback

    def remove_on_animation_end_callback(self) -> None:
        """Remove the animation end callback."""
        self._on_animation_end = None
