import time
from typing import NamedTuple, override

from arduino.app_utils import brick, Logger, Bridge
from arduino.app_utils.ledmatrix import Frame, FrameDesigner, MODULINO_MATRIX

logger = Logger("Modulino")
class ModulinoError(Exception):
    """Raised when a Modulino fails."""

class ModulinoColor(NamedTuple):
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
    def __init__(self, hub_address, port):
        self.hub_address = hub_address
        self.port = port

class Modulino_Hub:
    """Modulino Hub (I2C multiplexer with 8 ports).

    It holds no resources: the hub is created on the MCU together with the first
    Modulino that uses one of its ports, and released with the last one.
    """
    def __init__(self, address = 0x70) -> None:
        """Initialize a ModulinoHub with the given I2C address.

        Args:
            address (int): The I2C address of the hub, set by the A0-A2 jumpers (0x70-0x77). Defaults to 0x70.
        """
        if not 0x70 <= address <= 0x77:
            raise ModulinoError(f"invalid hub address 0x{address:x}, must be 0x70-0x77")
        self.address = address

    def port(self, n) -> ModulinoHubPort:
        """Get a port of the hub.

        Args:
            n (int): The port number, 0-7.

        Returns:
            ModulinoHubPort: The port, to pass as `hub_port` to a Modulino constructor.
        """
        if not 0 <= n < HUB_PORT_COUNT:
            raise ModulinoError(f"invalid hub port {n}, must be 0-{HUB_PORT_COUNT - 1}")
        return ModulinoHubPort(self.address, n)

@brick
class Modulino:
    """Base class for all Modulino modules."""
    def __init__(self, type, i2c_address, hub_port = None):
        """Initialize a Modulino with the given type and I2C address.

        Args:
            type (str): The type of the Modulino.
            i2c_address (int): The I2C address of the Modulino.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        self.type = type
        self.i2caddress = i2c_address
        self.hub_address = hub_port.hub_address if hub_port is not None else 0
        self.port = hub_port.port if hub_port is not None else 0
        # The MCU builds the RPC name and returns "" if the modulino can't be created
        # (e.g. a modulino of the same type and address already exists on the same bus or hub port).
        # The name is also the key the MCU uses in delete_modulino and poll_subscribe/unsubscribe.
        try:
            self.name = Bridge.call("create_modulino", type, i2c_address, self.hub_address, self.port)
        except Exception:
            self.name = ""
        if not self.name:
            raise ModulinoError(f"cannot create modulino {type} at 0x{i2c_address:x}{self._location()}")

    def _location(self) -> str:
        return f" on hub 0x{self.hub_address:x} port {self.port}" if self.hub_address else ""

    def start(self):
        logger.info("Starting Modulino")

    def stop(self):
        logger.info("Stopping Modulino")


class Modulino_Thermo(Modulino):
    """Class to handle a Modulino 'thermo' for temperature and humidity sensing."""
    def __init__(self, i2c_address = 0x44, hub_port = None) -> None:
        """Initialize a Modulino_Thermo with the given I2C address.

        Args:
            i2c_address (int): The I2C address of the Modulino_Thermo. Defaults to 0x44.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("thermo", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the Modulino_Thermo."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")


    @override
    def stop(self) -> None:
        """Stop the Modulino_Thermo."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def getHumidity(self) -> float:
        """Get the humidity from the Modulino_Thermo.

        Returns:
            float: The humidity value.
        """
        try:
            hum_float = Bridge.call(f"{self.name}:getHumidity")
        except:
            raise ModulinoError(f"cannot get humidity from modulino {self.name}")

        return hum_float

    def getTemperature(self) -> float:
        """Get the temperature from the Modulino_Thermo.

        Returns:
            float: The temperature value.
        """
        try:
            temp_float = Bridge.call(f"{self.name}:getTemperature")
        except:
            raise ModulinoError(f"cannot get temperature from modulino {self.name}")

        return temp_float

class MovementSample:
    """Class to contain a movement sample with x, y, z, roll, pitch, and yaw values.
    The values are typically obtained from a Modulino 'movement' sensor.
    Access the values through the attributes x, y, z, roll, pitch, and yaw.
    """
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class Modulino_Movement(Modulino):
    """Class to handle a Modulino 'movement' sensor."""
    def __init__(self, i2c_address = 0x6A, hub_port = None) -> None:
        """Initialize the Modulino_Movement with the specified I2C address.

        Args:
            i2c_address (int): The I2C address of the Modulino_Movement. Defaults to 0x6A.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("movement", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the Modulino_Movement."""
        try:

            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the Modulino_Movement."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def getSample(self) -> MovementSample:
        """Get a movement sample from the Modulino_Movement.

        Returns:
            MovementSample: The movement sample containing x, y, z, roll, pitch, and yaw values.
        """
        try:
            Bridge.call(f"{self.name}:update")
            x = Bridge.call(f"{self.name}:getX")
            y = Bridge.call(f"{self.name}:getY")
            z = Bridge.call(f"{self.name}:getZ")
            roll = Bridge.call(f"{self.name}:getRoll")
            pitch = Bridge.call(f"{self.name}:getPitch")
            yaw = Bridge.call(f"{self.name}:getYaw")
        except:
            raise ModulinoError(f"cannot get movement sample from modulino {self.name}")

        return MovementSample(x,y,z, roll, pitch, yaw)

    def startSample(self, ms, callback) -> None:
        """Start sampling movement data at the specified interval.
        The callback function will be called with each movement sample as an argument
        at the specified interval.
        To stop sampling, call the `stopSample` method.
        The callback must accept 6 arguments: x, y, z, roll, pitch, and yaw.

        Args:
            ms (int): The sampling interval in milliseconds (minimum 100 ms).
            callback (Callable): The callback function to be called with each sample
              The callback function should have the following signature:
                  def callback(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        """
        if ms < 100:
            raise ModulinoError("sample rate too small")
        Bridge.provide(f"{self.name}_callback", callback)
        Bridge.notify("poll_subscribe", self.name, ms)

    def stopSample(self) -> None:
        """Stop sampling movement data."""
        Bridge.notify("poll_unsubscribe", self.name)

class Modulino_Buttons(Modulino):
    """Class to handle a Modulino Buttons."""
    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Buttons with the specified I2C address.

        Args:
            i2c_address (int): The I2C address of the Modulino_Buttons. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("buttons", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the Modulino_Buttons."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the Modulino_Buttons."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def setLeds(self, ledA, ledB, ledC) -> None:
        """Set the state of the LEDs on the Modulino_Buttons.
        Args:
            ledA (bool): State of LED A.
            ledB (bool): State of LED B.
            ledC (bool): State of LED C.
        """
        try:
            Bridge.call(f"{self.name}:setLeds", ledA, ledB, ledC)
        except:
            raise ModulinoError(f"cannot set LEDs for modulino {self.name}")

    def setOnChangeCallback(self, callback) -> None:
        """Set the callback function to be called when a button state changes.
        The callback will report the current state of each button after the change event (1 pressed, 0 released).
        The callback must accept 3 arguments: buttonA, buttonB, and buttonC.

        Args:
            callback (Callable): The callback function to be called on button state change.
            The callback function should have the following signature:
                def callback(buttonA: int, buttonB: int, buttonC: int) -> None:
        """
        try:
            Bridge.provide(f"{self.name}_callback", callback)
            Bridge.notify("poll_subscribe", self.name, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for button state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except:
            raise ModulinoError(f"cannot remove callback for modulino {self.name}")

    def setDebounce(self, ms) -> None:
        """Set the debounce time for button state changes.

        Args:
            ms (int): Debounce time in milliseconds.
        """
        try:
            Bridge.call(f"{self.name}:setDebounce", ms)
        except:
            raise ModulinoError(f"cannot set debounce for modulino {self.name}")

class Modulino_Joystick(Modulino):
    """Class to handle a Modulino Joystick."""
    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Joystick with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the joystick modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("joystick", i2c_address, hub_port)
    @override
    def start(self) -> None:
        """Start the joystick modulino."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the joystick modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def setOnChangeCallback(self, callback) -> None:
        """Set the callback function for joystick state changes.
        The callback function will be called whenever the joystick state changes
        on x, y axes or the button state.
        The callback function should have the following signature:
                def callback(x: int, y: int, button: int) -> None:

        Args:
            callback (Callable[[int, int, int], None]): The callback function to be called when the joystick state changes.
        """
        try:
            Bridge.provide(f"{self.name}_callback", callback)
            Bridge.notify("poll_subscribe", self.name, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for joystick state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except:
            raise ModulinoError(f"cannot remove callback for modulino {self.name}")

    def setDebounce(self, ms) -> None:
        """Set the debounce time for the joystick modulino.

        Args:
            ms (int): Debounce time in milliseconds.
        """
        try:
            Bridge.call(f"{self.name}:setDebounce", ms)
        except:
            raise ModulinoError(f"cannot set debounce for modulino {self.name}")

class Modulino_Knob(Modulino):
    """Class to handle a Modulino Knob."""
    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Knob with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the knob modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("knob", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the knob modulino."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the knob modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def setOnChangeCallback(self, callback) -> None:
        """Set the callback function for knob state changes.
        The callback function will be called on every change of the knob state,
        on the increment or decrement of the knob value or the button press.

        The callback function should accept three integer arguments: the current knob value, the button state,
        and the direction of the knob rotation.

        Args:
            callback (Callable[[int, int, int], None]): The callback function to be called when the knob state changes.
        """
        try:
            Bridge.provide(f"{self.name}_callback", callback)
            Bridge.notify("poll_subscribe", self.name, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for knob state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except:
            raise ModulinoError(f"cannot remove callback for modulino {self.name}")

    def setDebounce(self, ms) -> None:
        """Set the debounce time for the knob modulino.

        Args:
            ms (int): Debounce time in milliseconds.
        """
        try:
            Bridge.call(f"{self.name}:setDebounce", ms)
        except:
            raise ModulinoError(f"cannot set debounce for modulino {self.name}")

    def setStartPosition(self, position) -> None:
        """Set the start position for the knob modulino.

        Args:
            position (int): The start position value.
        """
        try:
            Bridge.call(f"{self.name}:setStartPosition", position)
        except:
            raise ModulinoError(f"cannot set start position for modulino {self.name}")

class Modulino_Distance(Modulino):
    """Class to handle a Modulino Distance (time-of-flight sensor, I2C address 0x29)."""
    FAR = 0
    """Proximity state: nothing closer than the threshold."""
    NEAR = 1
    """Proximity state: an object is closer than the threshold."""

    I2C_ADDRESS = 0x29

    def __init__(self, hub_port = None) -> None:
        """Initialize the Modulino_Distance.

        Args:
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("distance", self.I2C_ADDRESS, hub_port)

    @override
    def start(self) -> None:
        """Start the Modulino_Distance.

        Waits up to 50 ms for the first measure, so the first `getDistance()` is already valid.
        """
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the Modulino_Distance."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def getDistance(self) -> float:
        """Get the last measured distance.

        Returns:
            float: The distance in millimeters, NaN if the measure is not valid (e.g. nothing in range).
        """
        try:
            return Bridge.call(f"{self.name}:getDistance")
        except:
            raise ModulinoError(f"cannot get distance from modulino {self.name}")

    def setThresholdCallback(self, threshold_mm, callback, hysteresis_mm = 20) -> None:
        """Call a callback when an object enters or leaves the zone closer than a threshold.

        The state becomes `NEAR` when the distance goes below `threshold_mm` and goes back
        to `FAR` only when it rises above `threshold_mm + hysteresis_mm`, so an object standing
        on the threshold does not flood the callback. The callback is called only on state changes.
        A new state must be confirmed by the following measures: `NEAR` only if they are still below
        `threshold_mm`, `FAR` only if they are still above `threshold_mm + hysteresis_mm`.
        An invalid measure (e.g. nothing in range) counts as `FAR`.
        Calling this again restarts from `FAR`, so an object already near is notified as `NEAR` again.

        This runs through the Linux side and is not a safety-rated emergency stop.

        Args:
            threshold_mm (int): The threshold distance in millimeters.
            callback (Callable[[int, int], None]): Called with the new state (`NEAR` or `FAR`)
              and the distance in millimeters (-1 if not valid):
                  def callback(state: int, distance_mm: int) -> None:
            hysteresis_mm (int, optional): The hysteresis in millimeters. Defaults to 20.
        """
        if not isinstance(threshold_mm, int) or not isinstance(hysteresis_mm, int) or threshold_mm <= 0 or hysteresis_mm < 0:
            raise ModulinoError("threshold_mm must be a positive integer and hysteresis_mm a non-negative integer")
        try:
            ok = Bridge.call(f"{self.name}:setThreshold", threshold_mm, hysteresis_mm)
        except:
            raise ModulinoError(f"cannot set threshold for modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot set threshold for modulino {self.name}")
        try:
            # The MCU notifies 3 integers, the third is unused
            Bridge.provide(f"{self.name}_callback", lambda state, distance, _unused: callback(state, distance))
            Bridge.notify("poll_subscribe", self.name, 20)   # the sensor measures every 20 ms
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeThresholdCallback(self) -> None:
        """Remove the threshold callback."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except:
            raise ModulinoError(f"cannot remove callback for modulino {self.name}")

    def setDebounce(self, ms) -> None:
        """Set how long a new state must be stable before the callback is called.

        Defaults to 0: a state change is confirmed by the next measure (about 20 ms).

        Args:
            ms (int): Debounce time in milliseconds.
        """
        try:
            Bridge.call(f"{self.name}:setDebounce", ms)
        except:
            raise ModulinoError(f"cannot set debounce for modulino {self.name}")

class Modulino_Buzzer(Modulino):
    """Class to handle a Modulino Buzzer."""
    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Buzzer with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the buzzer modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("buzzer", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the buzzer modulino."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the buzzer modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def tone(self, frequency, duration_ms) -> None:
        """Play a tone on the buzzer.

        Args:
            frequency (int): The tone frequency in Hz.
            duration_ms (int): The tone duration in milliseconds.
        """
        if not isinstance(frequency, int) or not isinstance(duration_ms, int) or frequency < 0 or duration_ms < 0:
            raise ModulinoError("frequency and duration_ms must be non-negative integers")
        try:
            ok = Bridge.call(f"{self.name}:tone", frequency, duration_ms)
        except:
            raise ModulinoError(f"cannot play tone on modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot play tone on modulino {self.name}")

    def noTone(self) -> None:
        """Stop the tone currently playing on the buzzer."""
        try:
            Bridge.call(f"{self.name}:noTone")
        except:
            raise ModulinoError(f"cannot stop tone on modulino {self.name}")

class Modulino_Pixels(Modulino):
    """Class to handle a Modulino Pixels (8 RGB LEDs)."""
    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Pixels with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the pixels modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("pixels", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the pixels modulino."""
        try:
            Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the pixels modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def set(self, idx, r, g, b, brightness = 25) -> None:
        """Set the color and brightness of one LED.

        Args:
            idx (int): The LED index, 0-7.
            r (int): Red, 0-255.
            g (int): Green, 0-255.
            b (int): Blue, 0-255.
            brightness (int, optional): Brightness in percent, 0-100. Defaults to 25.
        """
        try:
            ok = Bridge.call(f"{self.name}:set", idx, r, g, b, brightness)
        except:
            raise ModulinoError(f"cannot set LED {idx} on modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot set LED {idx} on modulino {self.name}: invalid arguments")

    def setColor(self, idx, color, brightness = 25) -> None:
        """Set the color and brightness of one LED.

        Args:
            idx (int): The LED index, 0-7.
            color (ModulinoColor): The color, or any `(r, g, b)` tuple with 0-255 per channel.
            brightness (int, optional): Brightness in percent, 0-100. Defaults to 25.
        """
        try:
            r, g, b = color
        except (TypeError, ValueError):
            raise ModulinoError(f"cannot set LED {idx} on modulino {self.name}: the color must be (r, g, b)")
        self.set(idx, r, g, b, brightness)

    def clear(self, idx) -> None:
        """Turn off one LED.

        Args:
            idx (int): The LED index, 0-7.
        """
        try:
            ok = Bridge.call(f"{self.name}:clear", idx)
        except:
            raise ModulinoError(f"cannot clear LED {idx} on modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot clear LED {idx} on modulino {self.name}: invalid index")

    def clearAll(self) -> None:
        """Turn off all the LEDs."""
        try:
            Bridge.call(f"{self.name}:clearAll")
        except:
            raise ModulinoError(f"cannot clear LEDs on modulino {self.name}")

class Modulino_Light(Modulino):
    """Class to handle a Modulino Light (color and ambient light sensor, I2C address 0x53).

    Every getter takes a new measure of all the sensors, so it can take up to 200 ms.
    """
    I2C_ADDRESS = 0x53

    def __init__(self, hub_port = None) -> None:
        """Initialize the Modulino_Light.

        Args:
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("light", self.I2C_ADDRESS, hub_port)

    @override
    def start(self) -> None:
        """Start the Modulino_Light."""
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the Modulino_Light."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def _read(self, method, what) -> int:
        # The MCU returns -1 when the measure fails
        try:
            value = Bridge.call(f"{self.name}:{method}")
        except:
            raise ModulinoError(f"cannot get {what} from modulino {self.name}")
        if value < 0:
            raise ModulinoError(f"cannot get {what} from modulino {self.name}: measure failed")
        return value

    def getLux(self) -> int:
        """Get the illuminance.

        Returns:
            int: The illuminance in lux.
        """
        return self._read("getLux", "lux")

    def getAmbientLight(self) -> int:
        """Get the raw ambient light.

        Returns:
            int: The raw ambient light measure of the sensor (not converted to lux).
        """
        return self._read("getAmbientLight", "ambient light")

    def getIR(self) -> int:
        """Get the raw infrared light.

        Returns:
            int: The raw infrared measure of the sensor.
        """
        return self._read("getIR", "infrared light")

    def getColor(self) -> ModulinoColor:
        """Get the color seen by the sensor.

        Returns:
            ModulinoColor: The color, 0-255 per channel.
        """
        # The MCU packs the color as 0x00RRGGBB
        packed = self._read("getColor", "color")
        return ModulinoColor((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)

class Modulino_LatchRelay(Modulino):
    """Class to handle a Modulino Latch Relay.

    The relay keeps its position (on or off) when the power is removed.
    """
    UNKNOWN = -1
    """Relay status: unknown, until the first `set()` or `reset()` after the power on."""
    OFF = 0
    """Relay status: off (open circuit)."""
    ON = 1
    """Relay status: on (closed circuit)."""

    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_LatchRelay with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the latch relay modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("latch_relay", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the latch relay modulino."""
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the latch relay modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def set(self) -> None:
        """Turn the relay on (close the circuit)."""
        try:
            Bridge.call(f"{self.name}:set")
        except:
            raise ModulinoError(f"cannot set modulino {self.name}")

    def reset(self) -> None:
        """Turn the relay off (open the circuit)."""
        try:
            Bridge.call(f"{self.name}:reset")
        except:
            raise ModulinoError(f"cannot reset modulino {self.name}")

    def getStatus(self) -> int:
        """Get the relay status.

        Returns:
            int: `ON`, `OFF` or `UNKNOWN` (until the first `set()` or `reset()` after the power on).
        """
        try:
            return Bridge.call(f"{self.name}:getStatus")
        except:
            raise ModulinoError(f"cannot get status from modulino {self.name}")

class Modulino_Vibro(Modulino):
    """Class to handle a Modulino Vibro (vibration motor)."""
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

    def __init__(self, i2c_address = 0xFF, hub_port = None) -> None:
        """Initialize the Modulino_Vibro with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the vibro modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
        """
        super().__init__("vibro", i2c_address, hub_port)

    @override
    def start(self) -> None:
        """Start the vibro modulino."""
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")

    @override
    def stop(self) -> None:
        """Stop the vibro modulino."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    def on(self, duration_ms, power = MAXIMUM, block = False) -> None:
        """Vibrate for the given time.

        The modulino stops by itself after `duration_ms`. By default the call returns right away.

        Args:
            duration_ms (int): The vibration duration in milliseconds.
            power (int, optional): The power level, `STOP` (0) to `MAXIMUM` (50). Any value in between
                is allowed, the named levels are `GENTLE`, `MODERATE`, `MEDIUM`, `INTENSE` and `POWERFUL`. Defaults to `MAXIMUM`.
            block (bool, optional): If True, wait `duration_ms` and then call `off()`. Defaults to False.
        """
        if not isinstance(duration_ms, int) or duration_ms < 0:
            raise ModulinoError("duration_ms must be a non-negative integer")
        if not isinstance(power, int) or not self.STOP <= power <= self.MAXIMUM:
            raise ModulinoError(f"power must be an integer between {self.STOP} and {self.MAXIMUM}")
        try:
            ok = Bridge.call(f"{self.name}:on", duration_ms, power)
        except:
            raise ModulinoError(f"cannot turn on modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot turn on modulino {self.name}")
        if block:
            time.sleep(duration_ms / 1000)
            self.off()

    def off(self) -> None:
        """Stop the vibration."""
        try:
            Bridge.call(f"{self.name}:off")
        except:
            raise ModulinoError(f"cannot turn off modulino {self.name}")

class Modulino_Motors(Modulino):
    """Class to handle a Modulino Motors (two DC motors on channels A and B, or one stepper motor).

    The modulino starts in DC mode. The stepper methods (`move`, `hold`, `release`, `setHalfStep`
    and the move complete callback) are available after `setStepperMode(True)`.
    """
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

    def __init__(self, i2c_address = 0xFF, hub_port = None, steps_per_revolution = None) -> None:
        """Initialize the Modulino_Motors with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the motors modulino. Defaults it will autoscan for the address.
            hub_port (ModulinoHubPort, optional): The hub port the Modulino is connected to. Defaults to None (main bus).
            steps_per_revolution (int, optional): Full steps per shaft revolution of the stepper motor (1 to 32767).
                Required only by `move()`. Defaults to None.
        """
        if steps_per_revolution is not None and (not isinstance(steps_per_revolution, int) or not 1 <= steps_per_revolution <= 32767):
            raise ModulinoError("steps_per_revolution must be an integer between 1 and 32767")
        super().__init__("motors", i2c_address, hub_port)
        self.steps_per_revolution = steps_per_revolution
        self._stepper = False

    def _command(self, what, method, *args) -> None:
        try:
            ok = Bridge.call(f"{self.name}:{method}", *args)
        except Exception:
            raise ModulinoError(f"cannot {what} for modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot {what} for modulino {self.name}")

    def _require_stepper(self, what) -> None:
        if not self._stepper:
            raise ModulinoError(f"cannot {what} for modulino {self.name}: stepper mode not enabled")

    def _require_dc(self, what) -> None:
        if self._stepper:
            raise ModulinoError(f"cannot {what} for modulino {self.name}: stepper mode enabled")

    def _status(self) -> int:
        try:
            status = Bridge.call(f"{self.name}:getStatus")
        except Exception:
            raise ModulinoError(f"cannot get status from modulino {self.name}")
        if status < 0:
            raise ModulinoError(f"cannot get status from modulino {self.name}")
        return status

    @override
    def start(self) -> None:
        """Start the motors modulino, in DC mode with both motors stopped."""
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except Exception:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")
        self._stepper = False

    @override
    def stop(self) -> None:
        """Stop the motors and the motors modulino.

        In DC mode both motors are stopped, in stepper mode the coils are released.
        """
        try:
            Bridge.call(f"{self.name}:release" if self._stepper else f"{self.name}:stop")
        except Exception:
            logger.warning(f"cannot stop the motors of modulino {self.name}")
        try:
            Bridge.call("delete_modulino", self.name)
        except Exception:
            raise ModulinoError(f"cannot delete modulino {self.name}")

    # Common methods

    def setStepperMode(self, enabled) -> None:
        """Switch between DC mode (two DC motors) and stepper mode (one stepper motor).

        Enabling the stepper mode stops the DC motors first.

        Args:
            enabled (bool): True for stepper mode, False for DC mode.
        """
        if not isinstance(enabled, bool):
            raise ModulinoError("enabled must be a boolean")
        self._command("set stepper mode", "setStepperMode", enabled)
        self._stepper = enabled

    def isStepperMode(self) -> bool:
        """Return True if the module is in stepper mode (read from the module)."""
        return bool(self._status() & 0x02)

    def setDecay(self, mode) -> None:
        """Set the current decay mode of the motor driver.

        Args:
            mode (int): `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`.
        """
        if not isinstance(mode, int) or not self.DECAY_SLOW <= mode <= self.DECAY_FAST:
            raise ModulinoError("mode must be DECAY_SLOW, DECAY_MIXED_SLOW, DECAY_MIXED_FAST or DECAY_FAST")
        self._command("set decay", "setDecay", mode)

    def getDecay(self) -> int:
        """Return the decay mode (read from the module): `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`."""
        return (self._status() >> 4) & 0x03

    def setHalfFullScale(self, enabled) -> None:
        """Set the current range of the motor driver.

        Args:
            enabled (bool): True for up to 1.9 A with better accuracy at low currents,
                False for up to 3.8 A (default).
        """
        if not isinstance(enabled, bool):
            raise ModulinoError("enabled must be a boolean")
        self._command("set half full scale", "setHalfFullScale", enabled)

    def isHalfFullScale(self) -> bool:
        """Return True if the half full scale current range is enabled (read from the module)."""
        return bool(self._status() & 0x08)

    def _current(self, channel) -> float:
        try:
            current = Bridge.call(f"{self.name}:getCurrent", channel)
        except Exception:
            raise ModulinoError(f"cannot get current from modulino {self.name}")
        if current < 0:
            raise ModulinoError(f"cannot get current from modulino {self.name}")
        return current

    def getCurrentA(self) -> float:
        """Return the current of channel A in milliamps (in stepper mode, the current of coil A)."""
        return self._current(self._CHANNEL_A)

    def getCurrentB(self) -> float:
        """Return the current of channel B in milliamps (in stepper mode, the current of coil B)."""
        return self._current(self._CHANNEL_B)

    # DC mode

    def _setSpeed(self, channel, percent) -> None:
        if not isinstance(percent, int) or not -100 <= percent <= 100:
            raise ModulinoError("percent must be an integer between -100 and 100")
        self._require_dc("set speed")
        self._command("set speed", "setSpeed", channel, percent)

    def setSpeedA(self, percent) -> None:
        """Set the speed of the DC motor on channel A, leaving channel B unchanged.

        Args:
            percent (int): The speed from -100 to 100 percent, the sign gives the direction. 0 stops the motor.
        """
        self._setSpeed(self._CHANNEL_A, percent)

    def setSpeedB(self, percent) -> None:
        """Set the speed of the DC motor on channel B, leaving channel A unchanged.

        Args:
            percent (int): The speed from -100 to 100 percent, the sign gives the direction. 0 stops the motor.
        """
        self._setSpeed(self._CHANNEL_B, percent)

    def stopMotors(self) -> None:
        """Stop both DC motors."""
        self._require_dc("stop motors")
        self._command("stop motors", "stop")

    def setFrequency(self, hz) -> None:
        """Set the PWM frequency of the DC motors.

        Args:
            hz (int): The frequency from 200 to 60000 Hz. Defaults to 20000 Hz, above the audible range.
        """
        if not isinstance(hz, int) or not 200 <= hz <= 60000:
            raise ModulinoError("hz must be an integer between 200 and 60000")
        self._command("set frequency", "setFrequency", hz)

    def getFrequency(self) -> int:
        """Return the PWM frequency of the DC motors in Hz (the last value set, the module does not report it)."""
        return (self._status() >> 8) & 0xFFFF

    # Stepper mode

    def move(self, steps, rpm, release_ms = 0, block = False) -> None:
        """Move the stepper motor by a number of steps.

        Requires the stepper mode and `steps_per_revolution`. In half step mode a step is a half step.

        Args:
            steps (int): The number of steps, negative to move backwards.
            rpm (float): The shaft speed in revolutions per minute.
            release_ms (int, optional): 0 keeps the motor holding its position at the end of the move,
                1 to 255 releases the coils that many milliseconds after the end of the move. Defaults to 0.
            block (bool, optional): If True, wait until the move is complete. Defaults to False.
        """
        if not isinstance(steps, int):
            raise ModulinoError("steps must be an integer")
        if not isinstance(rpm, (int, float)) or rpm <= 0:
            raise ModulinoError("rpm must be a positive number")
        if not isinstance(release_ms, int) or not 0 <= release_ms <= 255:
            raise ModulinoError("release_ms must be an integer between 0 and 255")
        if self.steps_per_revolution is None:
            raise ModulinoError(f"cannot move modulino {self.name}: steps_per_revolution not set")
        self._require_stepper("move")
        steps_per_revolution = self.steps_per_revolution * (2 if self.isHalfStep() else 1)
        period = int(600000 / (rpm * steps_per_revolution))   # 0.1 ms ticks between steps
        if not 1 <= period <= 65535:
            raise ModulinoError(f"rpm out of range for {steps_per_revolution} steps per revolution")
        self._command("move", "move", steps, period, release_ms)
        if block:
            while self.isBusy():
                time.sleep(0.01)

    def isBusy(self) -> bool:
        """Return True while the stepper motor is moving (read from the module)."""
        return bool(self._status() & 0x01)

    def hold(self) -> None:
        """Energize the stepper coils, so the motor holds its position."""
        self._require_stepper("hold")
        self._command("hold", "hold")

    def release(self) -> None:
        """Release the stepper coils, so the motor turns freely."""
        self._require_stepper("release")
        self._command("release", "release")

    def setHalfStep(self, enabled) -> None:
        """Set half step mode: twice the steps per revolution, smoother movement.

        Args:
            enabled (bool): True for half step, False for full step.
        """
        if not isinstance(enabled, bool):
            raise ModulinoError("enabled must be a boolean")
        self._require_stepper("set half step")
        self._command("set half step", "setHalfStep", enabled)

    def isHalfStep(self) -> bool:
        """Return True if half step mode is enabled (read from the module)."""
        return bool(self._status() & 0x04)

    def setOnMoveCompleteCallback(self, callback) -> None:
        """Call a callback each time a stepper move completes.

        Set it before calling `move()`. If a new move starts before the previous one completes,
        the callback is called once, at the end of the last move.

        Args:
            callback (Callable[[], None]): Called with no arguments:
                  def callback() -> None:
        """
        self._require_stepper("set move complete callback")
        try:
            # The MCU notifies 3 integers: busy (always 0) and two unused values
            Bridge.provide(f"{self.name}_callback", lambda _busy, _unused1, _unused2: callback())
            Bridge.notify("poll_subscribe", self.name, 20)
        except Exception:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnMoveCompleteCallback(self) -> None:
        """Remove the move complete callback."""
        try:
            Bridge.notify("poll_unsubscribe", self.name)
        except Exception:
            raise ModulinoError(f"cannot remove callback for modulino {self.name}")

class Modulino_LEDMatrix(Modulino):
    """Class to handle a Modulino LED Matrix (12x8 LEDs).

    Frames are `Frame` objects of 12x8 pixels (`MODULINO_MATRIX`), create them with `designer()`.
    The matrix is monochrome by default, `setGrayscale(True)` enables 16 brightness levels.
    The LED matrix is not supported on a Modulino Hub: connect it to the main bus.
    """
    MAX_FRAMES = 20
    """Maximum number of frames of an animation (MODULINO_LED_MATRIX_MAX_FRAMES of the sketch library)."""

    _MODE_MONOCHROME = 0
    _MODE_GRAYSCALE = 1
    _GRAYSCALE_LEVELS = 16

    def __init__(self, i2c_address = 0x39) -> None:
        """Initialize the Modulino_LEDMatrix with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the LED matrix. Defaults to 0x39.
        """
        if not isinstance(i2c_address, int) or not 0x08 <= i2c_address <= 0x77:
            raise ModulinoError("i2c_address must be an integer between 0x08 and 0x77")
        super().__init__("led_matrix", i2c_address)
        self._grayscale = False
        self._on_animation_end = None

    @override
    def start(self) -> None:
        """Start the LED matrix, in monochrome mode."""
        try:
            ok = Bridge.call(f"{self.name}:begin")
        except:
            raise ModulinoError(f"cannot start modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot start modulino {self.name}")
        self._grayscale = False
        try:
            # Always provided: the MCU notifies (3 unused integers) the end of every animation without loop
            Bridge.provide(f"{self.name}_callback", lambda _unused0, _unused1, _unused2: self._animation_ended())
        except Exception:
            raise ModulinoError(f"cannot start modulino {self.name}")

    def _animation_ended(self) -> None:
        callback = self._on_animation_end
        if callback is not None:
            callback()

    @override
    def stop(self) -> None:
        """Stop the LED matrix."""
        try:
            Bridge.call("delete_modulino", self.name)
        except:
            raise ModulinoError(f"cannot delete modulino {self.name}")

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

    def setGrayscale(self, enabled) -> None:
        """Set the display mode.

        Args:
            enabled (bool): True for grayscale (16 brightness levels), False for monochrome (on/off).
        """
        if not isinstance(enabled, bool):
            raise ModulinoError("enabled must be a boolean")
        mode = self._MODE_GRAYSCALE if enabled else self._MODE_MONOCHROME
        try:
            ok = Bridge.call(f"{self.name}:setMode", mode)
        except:
            raise ModulinoError(f"cannot set the mode of modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot set the mode of modulino {self.name}")
        self._grayscale = enabled

    def isGrayscale(self) -> bool:
        """Return True if the matrix is in grayscale mode."""
        return self._grayscale

    def _check_frame(self, frame) -> None:
        if not isinstance(frame, Frame):
            raise ModulinoError("frame must be a Frame")
        if (frame.width, frame.height) != MODULINO_MATRIX:
            raise ModulinoError(f"frame must be {MODULINO_MATRIX.width}x{MODULINO_MATRIX.height}, got {frame.width}x{frame.height}")

    def draw(self, frame) -> None:
        """Show a frame on the matrix. A running animation is stopped.

        In monochrome mode the pixels with a value greater than 0 are on. In grayscale mode the values are
        rescaled from the frame brightness levels to the 16 levels of the matrix.

        Args:
            frame (Frame): A 12x8 frame, with any number of brightness levels.
        """
        self._check_frame(frame)
        pixels = frame.to_bytes(self._GRAYSCALE_LEVELS) if self._grayscale else frame.to_bytes()
        try:
            ok = Bridge.call(f"{self.name}:draw", pixels)
        except:
            raise ModulinoError(f"cannot draw on modulino {self.name}")
        if not ok:
            raise ModulinoError(f"cannot draw on modulino {self.name}")

    def clear(self) -> None:
        """Turn off every LED. A running animation is stopped."""
        self.draw(Frame.empty(MODULINO_MATRIX))

    def playAnimation(self, frames, loop = False) -> None:
        """Play an animation on the matrix, the frames are timed by the board.

        The frames are shown on/off (a pixel is on when its value is greater than 0), also in grayscale mode.
        The call returns right away. A new animation replaces the running one, `draw()` and `clear()` stop it.
        Without loop, the callback set with `setOnAnimationEndCallback()` is called when the animation ends.

        Args:
            frames (Sequence[tuple]): `(frame, duration_ms)` or `(frame, duration_ms, label)` items, as taken
                by `sequence_to_c()`: 12x8 frames shown for `duration_ms` milliseconds each, up to `MAX_FRAMES`.
            loop (bool, optional): If True, repeat the animation until it is stopped. Defaults to False.
        """
        if not frames:
            raise ModulinoError("frames must not be empty")
        if len(frames) > self.MAX_FRAMES:
            raise ModulinoError(f"too many frames: {len(frames)}, the maximum is {self.MAX_FRAMES}")
        if not isinstance(loop, bool):
            raise ModulinoError("loop must be a boolean")
        entries = []
        for item in frames:
            frame, duration_ms = item[0], item[1]
            self._check_frame(frame)
            if not isinstance(duration_ms, int) or not 0 <= duration_ms <= 0x7FFFFFFF:
                raise ModulinoError("duration_ms must be a non-negative integer")
            entries.append([*frame.to_words(), duration_ms])   # 3 words + duration
        try:
            Bridge.call(f"{self.name}:stop_animation")   # drops the frames of the previous animation
            for entry in entries:
                Bridge.notify(f"{self.name}:load_frame", entry)
            count = Bridge.call(f"{self.name}:play_animation", loop)
        except:
            raise ModulinoError(f"cannot play animation on modulino {self.name}")
        if count != len(entries):
            raise ModulinoError(f"cannot play animation on modulino {self.name}: {count} of {len(entries)} frames loaded")
        try:
            # Starts the playback: the MCU shows the first frame right away, then sets the
            # polling period to the duration of each frame and unsubscribes at the end.
            Bridge.notify("poll_subscribe", self.name, 1)
        except:
            raise ModulinoError(f"cannot play animation on modulino {self.name}")

    def stopAnimation(self) -> None:
        """Stop the animation, the current frame stays on the matrix."""
        try:
            Bridge.call(f"{self.name}:stop_animation")
        except:
            raise ModulinoError(f"cannot stop animation on modulino {self.name}")

    def setOnAnimationEndCallback(self, callback) -> None:
        """Call a callback each time an animation without loop ends, after the duration of its last frame.

        An animation stopped by `stopAnimation()`, `draw()`, `clear()` or a new `playAnimation()` does not call it.
        The callback runs on the bridge thread: it must not call the methods of the Modulini, signal the app instead
        (e.g. with a `threading.Event`).

        Args:
            callback (Callable[[], None]): Called with no arguments:
                  def callback() -> None:
        """
        if not callable(callback):
            raise ModulinoError("callback must be callable")
        self._on_animation_end = callback

    def removeOnAnimationEndCallback(self) -> None:
        """Remove the animation end callback."""
        self._on_animation_end = None
