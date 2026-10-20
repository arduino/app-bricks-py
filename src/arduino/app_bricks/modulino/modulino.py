from typing import override

from arduino.app_utils import brick, Logger, Bridge

logger = Logger("Modulino")
class ModulinoError(Exception):
    """Raised when a Modulino fails."""

@brick
class Modulino:
    """Base class for all Modulino modules."""
    def __init__(self, type, i2c_address):
        """Initialize a Modulino with the given type and I2C address.

        Args:
            type (str): The type of the Modulino.
            i2c_address (int): The I2C address of the Modulino.
        """
        self.type = type
        self.i2caddress = i2c_address
        self.name = f"{type}_{i2c_address:x}"
        Bridge.call("create_modulino", type, i2c_address)

    def start(self):
        logger.info("Starting Modulino")

    def stop(self):
        logger.info("Stopping Modulino")


class Modulino_Thermo(Modulino):
    """Class to handle a Modulino 'thermo' for temperature and humidity sensing."""
    def __init__(self, i2c_address = 0x44) -> None:
        """Initialize a Modulino_Thermo with the given I2C address.

        Args:
            i2c_address (int): The I2C address of the Modulino_Thermo.
        """
        super().__init__("thermo", i2c_address )

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
            Bridge.call("delete_modulino", self.i2caddress)
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
    def __init__(self, i2c_address = 0x6A) -> None:
        super().__init__("movement", i2c_address )

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
            Bridge.call("delete_modulino", self.i2caddress)
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
        Bridge.notify("poll_subscribe", self.i2caddress, ms)

    def stopSample(self) -> None:
        """Stop sampling movement data."""
        Bridge.notify("poll_unsubscribe", self.i2caddress)

class Modulino_Buttons(Modulino):
    """Class to handle a Modulino Buttons."""
    def __init__(self, i2c_address = 0x3E) -> None:
        """Initialize the Modulino_Buttons with the specified I2C address."""
        super().__init__("buttons", i2c_address )

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
            Bridge.call("delete_modulino", self.i2caddress)
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
            Bridge.notify("poll_subscribe", self.i2caddress, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for button state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.i2caddress)
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
    def __init__(self, i2c_address = 0x2C) -> None:
        """Initialize the Modulino_Joystick with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the joystick modulino. Defaults to 0x2C.
        """
        super().__init__("joystick", i2c_address )
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
            Bridge.call("delete_modulino", self.i2caddress)
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
            Bridge.notify("poll_subscribe", self.i2caddress, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for joystick state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.i2caddress)
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
    def __init__(self, i2c_address = 0xFF) -> None:
        """Initialize the Modulino_Knob with the given I2C address.

        Args:
            i2c_address (int, optional): The I2C address of the knob modulino. Defaults to 0xFF.
        """
        super().__init__("knob", i2c_address )

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
            Bridge.call("delete_modulino", self.i2caddress)
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
            Bridge.notify("poll_subscribe", self.i2caddress, 10)
        except:
            raise ModulinoError(f"cannot set callback for modulino {self.name}")

    def removeOnChangeCallback(self) -> None:
        """Remove the callback function for knob state changes."""
        try:
            Bridge.notify("poll_unsubscribe", self.i2caddress)
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
