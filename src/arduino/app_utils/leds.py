# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils.logger import Logger

logger = Logger(__name__)


class Leds:
    """
    A utility class for controlling LED colors on Arduino hardware.

    This class provides static methods to control two RGB LEDs by writing to system
    brightness files. LED1 and LED2 can be controlled directly by the MPU, while
    LED3 and LED4 require MCU control via Bridge.

    LED1 (red:user, green:user, blue:user) is a pure user LED: its triggers start as
    "none" and Python saves/restores both trigger and brightness on shutdown.

    LED2 (red:panic, green:wlan, blue:bt) has system functions: red:panic is used
    by the kernel panic handler, green:wlan by the WiFi driver (phy0tx), blue:bt
    by bluetoothd (bluetooth-power). arduino-app-cli sets their triggers to "none"
    before the container starts and to "default" when it stops — but "default" does
    not restore the original hardware behavior. Python therefore explicitly restores
    the known correct triggers for LED2 on shutdown, along with brightness=0, so
    that the system services can resume control.

    Attributes:
        _led_ids (list): List of supported LED IDs [1, 2].
        _led1_brightness_files (list): System file paths for LED1 RGB channels.
        _led2_brightness_files (list): System file paths for LED2 RGB channels.

    Methods:
        set_led1_color(r, g, b): Set the RGB color state for LED1.
        set_led2_color(r, g, b): Set the RGB color state for LED2.
        restore_system_control(): Restore original triggers and brightness values.

    Example:
        >>> Leds.set_led1_color(True, False, True)  # LED1 shows magenta
        >>> Leds.set_led2_color(False, True, False)  # LED2 shows green
    """

    _led_ids = [1, 2]  # Supported LED IDs (Led 3 and 4 can't be controlled directly by MPU but only by MCU via Bridge)

    _led1_brightness_files = [
        "/sys/class/leds/red:user/brightness",
        "/sys/class/leds/green:user/brightness",
        "/sys/class/leds/blue:user/brightness",
    ]
    _led2_brightness_files = [
        "/sys/class/leds/red:panic/brightness",
        "/sys/class/leds/green:wlan/brightness",
        "/sys/class/leds/blue:bt/brightness",
    ]
    _led1_trigger_files = [
        "/sys/class/leds/red:user/trigger",
        "/sys/class/leds/green:user/trigger",
        "/sys/class/leds/blue:user/trigger",
    ]
    # Known correct system triggers for LED2 on Arduino UNO Q.
    # arduino-app-cli sets them to "none" on start and "default" on stop, but
    # "default" does not restore the original hardware behavior. Python restores
    # these explicitly on shutdown so system services can resume control.
    _led2_system_triggers = {
        "/sys/class/leds/red:panic/trigger": "none",
        "/sys/class/leds/green:wlan/trigger": "phy0tx",
        "/sys/class/leds/blue:bt/trigger": "bluetooth-power",
    }

    # Saved state: {brightness_file: original_brightness, trigger_file: original_trigger}
    _saved_state: dict[str, str] = {}

    @staticmethod
    def _read_file(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return None

    @staticmethod
    def _write_file(path, value):
        try:
            with open(path, "w") as f:
                f.write(f"{value}\n")
        except Exception as e:
            logger.warning(f"Error writing to {path}: {e}")

    @staticmethod
    def _read_trigger(trigger_file):
        """Read the active trigger (the one enclosed in brackets)."""
        content = Leds._read_file(trigger_file)
        if content:
            for part in content.split():
                if part.startswith("[") and part.endswith("]"):
                    return part[1:-1]
        return "none"

    @staticmethod
    def _acquire_led(brightness_files, trigger_files):
        """Save original state of the LED channels and disable their triggers.

        This must be called before writing to brightness files to ensure the
        system trigger is not fighting against manual writes.
        Only saves state the first time (subsequent calls are no-ops per channel).
        """
        for brightness_file, trigger_file in zip(brightness_files, trigger_files):
            if brightness_file in Leds._saved_state:
                continue  # Already acquired

            original_trigger = Leds._read_trigger(trigger_file)
            original_brightness = Leds._read_file(brightness_file) or "0"
            Leds._saved_state[brightness_file] = original_brightness
            Leds._saved_state[trigger_file] = original_trigger
            logger.debug(f"Saved state: {trigger_file}={original_trigger}, {brightness_file}={original_brightness}")

            if original_trigger != "none":
                Leds._write_file(trigger_file, "none")
                logger.debug(f"Disabled trigger for {trigger_file}")

    @staticmethod
    def _acquire_brightness_only(brightness_files):
        """Save original brightness of the LED channels without touching triggers.

        Used for LED2 (red:panic, green:wlan, blue:bt) whose triggers are already
        managed by arduino-app-cli on the host (sets to "none" on app start,
        restores on app stop). Python only needs to clear the brightness on shutdown.
        Only saves state the first time (subsequent calls are no-ops per channel).
        """
        for brightness_file in brightness_files:
            if brightness_file in Leds._saved_state:
                continue  # Already acquired
            original_brightness = Leds._read_file(brightness_file) or "0"
            Leds._saved_state[brightness_file] = original_brightness
            logger.debug(f"Saved brightness: {brightness_file}={original_brightness}")

    @staticmethod
    def restore_system_control():
        """Restore original trigger and brightness for all acquired LED channels.

        For LED1: restores both trigger and brightness to the values saved at
        first use (_acquire_led).
        For LED2: restores brightness to 0 and explicitly restores the known
        correct system triggers (_led2_system_triggers), overriding the "default"
        value written by arduino-app-cli on container stop so that system services
        (bluetoothd, WiFi driver) can resume LED control.
        """
        if not Leds._saved_state and not Leds._led2_system_triggers:
            return

        logger.info("Restoring system LED control...")

        # Restore LED1 triggers first, then brightness (from saved state)
        for key, value in Leds._saved_state.items():
            if key.endswith("/trigger"):
                Leds._write_file(key, value)
                logger.debug(f"Restored trigger: {key} -> {value}")

        for key, value in Leds._saved_state.items():
            if key.endswith("/brightness"):
                Leds._write_file(key, value)
                logger.debug(f"Restored brightness: {key} -> {value}")

        # Restore LED2 triggers to known correct system values.
        # This runs after arduino-app-cli has written "default", overriding it
        # so that bluetoothd and the WiFi driver can resume LED control.
        # Do NOT write brightness for LED2: the trigger driver manages it
        # autonomously (bluetooth-power sets brightness=1 when BT is on,
        # phy0tx flashes on WiFi TX). Writing brightness=0 would prevent
        # the driver from re-enabling the LED on the first event.
        for trigger_file, trigger_value in Leds._led2_system_triggers.items():
            Leds._write_file(trigger_file, trigger_value)
            logger.debug(f"Restored LED2 system trigger: {trigger_file} -> {trigger_value}")

        Leds._saved_state.clear()
        logger.info("System LED control restored")

    @staticmethod
    def set_led1_color(r: bool, g: bool, b: bool):
        Leds._acquire_led(Leds._led1_brightness_files, Leds._led1_trigger_files)
        Leds._write_file(Leds._led1_brightness_files[0], int(r))
        Leds._write_file(Leds._led1_brightness_files[1], int(g))
        Leds._write_file(Leds._led1_brightness_files[2], int(b))

    @staticmethod
    def set_led2_color(r: bool, g: bool, b: bool):
        # LED2 triggers (red:panic, green:wlan, blue:bt) are managed by arduino-app-cli on the
        # host: it sets them to "none" before the container starts and restores them after stop.
        # Python only manages brightness so the color is cleared on shutdown.
        Leds._acquire_brightness_only(Leds._led2_brightness_files)
        Leds._write_file(Leds._led2_brightness_files[0], int(r))
        Leds._write_file(Leds._led2_brightness_files[1], int(g))
        Leds._write_file(Leds._led2_brightness_files[2], int(b))
