# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import Logger
import os

logger = Logger(__name__)


class Leds:
    """
    A utility class for controlling LED colors on Arduino hardware.

    This class provides static methods to control two RGB LEDs by writing to system
    brightness files. LED1 and LED2 can be controlled directly by the MPU, while
    LED3 and LED4 require MCU control via Bridge.

    Attributes:
        _led_ids (list): List of supported LED IDs [1, 2].
        _led1_brightness_files_legacy (list): Legacy file paths for LED1 brightness control.
        _led2_brightness_files_legacy (list): Legacy file paths for LED2 brightness control.
        _led1_brightness_file (list): Compatible file paths for LED1 brightness control.
        _led2_brightness_file (list): Compatible file paths for LED2 brightness control.

    Methods:
        set_led1_color(r, g, b): Set the RGB color state for LED1.
        set_led2_color(r, g, b): Set the RGB color state for LED2.

    Example:
        >>> Leds.set_led1_color(True, False, True)  # LED1 shows magenta
        >>> Leds.set_led2_color(False, True, False)  # LED2 shows green
    """

    _led_ids = [1, 2]  # Supported LED IDs (Led 3 and 4 can't be controlled directly by MPU but only by MCU via Bridge)

    _led1_brightness_files_legacy = [
        "/sys/class/leds/red:user/brightness",
        "/sys/class/leds/green:user/brightness",
        "/sys/class/leds/blue:user/brightness",
    ]
    _led2_brightness_files_legacy = [
        "/sys/class/leds/red:panic/brightness",
        "/sys/class/leds/green:wlan/brightness",
        "/sys/class/leds/blue:bt/brightness",
    ]

    _led1_brightness_files = [
        "/dev/leds/builtin/led1_r/brightness",
        "/dev/leds/builtin/led1_g/brightness",
        "/dev/leds/builtin/led1_b/brightness",
    ]
    _led2_brightness_files = [
        "/dev/leds/builtin/led2_r/brightness",
        "/dev/leds/builtin/led2_g/brightness",
        "/dev/leds/builtin/led2_b/brightness",
    ]

    @staticmethod
    def _write_led_file(led_file, value: bool):
        try:
            with open(led_file, "w") as f:
                f.write(f"{int(value)}\n")
        except Exception as e:
            logger.exception(f"Error writing to {led_file}: {e}")

    @staticmethod
    def set_led1_color(r: bool, g: bool, b: bool):
        # check if /dev/leds/builtin/led1_r exists, if yes use compatible files, otherwise use legacy files
        if all(os.path.exists(f) for f in Leds._led1_brightness_files):
            Leds._write_led_file(Leds._led1_brightness_files[0], r)
            Leds._write_led_file(Leds._led1_brightness_files[1], g)
            Leds._write_led_file(Leds._led1_brightness_files[2], b)
        elif all(os.path.exists(f) for f in Leds._led1_brightness_files_legacy):
            Leds._write_led_file(Leds._led1_brightness_files_legacy[0], r)
            Leds._write_led_file(Leds._led1_brightness_files_legacy[1], g)
            Leds._write_led_file(Leds._led1_brightness_files_legacy[2], b)
        else:
            raise FileNotFoundError("No compatible LED files found for LED1.")

    @staticmethod
    def set_led2_color(r: bool, g: bool, b: bool):
        # check if /dev/leds/builtin/led2_r exists, if yes use compatible files, otherwise use legacy files
        if all(os.path.exists(f) for f in Leds._led2_brightness_files):
            Leds._write_led_file(Leds._led2_brightness_files[0], r)
            Leds._write_led_file(Leds._led2_brightness_files[1], g)
            Leds._write_led_file(Leds._led2_brightness_files[2], b)
        elif all(os.path.exists(f) for f in Leds._led2_brightness_files_legacy):
            Leds._write_led_file(Leds._led2_brightness_files_legacy[0], r)
            Leds._write_led_file(Leds._led2_brightness_files_legacy[1], g)
            Leds._write_led_file(Leds._led2_brightness_files_legacy[2], b)
        else:
            raise FileNotFoundError("No compatible LED files found for LED2.")
