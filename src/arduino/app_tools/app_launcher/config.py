# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The settings of the launcher, read before it loads: the app starts first and needs them too."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

# The packages the apps spend their startup importing, the backends of the peripherals named one by one, since their
# packages load them only on use
DEFAULT_PRELOAD = ",".join((
    # The library reads app.yaml and the brick configs with it when an app builds its bricks
    "yaml",
    "arduino.app_utils",
    "arduino.app_peripherals.camera.v4l_camera",
    "arduino.app_peripherals.camera.csi_camera",
    "arduino.app_peripherals.camera.ip_camera",
    "arduino.app_peripherals.camera.websocket_camera",
    "arduino.app_peripherals.microphone.alsa_microphone",
    "arduino.app_peripherals.microphone.websocket_microphone",
    "arduino.app_peripherals.speaker",
    "arduino.app_bricks.web_ui",
))
DEFAULT_SOCKET = "/run/arduino-app-cli/app-launcher.sock"


@dataclass(frozen=True)
class Config:
    """Launcher settings, from the environment."""

    preload: tuple[str, ...]
    socket_path: str
    stop_timeout_s: float
    app_dir: str
    """The app folder run.sh prepared before starting the launcher, the one it serves."""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        """Reads the settings.

        Raises:
            ValueError: when a value is malformed or APP_DIR is not set.
        """
        if not env.get("APP_DIR"):
            raise ValueError("APP_DIR is not set: the app folder to serve")
        return cls(
            preload=tuple(name.strip() for name in env.get("APP_LAUNCHER_PRELOAD", DEFAULT_PRELOAD).split(",") if name.strip()),
            socket_path=env.get("APP_LAUNCHER_SOCKET", DEFAULT_SOCKET),
            stop_timeout_s=float(env.get("APP_LAUNCHER_STOP_TIMEOUT_S", "5")),
            app_dir=os.path.normpath(env["APP_DIR"]),
        )
