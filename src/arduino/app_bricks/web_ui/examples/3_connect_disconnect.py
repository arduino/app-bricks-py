# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Print connections and disconnections"
from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI


ui = WebUI()
ui.on_connect(lambda sid: print(f"{sid} has connected!"))
ui.on_disconnect(lambda sid: print(f"{sid} has disconnected!"))

App.run()  # This will block until the app is stopped
