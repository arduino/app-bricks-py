# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "MQTT topic subscribe and read messages"
# EXAMPLE_REQUIRES = "Requires an MQTT broker running locally on port 1883."
from arduino.app_bricks.mqtt import MQTT
from arduino.app_utils import App

client = MQTT(broker_address="127.0.0.1", broker_port=1883, username="admin", password="password", topics=["test/topic"])

App.run()
