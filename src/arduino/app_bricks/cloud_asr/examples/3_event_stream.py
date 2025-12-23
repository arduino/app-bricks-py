# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Sends audio from microphone and receives all the streaming events"
# EXAMPLE_REQUIRES = "Requires an USB microphone connected to the Arduino board."
from arduino.app_bricks.cloud_asr import CloudASR
from arduino.app_utils import App

cloud_asr = CloudASR(
    api_key="YOUR_API_KEY",  # Replace with your actual API key
)
cloud_asr.on_update(lambda resp: print(f"{resp['event']}: {resp['data']}"))

App.run()
