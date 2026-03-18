# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Chat with a Local LLM"
# EXAMPLE_REQUIRES = "Models must be downloaded and available locally."

from arduino.app_bricks.vlm import VisionLanguageModel
from arduino.app_utils import App

vlm = VisionLanguageModel(base_url="http://192.168.1.218:9001/v1", model="qwen3-vl-4b")

print(vlm.chat("Describe the image.", images=["chair.jpg"]))
