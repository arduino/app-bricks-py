# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Record audio for a duration"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
import numpy as np
from arduino.app_peripherals.microphone import Microphone


mic = Microphone()
mic.start()
audio: np.ndarray = mic.record(5)  # Record 5 seconds of audio
# You can process the audio here if needed, e.g save it
mic.stop()

# Otherwise, you can use contexts
with Microphone() as mic:
    audio: np.ndarray = mic.record(5)
