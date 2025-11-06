# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Record audio to a file"
# EXAMPLE_REQUIRES = "Requires a connected microphone"
from arduino.app_peripherals.microphone import Microphone


mic = Microphone()
mic.start()
mic.record_file(5, "/assets/recording.wav")  # Record 5 seconds of audio
mic.stop()

# Otherwise, you can use contexts
with Microphone() as mic:
    mic.record_file(5, "/assets/recording.wav")
