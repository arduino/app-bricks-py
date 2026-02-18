# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Speak text through a speaker"

from arduino.app_bricks.local_tts import LocalTTS
from arduino.app_peripherals.speaker import Speaker


speaker = Speaker()
speaker.start()

tts = LocalTTS()
tts.speak("Hello, Arduino world!", speaker=speaker)

speaker.stop()
