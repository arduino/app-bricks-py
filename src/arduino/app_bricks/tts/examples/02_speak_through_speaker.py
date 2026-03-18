# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Speak text through a speaker"

from arduino.app_bricks.tts import TextToSpeech
from arduino.app_peripherals.speaker import Speaker


speaker = Speaker()
speaker.start()

tts = TextToSpeech()
tts.speak("Hello, Arduino world!", speaker=speaker)

speaker.stop()
