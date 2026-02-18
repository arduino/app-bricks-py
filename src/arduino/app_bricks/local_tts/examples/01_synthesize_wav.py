# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Synthesize speech audio from text and save as WAV file"

from arduino.app_bricks.local_tts import LocalTTS

tts = LocalTTS()

wav = tts.synthesize_wav("Hello, Arduino world!")
with open("synthesized_speech.wav", "wb") as f:
    f.write(wav)
